import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import wandb
from pymongo import MongoClient
from tqdm.auto import tqdm
from dataset import  APIBasedMoleculeDataset
from model import MultiTowerModel
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
#         F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#         if self.reduction == 'mean':
#             return torch.mean(F_loss)
#         elif self.reduction == 'sum':
#             return torch.sum(F_loss)
#         else:
#             return F_loss   
class SampledPairwiseMarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0, samples_per_positive=5):
        super(SampledPairwiseMarginRankingLoss, self).__init__()
        self.margin = margin
        self.samples_per_positive = samples_per_positive

    def forward(self, scores, target):
        positive_scores = scores[target == 1]
        negative_scores = scores[target == 0]

        loss = 0
        for pos_score in positive_scores:
            # Randomly sample negatives
            neg_samples = negative_scores[torch.randint(len(negative_scores), (self.samples_per_positive,))]
            for neg_score in neg_samples:
                loss += F.relu(self.margin - pos_score + neg_score)

        return loss / (len(positive_scores) * self.samples_per_positive)
CONNECTION_STRING = "mongodb://anonymous-server.local/belka"
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    if not hasattr(dataset, 'client'):
        dataset.client = MongoClient(CONNECTION_STRING)
    dataset.db = dataset.client['belka']
    dataset.full_molecules = dataset.db['full_molecules']

# Initialize Weights & Biases
wandb.init(project="belka", settings=wandb.Settings(code_dir="./mlp_on_fingerprints"), tags=['dry_run'])

# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch Binary Classification')
parser.add_argument('--batch-size', type=int, default=2048*2, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4096, help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
args = parser.parse_args()

# Setup CUDA
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
# Example of setting up DataLoader with prefetching
train_dataset = APIBasedMoleculeDataset(dataset_factor=1,  batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True, num_workers=10, shuffle=True, worker_init_fn=worker_init_fn)


# Example of setting up DataLoader with prefetching
test_dataset = APIBasedMoleculeDataset(is_test=True, dataset_factor=1, batch_size=args.test_batch_size)
test_loader = DataLoader(test_dataset, num_workers=5, pin_memory=True, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn)

# Model
model = MultiTowerModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Loss function
criterion = SampledPairwiseMarginRankingLoss()
# criterion = FocalLoss()
def train(epoch,dl):
    model.train()
    pbar = tqdm(enumerate(dl))
    train_targets = []
    train_outputs = []
    for batch_idx, (tower1, tower2, tower3, tower4, tower5, tower6, tower7, tower8, tower9, protein, target) in pbar:
        tower1, tower2, tower3, tower4, tower5,  tower6, tower7, tower8, tower9, protein, target = tower1.to(device), tower2.to(device), tower3.to(device), tower4.to(device), tower5.to(device), tower6.to(device), tower7.to(device), tower8.to(device), tower9.to(device), protein.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(tower1, tower2, tower3, tower4, tower5,  tower6, tower7, tower8, tower9, protein)
        train_outputs.extend(output.detach().sigmoid().squeeze().tolist())
        train_targets.extend(target.tolist())
        loss = criterion(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            wandb.log({"loss": loss.item()})
        pbar.set_description(f'Train Epoch: {epoch} [{batch_idx * len(tower1)}/{len(dl.dataset)} ({100. * batch_idx / len(dl):.0f}%)]\tLoss: {loss.item():.6f}')
    flattened_list = [item for sublist in train_targets for item in sublist]
    test_average_precision = average_precision_score(flattened_list, train_outputs, average='micro')
    wandb.log({f"traing_map_micro": test_average_precision})

def test(epoch, dl ):
    model.eval()
    test_targets = []
    test_outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(dl))
        for batch_idx, (tower1, tower2, tower3, tower4, tower5, tower6, tower7, tower8, tower9, protein, target)  in pbar:
            tower1, tower2, tower3, tower4, tower5,  tower6, tower7, tower8, tower9, protein, target = tower1.to(device), tower2.to(device), tower3.to(device), tower4.to(device), tower5.to(device), tower6.to(device), tower7.to(device), tower8.to(device), tower9.to(device), protein.to(device), target.to(device)
            output = model(tower1, tower2, tower3, tower4, tower5,  tower6, tower7, tower8, tower9, protein)
            test_outputs.extend(output.sigmoid().squeeze().tolist())
            test_targets.extend(target.tolist())
            pbar.set_description(f'Test Epoch: {epoch} [{batch_idx * len(tower1)}/{len(dl.dataset)} ({100. * batch_idx / len(dl):.0f}%)]')
 
    flattened_list = [item for sublist in test_targets for item in sublist]
    test_average_precision = average_precision_score(flattened_list, test_outputs, average='micro')
    wandb.log({f"test_map_micro": test_average_precision})
    return test_average_precision

best_map_unique = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch, train_loader)
    test_accuracy_unique = test(epoch, test_loader)


    # Checkpoint model if current MAP is better
    if test_accuracy_unique > best_map_unique:
        # flattened_list = [item for sublist in test_targets for item in sublist]
        best_map_unique = test_accuracy_unique
        # results = pd.DataFrame()
        # results['outputs'] = test_outputs
        # results['targets'] = flattened_list 
        # results.to_csv('best_set.csv')
        wandb.run.summary["best_map_unique_micro"] = best_map_unique
        torch.save(model.state_dict(), 'model_best_unique.pth')
        wandb.save('model_best_unique.pth')