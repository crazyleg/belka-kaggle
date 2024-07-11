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
from dataset import  SubMoleculeDatasetMongoBatched
from model import   MultiTowerSubmoleculeModel
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle 
import random
class MinMaxPairwiseMarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MinMaxPairwiseMarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, target):
        # Extract scores for positives and negatives
        positive_scores = scores[target == 1]
        negative_scores = scores[target == 0]

        # Initialize the loss
        loss = 0.0

        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            # No positives or no negatives in the batch, can't compute this loss
            return loss

        # Compute the max of negative scores
        max_negative_score = negative_scores.max()

        # Compute loss for each positive score against the max negative score
        for pos_score in positive_scores:
            loss += F.relu(self.margin - pos_score + max_negative_score)

        return loss / positive_scores.numel()
CONNECTION_STRING = "mongodb://anonymous-server.local/belka"
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    if not hasattr(dataset, 'client'):
        dataset.client = MongoClient(CONNECTION_STRING)
    dataset.db = dataset.client['belka']
    dataset.raw_data = dataset.db['train_metadata']
    import msgpack
    with open('submolecules_details.msgpack', 'rb') as f:
        dataset.submolecules_data = {x['SMILES']:x for x in msgpack.load(f)}

# Initialize Weights & Biases
wandb.init(project="belka", settings=wandb.Settings(code_dir="./mlp_on_fingerprints"), tags=['submolecules'])

# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch Binary Classification')
parser.add_argument('--batch-size', type=int, default=2048*2, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4096, help='input batch size for testing (default: 1000)')
parser.add_argument('--num_jobs', type=int, default=25, help='num jobs')
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

# Example of setting up DataLoader with prefetching
train_dataset = SubMoleculeDatasetMongoBatched('belka', dataset_factor=0.1,  batch_size=args.batch_size)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=args.num_jobs, shuffle=True, worker_init_fn=worker_init_fn)


# Example of setting up DataLoader with prefetching
test_dataset_unique = SubMoleculeDatasetMongoBatched('belka', is_test_unique=True, dataset_factor=0.1,batch_size=args.test_batch_size)
test_loader_unique = DataLoader(test_dataset_unique, num_workers=args.num_jobs, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn)

test_dataset_random = SubMoleculeDatasetMongoBatched('belka', is_test_random=True, dataset_factor=0.1,batch_size=args.test_batch_size)
test_loader_random = DataLoader(test_dataset_random, num_workers=args.num_jobs, batch_size=1, shuffle=True, worker_init_fn=worker_init_fn )

# Model
model = MultiTowerSubmoleculeModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Loss function
criterion = nn.BCEWithLogitsLoss()
criterion = MinMaxPairwiseMarginRankingLoss()
# criterion = FocalLoss()
def train(epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader))
    train_targets = []
    train_outputs = []
    for batch_idx, (tower1, tower2, tower3, target) in pbar:
        tower1, tower2, tower3, target = tower1.to(device), tower2.to(device), tower3.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(tower1, tower2, tower3)
        train_outputs.extend(output.detach().sigmoid().squeeze().tolist())
        train_targets.extend(target.tolist())
        loss = MinMaxPairwiseMarginRankingLoss(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            wandb.log({"loss": loss.item()})
            pbar.set_description(f'Train Epoch: {epoch} [{batch_idx * len(tower1)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    flattened_list = [item for sublist in train_targets for item in sublist]
    test_average_precision = average_precision_score(flattened_list, train_outputs, average='micro')
    wandb.log({f"traing_map_micro": test_average_precision})

def test(name, ds):
    model.eval()
    test_targets = []
    test_outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(ds))
        for batch_idx, (tower1, tower2, tower3, target) in pbar:
            tower1, tower2, tower3,target = tower1.to(device), tower2.to(device), tower3.to(device), target.to(device)
            output = model(tower1, tower2, tower3)
            test_outputs.extend(output.sigmoid().squeeze().tolist())
            test_targets.extend(target.tolist())
    
    flattened_list = [item for sublist in test_targets for item in sublist]
    test_average_precision = average_precision_score(flattened_list, test_outputs, average='micro')
    wandb.log({f"test_map_micro_{name}": test_average_precision})
    return test_average_precision

best_map_unique = 0
best_map_random = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_accuracy_unique = test("unique", test_loader_unique)
    test_accuracy_random = test('random', test_loader_random)

    # Checkpoint model if current MAP is better
    if test_accuracy_unique > best_map_unique:
        best_map_unique = test_accuracy_unique
        wandb.run.summary["best_map_unique_micro"] = best_map_unique
        torch.save(model.state_dict(), 'model_best_unique.pth')
        wandb.save('model_best_unique.pth')
    if test_accuracy_random > best_map_random:
        best_map_random = test_accuracy_random
        wandb.run.summary["best_map_random_micro"] = best_map_random
        torch.save(model.state_dict(), 'model_best_random.pth')
        wandb.save('model_best_random.pth')