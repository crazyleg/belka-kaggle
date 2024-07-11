import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import wandb
from pymongo import MongoClient
from tqdm.auto import tqdm
from dataset import  APIBasedMoleculeDataset
from model import GATBasedMolecularGraphNeuralNetwork
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F


CONNECTION_STRING = "mongodb://anonymous-server.local/belka"
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    if not hasattr(dataset, 'client'):
        dataset.client = MongoClient(CONNECTION_STRING)
    dataset.db = dataset.client['belka']
    dataset.full_molecules = dataset.db['full_molecules_fingerprints']

# Initialize Weights & Biases
wandb.init(project="belka", settings=wandb.Settings(code_dir="./gnn_on_fingerprints"), tags=['dry'])

# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch Binary Classification')
parser.add_argument('--batch-size', type=int, default=4096, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4096, help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
args = parser.parse_args()

# Setup CUDA
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Example of setting up DataLoader with prefetching
train_dataset = APIBasedMoleculeDataset( dataset_factor=0.1)
train_loader = DataLoader(train_dataset, batch_size=4096, num_workers=25, shuffle=True, worker_init_fn=worker_init_fn)


# Example of setting up DataLoader with prefetching
test_dataset = APIBasedMoleculeDataset( is_test=True, dataset_factor=0.3)
test_loader = DataLoader(test_dataset,  batch_size=4096, num_workers=25, shuffle=False, worker_init_fn=worker_init_fn)

# Model
model = GATBasedMolecularGraphNeuralNetwork(in_channels=3, hidden_size = 64, num_gcn_layers=6, ).to(device)
optimizer = optim.RAdam(model.parameters(), lr=args.lr)

# Loss function
criterion = nn.BCEWithLogitsLoss()
# criterion = FocalLoss()
def train(epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader))
    train_targets = []
    train_outputs = []
    for batch_idx, (data, protein, target) in pbar: 
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x, edge_index, batch, protein, target = x.to(device), edge_index.to(device), batch.to(device), protein.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(x, edge_index, batch, protein)       
        loss = criterion(output.squeeze(), target.squeeze().float())
        loss.backward()
        optimizer.step()
        train_outputs.extend(output.detach().sigmoid().squeeze().tolist())
        train_targets.extend(target.tolist())
        
        if batch_idx % args.log_interval == 0:
            wandb.log({"loss": loss.item()})
            pbar.set_description(f'Train Epoch: {epoch} [{batch_idx * len(target)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    test_average_precision = average_precision_score(train_targets, train_outputs, average='micro')
    wandb.log({f"traing_map_micro": test_average_precision})

def test(name, ds):
    model.eval()
    test_targets = []
    test_outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(ds))
        for batch_idx, (data, protein, target) in pbar:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x, edge_index, batch, protein, target = x.to(device), edge_index.to(device), batch.to(device), protein.to(device), target.to(device)
            output = model(x, edge_index, batch, protein)
            test_outputs.extend(output.detach().sigmoid().squeeze().tolist())
            test_targets.extend(target.tolist())
    
    test_average_precision = average_precision_score(test_targets, test_outputs, average='micro')
    wandb.log({f"test_map_micro_{name}": test_average_precision})
    return test_average_precision

best_map_unique = 0

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_accuracy_unique = test("unique", test_loader)

    # Checkpoint model if current MAP is better
    if test_accuracy_unique > best_map_unique:
        best_map_unique = test_accuracy_unique
        wandb.run.summary["best_map_unique_micro"] = best_map_unique
        torch.save(model.state_dict(), 'model_best_unique.pth')
        wandb.save('model_best_unique.pth')
   