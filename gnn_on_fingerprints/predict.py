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
from dataset import   APIBasedMoleculePredictDataset
from model import GATBasedMolecularGraphNeuralNetwork
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 

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
CONNECTION_STRING = "mongodb://anonymous-server.local/belka"

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    if not hasattr(dataset, 'client'):
        dataset.client = MongoClient(CONNECTION_STRING)
    dataset.db = dataset.client['belka']
    dataset.raw_data = dataset.db['full_molecules_fingerprints_test']

# Setup CUDA
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# Example of setting up DataLoader with prefetching
train_dataset = APIBasedMoleculePredictDataset()
train_loader = DataLoader(train_dataset, batch_size=4096, num_workers=10, worker_init_fn=worker_init_fn)


# Model

model = GATBasedMolecularGraphNeuralNetwork(in_channels=3, hidden_size = 64, num_gcn_layers=6, ).to(device)
model.load_state_dict(torch.load('model_best_unique.pth'))
model=model.to(device)
ids = []
scores = []
def test(ds):
    model.eval()
    with torch.no_grad():
         for batch_idx, (data, protein, target) in enumerate(tqdm(ds)):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x, edge_index, batch, protein, target = x.to(device), edge_index.to(device), batch.to(device), protein.to(device), target.to(device)
            output = model(x, edge_index, batch, protein)
            
            scores.extend(output.sigmoid().detach().squeeze().tolist())
            ids.extend(target.int().detach().squeeze().tolist())
    
    return ids, scores  

ids, scores = test(train_loader)

# Create a DataFrame
df = pd.DataFrame({
    'id': ids,
    'binds': scores
})

# Save the DataFrame to a CSV file
df.to_csv('output_gat.csv', index=False)  # index=False means do not write row numbers