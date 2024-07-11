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
from dataset import   FileMegaSet
from model import GATBasedMolecularGraphNeuralNetworkV2, GATBasedMolecularGraphNeuralNetworkAndEmd
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
# Setup CUDA
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device('cpu')
#
# Example of setting up DataLoader with prefetching
test_unique_dataset = FileMegaSet('/mnt/fastssd/datasets/belka/test_unique')
test_unique_loader = DataLoader(test_unique_dataset, batch_size=1, num_workers=5, shuffle=False)


# Model
model = GATBasedMolecularGraphNeuralNetworkAndEmd(hidden_size = 64, num_gcn_layers=6).to(device)
model.load_state_dict(torch.load('/home/anonymous/belka/systematic/mega_ml/models/model_s1_random_0.71.pth'))
model=model.to(device)
scores = []
def test(ds):
    model.eval()
    test_outputs = []
    test_smiles = []
    train_targets = []
    embs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(ds))
        for batch_idx, data in pbar: 
            data = data.to(device)
            output, emb = model(data)  
            test_outputs.extend(output.detach().sigmoid().squeeze().tolist())
            embs.extend(emb.detach().squeeze().tolist())
            train_targets.extend(data.y.cpu().tolist())
    
    return test_outputs, embs, train_targets 
scores, embs, target = test(test_unique_loader)

# Create a DataFrame
df = pd.DataFrame({
    'binds': scores,
    'emb': embs,
    'target': target
})

# Save the DataFrame to a CSV file
df.to_csv('model_emb_s1_random_0.71_xgb.csv', index=False)  # index=False means do not write row numbers