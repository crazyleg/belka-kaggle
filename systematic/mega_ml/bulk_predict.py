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
from model import GATBasedMolecularGraphNeuralNetworkV2, GATBasedMolecularGraphNeuralNetwork194
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import xgboost as xgb
import numpy as np
import os

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
test_unique_dataset = FileMegaSet('/mnt/fastssd/datasets/test/')
test_unique_loader = DataLoader(test_unique_dataset, batch_size=1, num_workers=5, shuffle=False)

model = GATBasedMolecularGraphNeuralNetwork194()
DIR = '/home/anonymous/belka/systematic/mega_ml/final/domain_research/61/models/'
DIR_TO_SAVE = '/home/anonymous/belka/systematic/mega_ml/final/domain_research/61/predictions/'
for file in os.listdir(DIR):

    model.load_state_dict(torch.load(DIR+file))

    model=model.to(device)
    ids = []
    scores = []
    def test(ds):
        model.eval()
        test_outputs = []
        test_smiles = []
        with torch.no_grad():
            pbar = tqdm(enumerate(ds))
            for batch_idx, data in pbar: 
                data = data.to(device)
                output, _emb = model(data)  
                test_outputs.extend(output.detach().sigmoid().squeeze().tolist())
                test_smiles.extend(data.smiles[0])
        
        return test_smiles, test_outputs,
    ids, scores = test(test_unique_loader)

    # Create a DataFrame
    df = pd.DataFrame({
        'id': ids,
        'binds': scores
    })

    # Save the DataFrame to a CSV file
    df.to_csv(DIR_TO_SAVE + file + '.csv', index=False)  # index=False means do not write row numbers