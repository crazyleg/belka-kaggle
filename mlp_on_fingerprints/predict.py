import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import wandb
from pymongo import MongoClient
from tqdm.auto import tqdm
from dataset import  APIBasedMoleculeTestDataset
from model import  MultiTowerModel
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np 

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
    dataset.full_molecules_test = dataset.db['full_molecules_test']

# Setup CUDA
# use_cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = True 
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# Example of setting up DataLoader with prefetching
train_dataset = APIBasedMoleculeTestDataset()
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=3, worker_init_fn=worker_init_fn)


# Model

model = MultiTowerModel()
model.load_state_dict(torch.load('model_best_unique.pth'))
model=model.to(device)
f = open('output_mlp_attention_pairwise.csv','w')
f.write('id,binds\n') 
f.close()
print(torch.multiprocessing.get_sharing_strategy())
torch.multiprocessing.set_sharing_strategy('file_system')
def test(ds):

    model.eval()
    current_index = 0
    with torch.no_grad():
        for batch_idx, (tower1, tower2, tower3, tower4, tower5, tower6, tower7, tower8, tower9, protein, target) in enumerate(tqdm(ds)):
            tower1, tower2, tower3, tower4, tower5,  tower6, tower7, tower8, tower9, protein = tower1.to(device), tower2.to(device), tower3.to(device), tower4.to(device), tower5.to(device), tower6.to(device), tower7.to(device), tower8.to(device), tower9.to(device), protein.to(device)
            output = model(tower1, tower2, tower3, tower4, tower5,  tower6, tower7, tower8, tower9, protein)
            
            target = target[0].tolist()
            score = output.sigmoid().cpu().tolist()
            f = open('output_mlp_attention_pairwise.csv','a')
            for id, score in zip(target, score):
                f.write(f'{id},{score}\n') 
            f.close()

test(train_loader)