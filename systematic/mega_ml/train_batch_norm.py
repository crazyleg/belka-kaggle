import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torchvision import transforms
from ranger import Ranger  # this is from ranger.py
import torch.nn.functional as F
import wandb
from pymongo import MongoClient
from tqdm.auto import tqdm
from dataset import  FileMegaSet
from model import GATBasedMolecularGraphNeuralNetworkV3, GATBasedMolecularGraphNeuralNetworkAlt
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import GradScaler, autocast
def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_model(model):
    # Freeze all layers
    set_requires_grad(model.GAT, False)
    # Only the final MLPs are trainable
    set_requires_grad(model.fc1, True)
    set_requires_grad(model.fc2, True)
    
# Initialize Weights & Biases
wandb.init(project="belka", settings=wandb.Settings(code_dir="./systematic/mega_ml"), tags=['GAT_naive','split1','research','ecfp'])
TRAIN_FILE = '/mnt/fastssd/datasets/belka/train'
TEST_FILE = '/mnt/fastssd/datasets/test'
TEST_RANDOM_FILE = '/mnt/fastssd/datasets/belka/test_random'
TEST_UNIQUE_FILE = '/mnt/fastssd/datasets/belka/test_unique'
# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch Binary Classification')
parser.add_argument('--batch-size', type=int, default=4096, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4096, help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
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
train_dataset = FileMegaSet(TEST_FILE)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=10, shuffle=True)

# Example of setting up DataLoader with prefetching
test_random_dataset = FileMegaSet(TEST_RANDOM_FILE)
# test_random_dataset = FileMegaSet(TRAIN_FILE)
test_random_loader = DataLoader(test_random_dataset, batch_size=1, num_workers=5, shuffle=False)

# Example of setting up DataLoader with prefetching
test_unique_dataset = FileMegaSet(TEST_UNIQUE_FILE)
# test_unique_dataset = FileMegaSet(TRAIN_FILE)
test_unique_loader = DataLoader(test_unique_dataset, batch_size=1, num_workers=5, shuffle=False)

# Model
model = GATBasedMolecularGraphNeuralNetworkAlt().to(device)
model.load_state_dict(torch.load('/home/anonymous/belka/systematic/mega_ml/models/tmp/267-e5.pth'))
optimizer = Ranger(model.parameters(), lr = args.lr)
# q = torch.load('pretrained_gnn_masked.pth')
# model.load_state_dict(q)
# Define the LR scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15775//3, T_mult=2, eta_min=1e-6)
# freeze_model(model)
# Loss function
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.FocalLoss()
eps = 0.1
# Initialize GradScaler
# focal_loss = torch.hub.load(
# 	'adeelh/pytorch-multi-class-focal-loss',
# 	model='FocalLoss',
# 	alpha=torch.tensor([.75, .25]),
# 	gamma=2,
# 	reduction='mean',
# 	force_reload=False
# )
# criterion = focal_loss.to(device)
def train(epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar: 
        data = data.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            output = model(data)

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    torch.save(model.state_dict(), 'model_batchnorm_fixed_from267e5.pth')
    wandb.save('model_batchnorm_fixed_from267e5.pth')
  
    
    # if test_accuracy_random > best_map_random:
    #     best_map_random = test_accuracy_random
    #     wandb.run.summary["best_map_random"] = best_map_random
    #     torch.save(model.state_dict(), 'model_best_random.pth')
    #     wandb.save('model_best_random.pth')
   