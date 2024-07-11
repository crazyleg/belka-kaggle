import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import schnetpack
import wandb
from pymongo import MongoClient
from tqdm.auto import tqdm
from dataset import  FileMegaSetMultiDir
from model import GATBasedMolecularGraphResNetPretrain
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Weights & Biases
wandb.init(project="belka_pretrain", settings=wandb.Settings(code_dir="./systematic/mega_ml"), tags=[])
TRAIN_FILE = '/mnt/storage/datasets/belka/train'
TEST_RANDOM_FILE = '/mnt/storage/datasets/belka/test_random'
TEST_UNIQUE_FILE = '/mnt/storage/datasets/belka/test_unique'

# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch Binary Classification')
parser.add_argument('--batch-size', type=int, default=4096, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4096, help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
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
train_dataset = FileMegaSetMultiDir([TRAIN_FILE, TEST_RANDOM_FILE, TEST_UNIQUE_FILE])
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=15, shuffle=True)

# Model
model = GATBasedMolecularGraphResNetPretrain(hidden_size = 64, num_gcn_layers=6).to(device)
optimizer = optim.RAdam(model.parameters(), lr=args.lr)
# Define the LR scheduler

# criterion = FocalLoss()
def train(epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader))
    train_targets = []
    train_outputs = []
    for batch_idx, data in pbar: 
        data = data.to(device)
        optimizer.zero_grad()
        class_output, pretrain_loss = model(data)  

        pretrain_loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            wandb.log({"loss": pretrain_loss.item()})
            pbar.set_description(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {pretrain_loss.item():.6f}')

for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    torch.save(model.state_dict(), 'pretrained_gnn_masked.pth')
    wandb.save('pretrained_gnn_masked.pth')



 