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
import torchvision
from dataset import  FileMegaSet
from model import GATBasedMolecularGraphNeuralNetworkV3, GNNEncoder
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss import SmoothAP
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
wandb.init(project="belka", settings=wandb.Settings(code_dir="./systematic/mega_ml"), tags=['GAT_naive','split2','startingsp2'])
TRAIN_FILE = '/mnt/fastssd/datasets/belka_full'
TEST_RANDOM_FILE = '/mnt/fastssd/datasets/belka/test_random'
TEST_UNIQUE_FILE = '/mnt/fastssd/datasets/belka/test_unique'
# Command Line Arguments
parser = argparse.ArgumentParser(description='PyTorch Binary Classification')
parser.add_argument('--batch-size', type=int, default=4096, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=4096, help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
args = parser.parse_args()
# Freeze BatchNorm layers
def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
# Setup CUDA
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

# Example of setting up DataLoader with prefetching
train_dataset = FileMegaSet(TRAIN_FILE, ratio=0.5)
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
model = GNNEncoder().to(device)
model.load_state_dict(torch.load('/home/anonymous/belka/systematic/mega_ml/models/tmp/contrastive_pretrainedoptimizer_encoder.pth'))

optimizer = optim.Adam(model.parameters(), lr = args.lr)

criterion = torch.nn.BCEWithLogitsLoss()

def train(epoch):
    model.train()
    # freeze_batchnorm(model)
    pbar = tqdm(enumerate(train_loader))
    train_targets = []
    train_outputs = []
    for batch_idx, data in pbar: 
        data = data.to(device)
        optimizer.zero_grad()
        # with autocast():
        output, emb = model(data)
        loss = criterion(output.squeeze(), data.y.squeeze().float() )
        loss.backward()
        optimizer.step()
        train_outputs.extend(output.detach().sigmoid().squeeze().cpu().tolist())
        train_targets.extend(data.y.cpu().tolist())
        
        if batch_idx % args.log_interval == 0:
            wandb.log({"loss": loss.item()})
            pbar.set_description(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    test_average_precision = average_precision_score(train_targets, train_outputs, average='micro')
    wandb.log({f"traing_map_micro": test_average_precision})
    torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
    wandb.save(f'model_epoch_{epoch}.pth')

def test(name, ds):
    model.eval()
    test_targets = []
    test_outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(ds))
        for batch_idx, data in pbar: 
            data = data.to(device)
            output = model(data)  
            test_outputs.extend(output.detach().sigmoid().squeeze().tolist())
            test_targets.extend(data.y.cpu().tolist())
            pbar.set_description(f'Test {name} Epoch: {epoch} [{batch_idx}/{len(ds)} ({100. * batch_idx / len(ds):.0f}%)]')
    
    test_average_precision = average_precision_score(test_targets, test_outputs, average='micro')
    wandb.log({f"test_map_micro_{name}": test_average_precision})
    return test_average_precision

best_map_unique = 0
best_map_random = 0
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_accuracy_unique = test("unique", test_unique_loader)
    test_accuracy_random = test("random", test_random_loader)
    # if epoch==5:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 0.003
    # if epoch==10:
    #     for g in optimizer.param_groups:
    #         g['lr'] = 0.001
    # set_requires_grad(model.GAT, True)



    # Checkpoint model if current MAP is better
    if test_accuracy_unique > best_map_unique:
        best_map_unique = test_accuracy_unique
        wandb.run.summary["best_map_unique"] = best_map_unique
        torch.save(model.state_dict(), 'model_best_unique.pth')
        wandb.save('model_best_unique.pth')
    
    # if test_accuracy_random > best_map_random:
    #     best_map_random = test_accuracy_random
    #     wandb.run.summary["best_map_random"] = best_map_random
    #     torch.save(model.state_dict(), 'model_best_random.pth')
    #     wandb.save('model_best_random.pth')
   