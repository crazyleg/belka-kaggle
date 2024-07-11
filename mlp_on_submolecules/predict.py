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
from dataset import  APIBasedMoleculeTestDataset
from model import BinaryClassificationMLP, MultiTowerModel, MultiTowerSubmoleculeModel
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
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
    dataset.raw_data = dataset.db['test_metadata']
    dataset.ds = pd.read_parquet('test.parquet')['id']

# Setup CUDA
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# Example of setting up DataLoader with prefetching
train_dataset = APIBasedMoleculeTestDataset('belka')
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=15, worker_init_fn=worker_init_fn)


# Model

model = MultiTowerModel()
model.load_state_dict(torch.load('model_best_random.pth'))
model=model.to(device)
ids = []
scores = []
def test(ds):
    model.eval()
    with torch.no_grad():
        for batch_idx, (tower1, tower2, tower3, tower4, tower5, submolecules, target) in enumerate(tqdm(ds)):
            tower1, tower2, tower3, tower4, tower5, submolecules, target = tower1.to(device), tower2.to(device), tower3.to(device), tower4.to(device), tower5.to(device), submolecules.to(device), target
            output = model(tower1, tower2, tower3, tower4, tower5, submolecules)
            
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
df.to_csv('output_random.csv', index=False)  # index=False means do not write row numbers