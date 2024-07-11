import torch
import torch.nn.functional as F
from torch.autograd import grad
from model import GATBasedMolecularGraphNeuralNetwork194
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn as nn
from dataset import  FileMegaSetMultiDir
from sklearn.metrics import average_precision_score
import wandb
from tqdm.auto import tqdm


# wandb.init(project="domain_belka1", settings=wandb.Settings(code_dir="./systematic/domain_adaptation"), tags=['testing'])
# Hyperparameters
batch_size = 64
lr = 0.001
n_epochs = 100
lambda_gp = 8 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
encoder = GATBasedMolecularGraphNeuralNetwork194(hidden_size = 64, num_gcn_layers=6).to(device)
encoder.load_state_dict(torch.load('/home/anonymous/belka/systematic/mega_ml/models/tmp/222-e5-focloss-bn.pth'))
# Optimizers
optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr)

TRAIN_SUBSET_FILE = '/mnt/fastssd/datasets/belka_full'
TEST_FULL = '/mnt/fastssd/datasets/test/'
# TEST_RANDOM_FILE = '/mnt/fastssd/datasets/belka/test_random'
# TEST_UNIQUE_FILE = '/mnt/fastssd/datasets/belka/test_unique'

# Example of setting up DataLoader with prefetching
train_dataset = FileMegaSetMultiDir([TRAIN_SUBSET_FILE, TEST_FULL],ratio=0.5,ratio_samples=0.5)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=10, shuffle=True)

# Example of setting up DataLoader with prefetching
# test_full_dataset = FileMegaSet(TEST_FULL, ratio=0.5, ratio_samples=1)
# test_full_loader = DataLoader(train_dataset, batch_size=1, num_workers=10, shuffle=True)

# # Example of setting up DataLoader with prefetching
# test_random_dataset = FileMegaSet(TEST_RANDOM_FILE)
# # test_random_dataset = FileMegaSet(TRAIN_FILE)
# test_random_loader = DataLoader(test_random_dataset, batch_size=1, num_workers=10, shuffle=False)

# # Example of setting up DataLoader with prefetching
# test_unique_dataset = FileMegaSet(TEST_UNIQUE_FILE)
# # test_unique_dataset = FileMegaSet(TRAIN_FILE)
# test_unique_loader = DataLoader(test_unique_dataset, batch_size=1, num_workers=10, shuffle=False)

#  https://wandb.ai/savsunenko-sasha/domain_belka1/runs/9ozfkfof

log_interval = 10

  
def contrastive_loss(z1, z2, temperature=0.5):
    # Normalize the embeddings
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Compute pairwise cosine similarity matrix
    similarity_matrix = torch.matmul(z1, z2.T)
    
    # Scale the similarities by the temperature
    scaled_sim = similarity_matrix / temperature
    
    # Extract positive pair similarities (diagonal elements)
    positive_sim = torch.diag(scaled_sim)
    
    # Compute the denominator (sum of exponentials of all similarities)
    denom = torch.sum(torch.exp(scaled_sim), dim=1)
    
    # Compute the contrastive loss
    loss = -torch.log(torch.exp(positive_sim) / denom**0.5)
    
    return torch.mean(loss)
def mask_node_features(x, mask_rate=0.15):
    num_nodes, num_features = x.size()
    mask = torch.rand(num_nodes, num_features) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask

def pretrain_encoder(encoder, source_data, epochs=10):
    encoder.train()

    for epoch in range(0,10):
        pbar = tqdm(enumerate(source_data))
        for batch_idx, data_source in pbar: 
            data_target = data_source.detach().clone()
            data_target.x, mask = mask_node_features(data_target.x)

            data_source = data_source.to(device)
            data_target = data_target.to(device)
            optimizer_encoder.zero_grad()
            # with autocast():
            _, features_source = encoder(data_source)
            _, features_target = encoder(data_target)

            loss = contrastive_loss(features_source, features_target)
            loss.backward()
            optimizer_encoder.step()
            # scheduler.step()
            
            if batch_idx % log_interval == 0:
                # wandb.log({"loss_pretrain": loss.item()})
                pbar.set_description(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        torch.save(encoder.state_dict(), f'optimizer_encoder.pth')
        # wandb.save(f'optimizer_encoder.pth')

pretrain_encoder(encoder, train_loader)

