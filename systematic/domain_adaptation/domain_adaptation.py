import torch
import torch.nn.functional as F
from torch.autograd import grad
from model import GATBasedMolecularGraphNeuralNetwork194
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn as nn
from dataset import  FileMegaSet
from sklearn.metrics import average_precision_score
import wandb
from tqdm.auto import tqdm


wandb.init(project="domain_belka1", settings=wandb.Settings(code_dir="./systematic/domain_adaptation"), tags=['domain_adapt'])
# Hyperparameters
batch_size = 64
lr = 0.001
n_epochs = 100
lambda_gp = 8 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
encoder = GATBasedMolecularGraphNeuralNetwork194(hidden_size = 64, num_gcn_layers=6).to(device)
encoder.load_state_dict(torch.load('/home/anonymous/belka/systematic/mega_ml/models/tmp/domain-23-optimizer_encoder_0.pth'))

# Optimizers
optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr)

optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.0001)
gamma = 1 

    
TRAIN_SUBSET_FILE = '/mnt/fastssd/datasets/belka_full'
TEST_FULL = '/mnt/fastssd/datasets/test'
TEST_RANDOM_FILE = '/mnt/fastssd/datasets/belka/test_random'
TEST_UNIQUE_FILE = '/mnt/fastssd/datasets/belka/test_unique'

# Example of setting up DataLoader with prefetching
train_dataset = FileMegaSet(TRAIN_SUBSET_FILE, ratio=0.4, ratio_samples=1)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=10, shuffle=True)

# Example of setting up DataLoader with prefetching
test_full_dataset = FileMegaSet(TEST_FULL, ratio=0.4, ratio_samples=1)
test_full_loader = DataLoader(train_dataset, batch_size=1, num_workers=10, shuffle=True)

# # Example of setting up DataLoader with prefetching
test_random_dataset = FileMegaSet(TEST_RANDOM_FILE)
# test_random_dataset = FileMegaSet(TRAIN_FILE)
test_random_loader = DataLoader(test_random_dataset, batch_size=1, num_workers=10, shuffle=False)

# Example of setting up DataLoader with prefetching
test_unique_dataset = FileMegaSet(TEST_UNIQUE_FILE)
# test_unique_dataset = FileMegaSet(TRAIN_FILE)
test_unique_loader = DataLoader(test_unique_dataset, batch_size=1, num_workers=10, shuffle=False)

#  https://wandb.ai/savsunenko-sasha/domain_belka1/runs/9ozfkfof

log_interval = 10
criterion = torch.nn.BCEWithLogitsLoss()
  
def compute_similarity(z1, z2, temperature):
    return torch.exp(torch.matmul(z1, z2.t()) / temperature)

def generate_pseudo_labels(target_embeddings, source_embeddings, source_labels, temperature):
    similarities = compute_similarity(target_embeddings, source_embeddings, temperature)
    weights = similarities / similarities.sum(dim=1, keepdim=True)
    pseudo_labels = torch.matmul(weights, source_labels.float())
    p1 = source_labels[:,0].float().mean()
    p2 = source_labels[:,1].float().mean()
    p3 = source_labels[:,2].float().mean()
    ps1 = (pseudo_labels[:,0] > torch.quantile(pseudo_labels[:,0],(1-p1))).float()
    ps2 = (pseudo_labels[:,1] > torch.quantile(pseudo_labels[:,1],(1-p2))).float()
    ps3 = (pseudo_labels[:,2] > torch.quantile(pseudo_labels[:,2],(1-p3))).float()

    return torch.stack([ps1,ps2,ps3],dim=1) 

def cross_domain_contrastive_loss(target_embeddings, source_embeddings, target_pseudo_labels, source_labels, temperature, epsilon=1e-10):
    # Compute similarities between target and source embeddings
    similarities = compute_similarity(target_embeddings, source_embeddings, temperature)
    
    # Expand target_pseudo_labels to match the dimensions of source_labels
    expanded_target_labels = target_pseudo_labels.unsqueeze(1).expand(-1, source_labels.size(0), -1)
    
    # Compute mask for positive samples
    positives_mask = (expanded_target_labels == source_labels.unsqueeze(0)).all(dim=2)
    
    # Compute positive similarities
    positive_similarities = similarities * positives_mask
    
    # Sum of positive similarities for each target sample
    positive_sums = positive_similarities.sum(dim=1) + epsilon
    
    # Sum of all similarities for each target sample
    all_sums = similarities.sum(dim=1) + epsilon
    
    # Compute the loss for each target sample
    loss_per_sample = -torch.log(positive_sums / all_sums)
    
    # Average the loss over all target samples
    loss = loss_per_sample.mean()
    return loss

def test(name, ds, epoch):
    encoder.eval()
    test_targets = []
    test_outputs = []
    with torch.no_grad():
        pbar = tqdm(enumerate(ds))
        for batch_idx, data in pbar: 
            data = data.to(device)
            output, __ = encoder(data)  
            test_outputs.extend(output.detach().sigmoid().squeeze().tolist())
            test_targets.extend(data.y.cpu().tolist())
            pbar.set_description(f'Test {name} Epoch: {epoch} [{batch_idx}/{len(ds)} ({100. * batch_idx / len(ds):.0f}%)]')
    
    test_average_precision = average_precision_score(test_targets, test_outputs, average='micro')
    wandb.log({f"test_map_micro_{name}": test_average_precision})
    return test_average_precision

def pretrain_encoder(encoder, source_data, epochs=10):
    encoder.train()
    best_map_unique = 0
    best_map_random = 0
    for epoch in range(0,25):
        pbar = tqdm(enumerate(zip(source_data, test_full_loader)))
        for batch_idx, (data_source, test_source) in pbar: 
            data_source = data_source.to(device)
            test_source = test_source.to(device)

            optimizer_encoder.zero_grad()
            labels_source, features_source = encoder(data_source)
            labels_test, features_target = encoder(test_source)
            features_source = F.normalize(features_source, p=2, dim=1)
            features_target = F.normalize(features_target, p=2, dim=1)

            pseudo_labels = generate_pseudo_labels(features_target, features_source, data_source.y, 0.5)
            loss_CD = cross_domain_contrastive_loss(features_target, features_source, pseudo_labels, data_source.y, 0.5)
            loss_CL = criterion(labels_source.squeeze(), data_source.y.squeeze().float())
            if torch.isinf(loss_CD) or torch.isinf(loss_CL):
                loss = loss_CL
            else:
                loss = loss_CD*2+loss_CL

            loss.backward()
            optimizer_encoder.step()
            if batch_idx % log_interval == 0:
                wandb.log({"loss_CL": loss_CL.item()})
                wandb.log({"loss_CD": loss_CD.item()})
                wandb.log({"loss": loss.item()})
                pbar.set_description(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tloss_CL: {loss_CL.item():.6f}\tloss_CD: {loss_CD.item():.6f}\tLoss: {loss.item():.6f}')
        
        
        test_accuracy_unique = test("unique", test_unique_loader, epoch)
        test_accuracy_random = test("random", test_random_loader, epoch)
        torch.save(encoder.state_dict(), f'optimizer_encoder_{epoch}.pth')
        wandb.save(f'optimizer_encoder_{epoch}.pth')
        

pretrain_encoder(encoder, train_loader)

