import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.conv import GraphConv 
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.models import GAT


class GATBasedMolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, in_channels=3, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetwork, self).__init__()
        self.emb = nn.Embedding(54, 10)


        self.GAT = GAT(in_channels=in_channels+10, hidden_channels=hidden_size, num_layers=num_gcn_layers, out_channels=None, v2=True, act='leaky_relu', dropout=0.3)
        self.fc = nn.Linear(hidden_size*2+3, 64)
        self.dropout_concat2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, edge_index, batch, protein):
        q = self.emb(x[:,0].int())
        x = self.GAT(torch.cat([x,q], dim=1), edge_index)
        
        # Global mean pooling
        p1 = global_mean_pool(x, batch)
        p2 = global_max_pool(x, batch)
        x = torch.cat([p1, protein, p2], dim=1)
        
        # Final fully connected layer
        x = self.dropout_concat2(F.leaky_relu(self.fc(x)))
        x = self.fc2(x)
        return x
    
class SOTAMolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, num_gcn_layers, num_gat_layers):
        super(SOTAMolecularGraphNeuralNetwork, self).__init__()
        # self.embedding = nn.Embedding(N_fingerprints, dim)
        self.first_gcb = GraphConv(3, dim) 
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GraphConv(dim, dim) for _ in range(num_gcn_layers)
        ])
        self.agg = MeanAggregation()
        
        # # GAT layers
        # self.gat_layers = nn.ModuleList([
        #     GATConv(dim, dim) for _ in range(num_gat_layers)
        # ])
        
        self.fc = nn.Linear(dim+3, 64)
        self.ln = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, edge_index, batch, protein):
        
        # x = self.embedding(x)
        x = F.relu(self.first_gcb(x, edge_index))
        # GCN layers with residual connections and ReLU activation
        for conv in self.gcn_layers:
            x = F.relu(conv(x, edge_index))
        
        # # Apply dropout
        # x = self.dropout(x)
        
        # # GAT layers with ELU activation
        # for conv in self.gat_layers:
        #     x = F.elu(conv(x, edge_index)[0]) + x
        
        # Global mean pooling
        x = global_mean_pool(x, batch)
        x = torch.cat([x, protein], dim=1)
        
        # Final fully connected layer
        x = F.elu(self.ln(self.fc(x)))
        x = self.dropout((x))
        x = self.fc2(x)
        return x