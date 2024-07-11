import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling
from torch_geometric.nn.conv import GraphConv, NNConv 
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.models import GAT, GCN, GIN
from torch_geometric.nn.norm import GraphNorm

from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool, TopKPooling, SoftmaxAggregation, LSTMAggregation
from torch_geometric.nn import GraphConv, GATv2Conv


class GNNEncoder(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GNNEncoder, self).__init__()

        self.agg1 = SoftmaxAggregation(learn=True)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='GraphNorm', out_channels=128, act='prelu', dropout=0)
        self.fc = nn.Linear(128*2, 128)
        self.dropout_ecfp = nn.Dropout(0.5)
        self.dropout_molvec = nn.Dropout(0.1)
        self.ecfp = nn.Linear(2048, 64)
        self.mol_vec = nn.Linear(300, 64)
        self.fc2 = nn.Linear(128, 64)
        self.A = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        
         # Global mean pooling
        p1 = self.agg1(x, data.batch)
        p2 = global_add_pool(x, data.batch)

        x = torch.cat([p1, p2], dim=1)
        
        # Final fully connected layer
        x = F.mish(self.fc(x))
        features = self.fc2(x)
        A = self.A(F.mish(features))
        return A, features


class GATBasedMolecularGraphNeuralNetworkAlt(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkAlt, self).__init__()
      

       
        self.GIN = GIN(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, norm='BatchNorm', out_channels=128, act='leaky_relu', dropout=0)
        self.fc = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GIN(data.x, data.edge_index.to(torch.int64), data.edge_attr.to(torch.float32))
        
         # Global mean pooling
        p1 = global_mean_pool(x, data.batch)
        p2 = global_max_pool(x, data.batch)

        x = torch.cat([p1, p2], dim=1)
        # Final fully connected layer
        emb = x
        x = F.leaky_relu(self.fc(x))
        x = self.fc2(x)
        return x, emb


class GATBasedMolecularGraphNeuralNetwork194(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetwork194, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

       
        self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='leaky_relu', dropout=0)
        self.fc = nn.Linear(128*2, 64)
        # self.agg1 = SoftmaxAggregation(learn=True)
        # self.dropout_concat2 = nn.Dropout(0.3)
        # self.ecfp = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        
         # Global mean pooling
        p1 = global_mean_pool(x, data.batch)
        p2 = global_max_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        emb = torch.cat([p1, p2], dim=1)

        # Final fully connected layer
        x = F.leaky_relu(self.fc(emb))
        x = self.fc2(x)
        return x, emb