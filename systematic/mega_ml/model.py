import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling
from torch_geometric.nn.conv import GraphConv, NNConv 
from torch_geometric.nn.aggr import MeanAggregation, LSTMAggregation
from torch_geometric.nn.models import GAT, GCN, GIN
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool, TopKPooling, SoftmaxAggregation, LSTMAggregation
from torch_geometric.nn import GraphConv, GATv2Conv

class GATBasedMolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetwork, self).__init__()

        self.agg1 = SoftmaxAggregation(learn=True)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='prelu', dropout=0)
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
        x = F.mish(self.fc2(x))
        A = self.A(x)
        return A


class GATBasedMolecularGraphNeuralNetworkWithVectors(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkWithVectors, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='leaky_relu', dropout=0)
        self.fc = nn.Linear(128*2 + 300, 64)
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

        x = torch.cat([p1, p2, data.word300_mean], dim=1)
        

        # Final fully connected layer
        x = F.leaky_relu(self.fc(x))
        x = self.fc2(x)
        return x


class GATBasedMolecularGraphNeuralNetworkAndEmd(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkAndEmd, self).__init__()
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

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x1 = F.leaky_relu(self.fc(x))
        x = self.fc2(x1)
        return x, x1


class GATBasedMolecularGraphNeuralNetworkV3(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkV3, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        # self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='prelu', dropout=0.2)
        self.fc = nn.Linear(128, 64)
        self.agg1 = SoftmaxAggregation(learn=True)
        self.dropout_concat2 = nn.Dropout(0.2)
        # self.ecfp = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        
        # Global mean pooling
        # p1 = self.agg1(x, data.batch)
        p2 = global_add_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        x = torch.cat([p2], dim=1)
        

        # Final fully connected layer
        x = F.mish(self.dropout_concat2(self.fc(x)))
        x = self.fc2(x)
        return x
    
class GATBasedMolecularGraphNeuralNetworkV2(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkV2, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='prelu', dropout=0)
        self.fc = nn.Linear(128*2, 64)
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
        p2 = global_sum_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x = F.mish(self.fc(x))
        x = self.fc2(x)
        return x
class GATResNetBlock(nn.Module):
    def __init__(self, channels):
        super(GATResNetBlock, self).__init__()
        self.gat1 = GATConv(channels, channels, heads=1, concat=True, dropout=0)
        self.gat2 = GATConv(channels, channels, heads=1, concat=True, dropout=0)
        self.norm = GraphNorm(channels)
        # TODO batch norm

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = F.leaky_relu(self.norm(self.gat1(x, edge_index, edge_attr)))
        x = self.norm(self.gat2(x, edge_index, edge_attr))
        x += residual
        return F.leaky_relu(x)


class GATBasedMolecularGraphResNet(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64, num_gcn_layers=4, num_classes=3):
        super(GATBasedMolecularGraphResNet, self).__init__()
        self.GAT = GIN(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, norm='GraphNorm', out_channels=128, act='leaky_relu', dropout=0)
       
        self.fc1 = nn.Linear(hidden_size*6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.to(torch.float32) 
        big_GAT = self.GAT(x, edge_index, edge_attr)

        big_GAT0 = global_mean_pool(big_GAT, data.batch)
        big_GAT1 = global_add_pool(big_GAT, data.batch)
        big_GAT2 = global_max_pool(big_GAT, data.batch)

        
        x = F.leaky_relu(self.fc1(torch.cat([big_GAT0, big_GAT1, big_GAT2], dim=1)))
        x = self.fc2(x)
        
        return  x

class DeepHGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, num_layers=6):
        super(DeepHGNN, self).__init__()
        self.num_layers = num_layers
        
        # Initial GCN layer
        self.convs = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.attns.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False))
            self.pools.append(TopKPooling(hidden_dim, ratio=0.8))

        # Output layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.to(torch.float32)
        batch = data.batch

        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            if i < self.num_layers - 1:  # Apply pooling and attention except the last layer
                x, edge_index, edge_attr, batch, _, _ = self.pools[i](x, edge_index, edge_attr, batch=batch)
                x = F.relu(self.attns[i](x, edge_index))

        # Global Pooling
        x = global_mean_pool(x, batch) + global_max_pool(x, batch) + global_add_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x 

def mask_node_features(x, mask_rate=0.15):
    num_nodes, num_features = x.size()
    mask = torch.rand(num_nodes, num_features) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask

class GATBasedMolecularGraphResNetPretrain(nn.Module):
    def __init__(self, in_channels=48, hidden_size=64, num_gcn_layers=4, num_classes=3):
        super(GATBasedMolecularGraphResNetPretrain, self).__init__()
        self.GAT = GIN(in_channels, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, norm='GraphNorm', out_channels=128, act='leaky_relu', dropout=0)
        
        # For pretraining (decoder to predict masked features)
        self.decoder = nn.Linear(128, in_channels)
        
        # For classification
        self.fc1 = nn.Linear(hidden_size*6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc2_ecpf = nn.Linear(hidden_size, 2048)
        self.fc2_torsion = nn.Linear(hidden_size, 2048)
        self.fc2_word300_m = nn.Linear(hidden_size, 300)
        self.fc2_word300_s = nn.Linear(hidden_size, 1)
        
    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.to(torch.float32)
        ecpf = data.ecfp.to(torch.float32)
        word300_m = data.word300_mean.to(torch.float32)
        torsion = data.torsion.to(torch.float32)
        
        # Pretraining step
        x_masked, mask = mask_node_features(x,0.3)
        edge_masked, _ = mask_node_features(edge_attr,0.3)
        big_GAT = self.GAT(x_masked, edge_index, edge_masked)
        x_reconstructed = self.decoder(big_GAT)
        
        # Apply the mask to filter out unmasked positions for the loss calculation
        pretrain_loss = F.mse_loss(x_reconstructed[mask], x[mask])

        
        # Pooling operations for classification
        big_GAT0 = global_mean_pool(big_GAT, data.batch)
        big_GAT1 = global_add_pool(big_GAT, data.batch)
        big_GAT2 = global_max_pool(big_GAT, data.batch)
        
        x_pooled = F.leaky_relu(self.fc1(torch.cat([big_GAT0, big_GAT1, big_GAT2], dim=1)))

        class_output = self.fc2(x_pooled)/2
        ecpf_loss = F.mse_loss(self.fc2_ecpf(x_pooled),ecpf) 
        torsion_loss = F.mse_loss(self.fc2_torsion(x_pooled),torsion) 
        w300_m_loss = F.mse_loss(self.fc2_word300_m(x_pooled),word300_m) 

        
        return class_output, (pretrain_loss+ecpf_loss+torsion_loss+w300_m_loss)


class GATBasedMolecularGraphNeuralNetwork160(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetwork160, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        # self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='GraphNorm', out_channels=128, act='prelu', dropout=0)
        self.fc = nn.Linear(128*2, 64)
        self.agg1 = SoftmaxAggregation(learn=True)
        # self.dropout_concat2 = nn.Dropout(0.3)
        # self.ecfp = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        
        # Global mean pooling
        p1 = self.agg1(x, data.batch)
        p2 = global_add_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x = F.mish(self.fc(x))
        x = self.fc2(x)
        return x
    
class DomainDiscriminator(nn.Module):
    def __init__(self, h_feats):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(h_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, 1)
        
    def forward(self, h):
        x = F.relu(self.fc1(h))
        return torch.sigmoid(self.fc2(x))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool import SAGPooling, ASAPooling
from torch_geometric.nn.conv import GraphConv, NNConv 
from torch_geometric.nn.aggr import MeanAggregation, LSTMAggregation
from torch_geometric.nn.models import GAT, GCN, GIN
from torch_geometric.nn.norm import GraphNorm

from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool, TopKPooling, SoftmaxAggregation, LSTMAggregation
from torch_geometric.nn import GraphConv, GATv2Conv

class GATBasedMolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetwork, self).__init__()

        self.agg1 = SoftmaxAggregation(learn=True)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='prelu', dropout=0)
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
        x = F.mish(self.fc2(x))
        A = self.A(x)
        return A


class GATBasedMolecularGraphNeuralNetworkWithVectors(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkWithVectors, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='leaky_relu', dropout=0)
        self.fc = nn.Linear(128*2 + 300, 64)
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

        x = torch.cat([p1, p2, data.word300_mean], dim=1)
        

        # Final fully connected layer
        x = F.leaky_relu(self.fc(x))
        x = self.fc2(x)
        return x


class GATBasedMolecularGraphNeuralNetworkAndEmd(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkAndEmd, self).__init__()
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

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x1 = F.leaky_relu(self.fc(x))
        x = self.fc2(x1)
        return x, x1


class GATBasedMolecularGraphNeuralNetworkV3(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkV3, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        # self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='prelu', dropout=0.2)
        self.fc = nn.Linear(128, 64)
        self.agg1 = SoftmaxAggregation(learn=True)
        self.dropout_concat2 = nn.Dropout(0.2)
        # self.ecfp = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        
        # Global mean pooling
        # p1 = self.agg1(x, data.batch)
        p2 = global_add_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        x = torch.cat([p2], dim=1)
        

        # Final fully connected layer
        x = F.mish(self.dropout_concat2(self.fc(x)))
        x = self.fc2(x)
        return x
    
class GATBasedMolecularGraphNeuralNetworkV2(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetworkV2, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='BatchNorm', out_channels=128, act='prelu', dropout=0)
        self.fc = nn.Linear(128*2, 64)
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
        p2 = global_sum_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x = F.mish(self.fc(x))
        x = self.fc2(x)
        return x
class GATResNetBlock(nn.Module):
    def __init__(self, channels):
        super(GATResNetBlock, self).__init__()
        self.gat1 = GATConv(channels, channels, heads=1, concat=True, dropout=0)
        self.gat2 = GATConv(channels, channels, heads=1, concat=True, dropout=0)
        self.norm = GraphNorm(channels)
        # TODO batch norm

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = F.leaky_relu(self.norm(self.gat1(x, edge_index, edge_attr)))
        x = self.norm(self.gat2(x, edge_index, edge_attr))
        x += residual
        return F.leaky_relu(x)


class GATBasedMolecularGraphResNet(nn.Module):
    def __init__(self, in_channels=3, hidden_size=64, num_gcn_layers=4, num_classes=3):
        super(GATBasedMolecularGraphResNet, self).__init__()
        self.GAT = GIN(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, norm='GraphNorm', out_channels=128, act='leaky_relu', dropout=0)
       
        self.fc1 = nn.Linear(hidden_size*6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.to(torch.float32) 
        big_GAT = self.GAT(x, edge_index, edge_attr)

        big_GAT0 = global_mean_pool(big_GAT, data.batch)
        big_GAT1 = global_add_pool(big_GAT, data.batch)
        big_GAT2 = global_max_pool(big_GAT, data.batch)

        
        x = F.leaky_relu(self.fc1(torch.cat([big_GAT0, big_GAT1, big_GAT2], dim=1)))
        x = self.fc2(x)
        
        return  x

class DeepHGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, num_layers=6):
        super(DeepHGNN, self).__init__()
        self.num_layers = num_layers
        
        # Initial GCN layer
        self.convs = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.attns.append(GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False))
            self.pools.append(TopKPooling(hidden_dim, ratio=0.8))

        # Output layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.to(torch.float32)
        batch = data.batch

        for i in range(self.num_layers):
            x = F.relu(self.convs[i](x, edge_index))
            if i < self.num_layers - 1:  # Apply pooling and attention except the last layer
                x, edge_index, edge_attr, batch, _, _ = self.pools[i](x, edge_index, edge_attr, batch=batch)
                x = F.relu(self.attns[i](x, edge_index))

        # Global Pooling
        x = global_mean_pool(x, batch) + global_max_pool(x, batch) + global_add_pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x 

def mask_node_features(x, mask_rate=0.15):
    num_nodes, num_features = x.size()
    mask = torch.rand(num_nodes, num_features) < mask_rate
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked, mask

class GATBasedMolecularGraphResNetPretrain(nn.Module):
    def __init__(self, in_channels=48, hidden_size=64, num_gcn_layers=4, num_classes=3):
        super(GATBasedMolecularGraphResNetPretrain, self).__init__()
        self.GAT = GIN(in_channels, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, norm='GraphNorm', out_channels=128, act='leaky_relu', dropout=0)
        
        # For pretraining (decoder to predict masked features)
        self.decoder = nn.Linear(128, in_channels)
        
        # For classification
        self.fc1 = nn.Linear(hidden_size*6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc2_ecpf = nn.Linear(hidden_size, 2048)
        self.fc2_torsion = nn.Linear(hidden_size, 2048)
        self.fc2_word300_m = nn.Linear(hidden_size, 300)
        self.fc2_word300_s = nn.Linear(hidden_size, 1)
        
    def forward(self, data):
        x = data.x.to(torch.float32)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.to(torch.float32)
        ecpf = data.ecfp.to(torch.float32)
        word300_m = data.word300_mean.to(torch.float32)
        torsion = data.torsion.to(torch.float32)
        
        # Pretraining step
        x_masked, mask = mask_node_features(x,0.3)
        edge_masked, _ = mask_node_features(edge_attr,0.3)
        big_GAT = self.GAT(x_masked, edge_index, edge_masked)
        x_reconstructed = self.decoder(big_GAT)
        
        # Apply the mask to filter out unmasked positions for the loss calculation
        pretrain_loss = F.mse_loss(x_reconstructed[mask], x[mask])

        
        # Pooling operations for classification
        big_GAT0 = global_mean_pool(big_GAT, data.batch)
        big_GAT1 = global_add_pool(big_GAT, data.batch)
        big_GAT2 = global_max_pool(big_GAT, data.batch)
        
        x_pooled = F.leaky_relu(self.fc1(torch.cat([big_GAT0, big_GAT1, big_GAT2], dim=1)))

        class_output = self.fc2(x_pooled)/2
        ecpf_loss = F.mse_loss(self.fc2_ecpf(x_pooled),ecpf) 
        torsion_loss = F.mse_loss(self.fc2_torsion(x_pooled),torsion) 
        w300_m_loss = F.mse_loss(self.fc2_word300_m(x_pooled),word300_m) 

        
        return class_output, (pretrain_loss+ecpf_loss+torsion_loss+w300_m_loss)


class GATBasedMolecularGraphNeuralNetwork160(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=6):
        super(GATBasedMolecularGraphNeuralNetwork160, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        # self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='GraphNorm', out_channels=128, act='prelu', dropout=0)
        self.fc = nn.Linear(128*2, 64)
        self.agg1 = SoftmaxAggregation(learn=True)
        # self.dropout_concat2 = nn.Dropout(0.3)
        # self.ecfp = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        
        # Global mean pooling
        p1 = self.agg1(x, data.batch)
        p2 = global_add_pool(x, data.batch)

        # ecfp = self.dropout_concat2(F.leaky_relu(self.ecfp(data.ecfp.to(torch.float)))) 

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x = F.mish(self.fc(x))
        x = self.fc2(x)
        return x
    
class DomainDiscriminator(nn.Module):
    def __init__(self, h_feats):
        super(DomainDiscriminator, self).__init__()
        self.fc1 = nn.Linear(h_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, 1)
        
    def forward(self, h):
        x = F.relu(self.fc1(h))
        return torch.sigmoid(self.fc2(x))


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

        emb = torch.cat([p1, p2], dim=1)
        # Final fully connected layer
        x = F.leaky_relu(self.fc(emb))
        x = self.fc2(x)
        return x,emb

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

        x = torch.cat([p1, p2], dim=1)
        

        # Final fully connected layer
        x = F.leaky_relu(self.fc(x))
        emb = x
        x = self.fc2(x)
        return x, emb


class GATBasedMolecularGraphNeuralNetwork194New(nn.Module):
    def __init__(self, in_channels=48, hidden_size = 64, num_gcn_layers=5):
        super(GATBasedMolecularGraphNeuralNetwork194New, self).__init__()
        # NNConv setup
        # self.edge_nn = nn.Sequential(
        #     nn.Linear(10, hidden_size * in_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size * in_channels, hidden_size * in_channels)
        # )
        # self.nnconv = NNConv(in_channels=48, out_channels=hidden_size, nn=self.edge_nn, aggr='mean')

        self.readout_in = nn.Linear(128,64) 
        self.readout_out = nn.Linear(64,64) 
        self.pool = SAGPooling(hidden_size)
        self.GAT = GAT(in_channels=48, hidden_channels=hidden_size*2, num_layers=num_gcn_layers, v2=True, norm='GraphNorm', out_channels=128, act='prelu', dropout=0)
        self.fc_n = nn.Linear(64, 64)
        # self.agg1 = SoftmaxAggregation(learn=True)
        # self.dropout_concat2 = nn.Dropout(0.3)
        # self.ecfp = nn.Linear(2048, 64)
        self.fc_n2 = nn.Linear(64, 3)
        
    def forward(self, data):
        data.x = data.x.to(torch.float32)
        # x = self.nnconv(data.x, data.edge_index.to(torch.long), data.edge_attr.to(torch.float32))
        # x = F.relu(x)

        x = self.GAT(data.x, data.edge_index, data.edge_attr.to(torch.float32))
        in_project = self.readout_in(x) 
        out_project = self.readout_out(F.leaky_relu(in_project)) 
         # Global mean pooling
        p1 = global_mean_pool(out_project, data.batch)

        # Final fully connected layer
        x = F.leaky_relu(self.fc_n(p1))
        emb = x
        x = self.fc_n2(x)
        return x, emb
    

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
