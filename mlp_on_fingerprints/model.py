import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights.expand_as(x)
    
class TowerModel(nn.Module):
    def __init__(self, initial_layers):
        super(TowerModel, self).__init__()
        self.fc1 = nn.Linear(initial_layers, 1024)
        self.ln1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.attention1 = FeatureAttention(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.ln2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)


        self.fc3 = nn.Linear(1024, 256)
        self.ln3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x1 = self.attention1(x)

        x = F.leaky_relu(self.ln2(self.fc2(x1)))
        x = self.dropout2(x)
        x += x1

        x = F.leaky_relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.ln4(self.fc4(x)))
        return x

class MultiTowerModel(nn.Module):
    def __init__(self):
        super(MultiTowerModel, self).__init__()
        # Instantiate five towers
        self.towers = nn.ModuleList([TowerModel(l) for _, l in zip(range(9), [2048,2048,2048,167,512,2048,2048,300,300])])
        self.fc_concat1 = nn.Linear(128*9 + 3, 256)
        self.ln_concat1 = nn.BatchNorm1d(256)
        self.attention1 = FeatureAttention(256)
        
        self.dropout_concat1 = nn.Dropout(0.3)
        self.fc_concat2 = nn.Linear(256, 128)
        self.ln_concat2 = nn.BatchNorm1d(128)
        self.dropout_concat2 = nn.Dropout(0.3)
        # Final output layer
        self.fc_final = nn.Linear(128 + 3, 1)


    def forward(self, *inputs):
        # Process each input through its respective tower
        outputs = [tower(input_tensor.squeeze()) for tower, input_tensor in zip(self.towers, inputs)]
        outputs.append(inputs[-1].squeeze())
        # Concatenate all tower outputs

        concatenated = torch.cat(outputs, dim=1)
        # Further processing
        x = F.leaky_relu(self.ln_concat1(self.fc_concat1(concatenated)))
        x = self.dropout_concat1(x)
        x = self.attention1(x)
        x = F.leaky_relu(self.ln_concat2(self.fc_concat2(x)))
        x = self.dropout_concat2(x)
        x = self.fc_final(torch.cat([x, inputs[-1].squeeze()],dim=1))
        return x.squeeze()
