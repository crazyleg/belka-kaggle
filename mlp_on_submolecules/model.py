import torch
import torch.nn as nn
import torch.nn.functional as F


class TowerModel(nn.Module):
    def __init__(self, initial_layers):
        super(TowerModel, self).__init__()
        self.fc1 = nn.Linear(initial_layers, 1024)
        self.ln1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.leaky_relu(self.ln4(self.fc4(x)))
        x = self.dropout4(x)
        return x

class MultiTowerModel(nn.Module):
    def __init__(self):
        super(MultiTowerModel, self).__init__()
        # Instantiate five towers
        self.towers = nn.ModuleList([TowerModel(l) for _, l in zip(range(6), [2052,2052,2052])])
        self.fc_concat1 = nn.Linear(128*6, 256)
        self.ln_concat1 = nn.BatchNorm1d(256)
        self.dropout_concat1 = nn.Dropout(0.5)
        self.fc_concat2 = nn.Linear(256, 128)
        self.ln_concat2 = nn.BatchNorm1d(128)
        self.dropout_concat2 = nn.Dropout(0.5)
        # Final output layer
        self.fc_final = nn.Linear(128, 1)

    def forward(self, *inputs):
        # Process each input through its respective tower
        outputs = [tower(input_tensor.squeeze()) for tower, input_tensor in zip(self.towers, inputs)]
        # Concatenate all tower outputs
        concatenated = torch.cat(outputs, dim=1)
        # Further processing
        x = F.leaky_relu(self.ln_concat1(self.fc_concat1(concatenated)))
        x = self.dropout_concat1(x)
        x = F.leaky_relu(self.ln_concat2(self.fc_concat2(x)))
        x = self.dropout_concat2(x)
        x = self.fc_final(x)
        return x.squeeze()

class MultiTowerSubmoleculeModel(nn.Module):
    def __init__(self):
        super(MultiTowerSubmoleculeModel, self).__init__()
        # Instantiate five towers
        self.towers = nn.ModuleList([TowerModel(l) for _, l in zip(range(3), [6876,6876,6876])])
        # Concatenated output size from 5 towers: 5*128=640
        self.fc_concat1 = nn.Linear(384, 256)
        self.ln_concat1 = nn.BatchNorm1d(256)
        self.dropout_concat1 = nn.Dropout(0.5)
        self.fc_concat2 = nn.Linear(256, 128)
        self.ln_concat2 = nn.BatchNorm1d(128)
        self.dropout_concat2 = nn.Dropout(0.5)
        # Final output layer
        self.fc_final = nn.Linear(128, 1)

    def forward(self, *inputs):
        # Process each input through its respective tower
        outputs = [tower(input_tensor.squeeze()) for tower, input_tensor in zip(self.towers, inputs)]
        # Concatenate all tower outputs
        concatenated = torch.cat(outputs, dim=1)
        # Further processing
        x = F.leaky_relu(self.ln_concat1(self.fc_concat1(concatenated)))
        x = self.dropout_concat1(x)
        x = F.leaky_relu(self.ln_concat2(self.fc_concat2(x)))
        x = self.dropout_concat2(x)
        x = self.fc_final(x)
        return x.squeeze()