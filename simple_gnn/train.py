from collections import defaultdict
import sys
import timeit

import numpy as np
from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import wandb
import preprocess as pp

from torch.utils.data.dataloader import default_collate
def pad_tensor(tensor, pad_size):
    if tensor.size(0) < pad_size:
        padding = torch.zeros(pad_size - tensor.size(0), tensor.size(1))
        tensor = torch.cat([tensor, padding], dim=0)
        padding = torch.zeros(pad_size, pad_size - tensor.size(1))
        tensor = torch.cat([tensor, padding], dim=1)
    return tensor
def custom_collate(batch):
    # Determine maximum size in this batch for padding
    # max_atoms = max([item[1].shape[0] for item in batch])  # Assuming item[1] is adjacency matrix

    # Initialize lists to hold batch data
    fingerprints_batch = []
    adjacency_batch = []
    molecular_size_batch = []
    properties_batch = []

    # # Process each item in the batch
    # for fingerprints, adjacency, molecular_size, properties in batch:
    #     # Pad adjacency matrix to max_atoms x max_atoms
    #     pad_size = (max_atoms, max_atoms)
    #     padded_adjacency = torch.zeros(pad_size)
    #     padded_adjacency[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
    #     adjacency_batch.append(padded_adjacency.to(properties.device))

    #     # Handle fingerprints and other data
    #     fingerprints_batch.append(fingerprints)
    #     molecular_size_batch.append(molecular_size)
    #     properties_batch.append(properties)

    # Stack all lists to create batch tensors
    # fingerprints_batch = torch.stack(fingerprints_batch, dim=0)
    # adjacency_batch = torch.stack(adjacency_batch, dim=0)
    # molecular_size_batch = torch.tensor(molecular_size_batch)
    # properties_batch = torch.stack(properties_batch, dim=0)

    # return fingerprints_batch, adjacency_batch, molecular_size_batch, properties_batch
    return list(zip(*batch))

class MolecularGraphNeuralNetwork(nn.Module):
    
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output_init = nn.Linear(dim+3, dim) 
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)
    def pad2(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        # shapes = [m.shape for m in matrices]
        # M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        # zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        # pad_matrices = pad_value + zeros
        # i, j = 0, 0
        # for k, matrix in enumerate(matrices):
        #     m, n = shapes[k]
        #     pad_matrices[i:i+m, j:j+n] = matrix
        #     i += m
        #     j += n
        padded_matrix = torch.block_diag(*matrices).float()
        return padded_matrix
    
    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        # padded_matrix = torch.block_diag(*matrices)
        return pad_matrices
    
    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)
    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)
    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)
    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad2(adjacencies, 0)

        """GNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors
    def mlp(self, vectors, proteins):
        """Classifier or regressor based on multilayer perceptron."""


        vectors = torch.concat((vectors, proteins), dim=1)
        vectors = torch.relu(self.W_output_init(vectors)) 
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs

    def forward_classifier(self, data_batch):

        inputs = data_batch[:-2]

        molecular_vectors = self.gnn(inputs)
        predicted_scores = self.mlp(molecular_vectors, torch.stack(data_batch[-2],dim=0))
        return predicted_scores 
     


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    def train(self, dataset):
        N = len(dataset)
        loss_total = 0
        for fingerprints, adjacency, molecular_size, properties, proteins in tqdm(data_loader):

            fingerprints = [torch.LongTensor(x).to(device) for x in fingerprints]
            proteins = [torch.LongTensor(x).to(device) for x in proteins] 
            adjacency = [torch.LongTensor(x).to(device) for x in adjacency]
            
            if task == 'classification':
                properties = torch.LongTensor(properties).to(device)
            elif task == 'regression':
                properties = torch.FloatTensor([[float(properties)]]).to(device)
            if task == 'classification':
                loss = self.model.forward_classifier((fingerprints, adjacency, molecular_size, proteins, properties ), train=True)
            if task == 'regression':
                loss = self.model.forward_regressor((fingerprints, adjacency, molecular_size, properties), train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
            wandb.log({"train_loss": loss.item()})
        average_loss = loss_total / len(data_loader)
        wandb.log({"average_train_loss": average_loss})
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test_classifier(self, dataset):
        N = len(dataset)
        P, C = [], []
        for fingerprints, adjacency, molecular_size, properties, protein in data_loader:
            fingerprints = [torch.LongTensor(x).to(device) for x in fingerprints]
            proteins = [torch.LongTensor(x).to(device) for x in protein] 
            adjacency = [torch.LongTensor(x).to(device) for x in adjacency]
            
            if task == 'classification':
                properties = torch.LongTensor(properties).to(device)
            elif task == 'regression':
                properties = torch.FloatTensor([[float(properties)]]).to(device)
            predicted_scores, correct_labels = self.model.forward_classifier(
                                               (fingerprints, adjacency, molecular_size, proteins, properties), train=False)
            P.append(predicted_scores)
            C.append(correct_labels)
        AUC = roc_auc_score(np.concatenate(C), np.concatenate(P))
        wandb.log({f"AUC_metric": AUC})
        return AUC

    def test_regressor(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        for fingerprints, adjacency, molecular_size, properties in data_loader:
            predicted_values, correct_values = self.model.forward_regressor(
                                               (fingerprints, adjacency, molecular_size, properties), train=False)
            SAE += sum(np.abs(predicted_values-correct_values))
        MAE = SAE / N  # mean absolute error.
        return MAE

    def save_result(self, result, filename, epoch):
        with open(filename, 'a') as f:
            f.write(result + '\n')
        torch.save(self.model.state_dict(), f'./models/{epoch}_model.pth')
        


atom_dict = {'C': 0,
             'N': 1,
             ('C', 'aromatic'): 2,
             ('N', 'aromatic'): 3,
             'O': 4,
             'H': 5,
             ('S', 'aromatic'): 6,
             'Cl': 7,
             'S': 8,
             ('O', 'aromatic'): 9,
             'Br': 10,
             'F': 11,
             'Si': 12,
             'B': 13,
             'I': 14}

bond_dict = {'TRIPLE': 0, 'SINGLE': 1, 'AROMATIC': 2, 'DOUBLE': 3}

fingerprint_dict = {0: 0,
             1: 1,
             2: 2,
             3: 3,
             4: 4,
             5: 5,
             6: 6,
             7: 7,
             8: 8,
             9: 9,
             10: 10,
             11: 11,
             12: 12,
             13: 13,
             14: 14}

edge_dict = {}

CONNECTION_STRING = "mongodb://anonymous-server.local/belka"
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.client = MongoClient(CONNECTION_STRING)
    dataset.db = dataset.client['belka']
    dataset.collection = dataset.db['bindings_raw_train']
if __name__ == "__main__":
    wandb.init(project='belka',settings=wandb.Settings(code_dir="."))
    torch.multiprocessing.set_start_method('spawn')

    (task, dataset, radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = ('classification', 'any', 1, 50, 6, 6, 256, 256, 1e-4, 0.99, 10, 20, 'dunno')
    
    (radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, decay_interval,
     iteration) = map(int, [radius, dim, layer_hidden, layer_output,
                            batch_train, batch_test,
                            decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print('The code uses a Apple Silicon!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', dataset, 'dataset.')
    print('Just a moment......')
    # (dataset_train, dataset_dev, dataset_test,
    #  N_fingerprints) = pp.create_datasets(task, dataset, radius, device)
    print('-'*100)

    print('The preprocess has finished!')
    # print('# of training data samples:', len(dataset_train))
    # print('# of development data samples:', len(dataset_dev))
    # print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork(
            16, dim, layer_hidden, layer_output).to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    file_result = 'result--' + setting + '.txt'
    if task == 'classification':
        result = 'Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test'
    if task == 'regression':
        result = 'Epoch\tTime(sec)\tLoss_train\tMAE_dev\tMAE_test'

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()
    import pickle


    # Initialize MongoDB connection here if necessary

    dataset = pp.MoleculeDatasetMongo(filename="BRD4_train.txt", dir_dataset="./", task="classification", atom_dict=atom_dict, bond_dict=bond_dict, fingerprint_dict=fingerprint_dict, edge_dict=edge_dict, radius=0, device=device, leng=1_000_000)
    dataset_test = pp.MoleculeDatasetMongo(filename="BRD4_test.txt", dir_dataset="./", task="classification", atom_dict=atom_dict, bond_dict=bond_dict, fingerprint_dict=fingerprint_dict, edge_dict=edge_dict, radius=0, device=device, leng=1_000_00)
    data_loader = DataLoader(dataset, batch_size=batch_train, shuffle=True, num_workers=4,  worker_init_fn=worker_init_fn,collate_fn=custom_collate)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_train, shuffle=True, num_workers=4,  worker_init_fn=worker_init_fn, collate_fn=custom_collate)

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(data_loader)

        if task == 'classification':
            prediction_dev = tester.test_classifier(data_loader)
            prediction_test = tester.test_classifier(data_loader_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_train,
                                     prediction_dev, prediction_test]))
        tester.save_result(result, file_result, epoch)

        print(result)