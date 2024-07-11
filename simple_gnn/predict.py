import torch
from torch.utils.data import DataLoader
import pandas as pd
import preprocess as pp
from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
CONNECTION_STRING = "mongodb://anonymous-server.local/belka"
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.client = MongoClient(CONNECTION_STRING)
    dataset.db = dataset.client['belka']
    dataset.collection = dataset.db['bindings_raw_test']
# Assuming the model class and other necessary imports are defined as per your provided code
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


class MolecularGraphNeuralNetwork(nn.Module):
    
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output_init = nn.Linear(dim+3, dim) 
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        task = 'classification'
        if task == 'classification':
            self.W_property = nn.Linear(dim, 2)
        if task == 'regression':
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
    def forward_classifier(self, data_batch, train):

        inputs = data_batch[:-1]
        with torch.no_grad():
            molecular_vectors = self.gnn(inputs)
            predicted_scores = self.mlp(molecular_vectors, torch.stack(data_batch[-1],dim=0))

        return predicted_scores

def load_model(model_path, device):
    model = MolecularGraphNeuralNetwork(N_fingerprints=15, dim=50, layer_hidden=6, layer_output=6)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model
from tqdm.auto import tqdm
def predict(model, data_loader, device):
    predictions = []
    ids = []
    for data in tqdm(data_loader):
        with torch.no_grad():
            # Assuming your data_loader provides a tuple of (fingerprints, adjacency, molecular_size, ids, properties)
            fingerprints, adjacency, molecular_size, proteins, id = data
            fingerprints = [torch.LongTensor(x).to(device) for x in fingerprints]
            proteins = [torch.LongTensor(x).to(device) for x in proteins] 
            adjacency = [torch.LongTensor(x).to(device) for x in adjacency]
            # The 'properties' might contain the ids, or they might be separate, depending on your DataLoader setup
            ids.extend(id)  # Adjust this line based on your ID storage
            scores = model.forward_classifier((fingerprints, adjacency, molecular_size, proteins), train=False)
            predictions.extend([x[1]for x in F.sigmoid(torch.Tensor(scores)).cpu().numpy().tolist()])
    return ids, predictions

def save_predictions(ids, predictions, filename):
    df = pd.DataFrame({'id': ids, 'binds': predictions})
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    from pymongo import MongoClient
    def get_database():
        
        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        CONNECTION_STRING = "mongodb://anonymous-server.local/belka"
        
        # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
        client = MongoClient(CONNECTION_STRING)
        
        # Create the database for our example (we will use the same database throughout the tutorial
        return client['belka']
        
    # Get the database
    dbname = get_database()
    collection = dbname["bindings_raw_test"]
    documents = list(collection.find({}, {'id': 1, '_id': 0}))
    id_map = {i:x['id'] for i,x in enumerate(documents)}

    (task, dataset, radius, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = ('classification', 'any', 1, 50, 6, 6, 256, 256, 1e-4, 0.99, 10, 20, 'dunno')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("./simple_gnn/models/10_model.pth", device)
    test_dataset = pp.MoleculeDatasetMongoTest(filename="BRD4_test.txt", dir_dataset="./", task="classification", atom_dict=atom_dict, bond_dict=bond_dict, fingerprint_dict=fingerprint_dict, edge_dict=edge_dict, radius=0, device=device, id_map = id_map)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=6, worker_init_fn=worker_init_fn,collate_fn=custom_collate)
    ids, predictions = predict(model, test_loader, device)
    save_predictions(ids, predictions, 'predictions.csv')
