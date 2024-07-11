from collections import defaultdict

import numpy as np
import pandas as pd
import random
from tqdm.auto import tqdm
from rdkit import Chem
from torch.utils.data import Dataset
import torch
import numpy as np
from pymongo import MongoClient
import torch
import pickle


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(task, dataset, radius, device):

    dir_dataset = '../'

    """Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    def create_dataset(filename):

        print(filename)


        """Load a dataset."""
        with open(dir_dataset + filename, 'r') as f:
            smiles_property = f.readline().strip().split()
            data_original = f.read().strip().split('\n')

        """Exclude the data contains '.' in its smiles."""
        data_original = [data for data in data_original
                         if '.' not in data.split()[0]]

        dataset = []

        for data in tqdm(data_original):

            smiles, property = data.strip().split()

            """Create each data with the above defined functions."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            if task == 'classification':
                property = torch.LongTensor([int(property)]).to(device)
            if task == 'regression':
                property = torch.FloatTensor([[float(property)]]).to(device)

            dataset.append((fingerprints, adjacency, molecular_size, property))

        return dataset

    dataset_train = create_dataset('BRD4_train.txt')
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset('BRD4_train.txt')

    N_fingerprints = len(fingerprint_dict)

    return dataset_train, dataset_dev, dataset_test, N_fingerprints


class MoleculeDataset(Dataset):
    def __init__(self, filename, dir_dataset, task, atom_dict, bond_dict, fingerprint_dict, edge_dict, radius, device):
        self.task = task
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.fingerprint_dict = fingerprint_dict
        self.edge_dict = edge_dict
        self.radius = radius
        self.device = device

        with open(filename, 'r') as f:
            self.data_original = f.read().strip().split('\n')
        
        # Exclude data with '.' in smiles
        self.data_original = [data for data in self.data_original if '.' not in data.split()[0]]

    def __len__(self):
        return len(self.data_original)

    def __getitem__(self, idx):

        data = self.data_original[idx].strip().split()
        smiles, property = data

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, self.atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, self.bond_dict)
        fingerprints = extract_fingerprints(self.radius, atoms, i_jbond_dict, self.fingerprint_dict, self.edge_dict)
        adjacency = Chem.GetAdjacencyMatrix(mol)


        return fingerprints, adjacency, molecular_size, int(property)

class MoleculeDatasetMongo(Dataset):
    def __init__(self, filename, dir_dataset, task, atom_dict, bond_dict, fingerprint_dict, edge_dict, radius, device, leng):
        self.task = task
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.fingerprint_dict = fingerprint_dict
        self.edge_dict = edge_dict
        self.radius = radius
        self.device = device
        self.l = leng
        # self.client = client 
        # self.db = self.client['belka']
        # self.collection = self.db["bindings_raw_train"]
        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
        
        # self.db = self.client['belka']
        # self.collection = self.db["bindings_raw_train"]

    
    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        data = self.collection.find_one({"id":idx})
        one_hot_protein = [0,0,0]
        if data['protein_name'] == 'BRD4':
            one_hot_protein[0]=1
        if data['protein_name'] == 'HSA':
            one_hot_protein[1]=1
        if data['protein_name'] == 'sEH':
            one_hot_protein[2]=1


        return data['fingerprints'], pickle.loads(data['adjacency']), data['molecular_size'], int(data['binds']), one_hot_protein
class MoleculeDatasetMongoTest(Dataset):
    def __init__(self, filename, dir_dataset, task, atom_dict, bond_dict, fingerprint_dict, edge_dict, radius, device, id_map):
        self.task = task
        self.atom_dict = atom_dict
        self.bond_dict = bond_dict
        self.fingerprint_dict = fingerprint_dict
        self.edge_dict = edge_dict
        self.radius = radius
        self.device = device
        self.id_map = id_map
        print(f'dataset size {len(id_map)}')
        # self.client = client 
        # self.db = self.client['belka']
        # self.collection = self.db["bindings_raw_train"]
        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
        
        # self.db = self.client['belka']
        # self.collection = self.db["bindings_raw_train"]

    
    def __len__(self):
        return len(self.id_map)

    def __getitem__(self, idx):
        data = self.collection.find_one({"id":self.id_map[idx]})
        one_hot_protein = [0,0,0]
        if data['protein_name'] == 'BRD4':
            one_hot_protein[0]=1
        if data['protein_name'] == 'HSA':
            one_hot_protein[1]=1
        if data['protein_name'] == 'sEH':
            one_hot_protein[2]=1


        return data['fingerprints'], pickle.loads(data['adjacency']), data['molecular_size'],  one_hot_protein,  data['id']



# class MoleculeDatasetMongoP(Dataset):
#     def __init__(self, filename, dir_dataset, task, atom_dict, bond_dict, fingerprint_dict, edge_dict, radius, device, leng):
#         self.task = task
#         self.atom_dict = atom_dict
#         self.bond_dict = bond_dict
#         self.fingerprint_dict = fingerprint_dict
#         self.edge_dict = edge_dict
#         self.radius = radius
#         self.device = device
#         self.l = leng
#         # Provide the mongodb atlas url to connect python to mongodb using pymongo
#         self.connection_string = "mongodb://anonymous-server.local/belka"
#         # Other initializations...

#     def connect_to_db(self):
#         self.client = MongoClient(self.connection_string)
#         db = self.client['belka']
#         collection = db["bindings_raw_train"]
#         return collection
    
#     def __len__(self):
#         return self.l

#     def __getitem__(self, idx):

#         collection = self.connect_to_db()
#         data = collection.find_one({"id": idx})
#         self.client.close()

        
#         one_hot_protein = [0,0,0]
#         if data['protein_name'] == 'BRD4':
#             one_hot_protein[0]=1
#         if data['protein_name'] == 'HSA':
#             one_hot_protein[1]=1
#         if data['protein_name'] == 'sEH':
#             one_hot_protein[2]=1


#         return data['fingerprints'], pickle.loads(data['adjacency']), data['molecular_size'], int(data['binds']), one_hot_protein