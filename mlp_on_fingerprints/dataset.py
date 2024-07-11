import requests
from torch.utils.data import Dataset
import torch
import pickle
from tqdm.auto import tqdm
import random
import msgpack
import numpy as np

class APIBasedMoleculeDataset(Dataset):
    def __init__(self, is_test=False, dataset_factor=0.1, batch_size = 2048):
        self.batch_size = batch_size
        self.is_test = is_test
        self.dataset_factor = dataset_factor
        self.protein_map = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
        random.seed(10)
        all_ids = list(range(1, 11288087))
        train_ids = random.sample(all_ids, int(11288087*0.95))
        test_ids = list(set(all_ids) - set(train_ids))
        self.ids = train_ids if is_test==False else test_ids 


        # Define the sizes for different vectors (example sizes, need to be set according to your actual data)
        self.vector_sizes = {
            'ecfp': 2048,
            'ecfp2': 2048,
            'ecfp3': 2048,
            'maccs': 167,
            'Avalon': 512,
            'RDKFingerprint': 2048, 
            'Torsion': 2048,
            'sum_mol_vectors': 300,
            'mean_mol_vectors': 300
        }
        

    def __len__(self):
        return int((len(self.ids) // self.batch_size)*self.dataset_factor) + 1

    def __getitem__(self, index):
     
        batch_ids = random.sample(self.ids, self.batch_size)
        query = {'sequential_id': {'$in': batch_ids}}
        pipeline = [
            {
                '$match': query  # Your existing filter query if any
            },
            {'$lookup': {
                'from': 'train_metadata',
                'localField': 'molecule_smiles',
                'foreignField': 'molecule_smiles',
                'as': 'metadata'
            }},
            {'$unwind': '$metadata'},
            {
                '$project': {
                    '_id': 0,
                    'protein_name': '$metadata.protein_name',
                    'binds': '$metadata.binds',
                    'ecfp': 1,
                    'ecfp2': 1,
                    'ecfp3': 1,
                    'maccs': 1,
                    'RDKFingerprint': 1,
                    'Torsion': 1,
                    'Avalon': 1,
                    'sum_mol_vectors': 1,
                    'mean_mol_vectors': 1,
                }
            }
        ]

        batch_data = list(self.full_molecules.aggregate(pipeline).batch_size(1000))

        tower_tensors = [torch.zeros((len(batch_data), size)) for size in self.vector_sizes.values()]
        binds = torch.zeros(len(batch_data)).float()
        proteins = torch.zeros((len(batch_data), 3))

        for idx, item in enumerate(batch_data):
            binds[idx] = item['binds']
            proteins[idx,self.protein_map[item['protein_name']]] = 1 
            for tower, key in zip(tower_tensors, self.vector_sizes.keys()):
                tower[idx] = torch.tensor(pickle.loads(item[key]), dtype=torch.float32)
                

        return tower_tensors[0],tower_tensors[1],tower_tensors[2],tower_tensors[3],tower_tensors[4],tower_tensors[5],tower_tensors[6],tower_tensors[7],tower_tensors[8], proteins, binds
import pickle
class APIBasedMoleculeTestDataset(Dataset):
    def __init__(self,  batch_size = 2048):
        self.batch_size = batch_size
        self.protein_map = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
        with open('unique_test_ids.pkl','rb') as f:
            self.ids = pickle.load(f)


        # Define the sizes for different vectors (example sizes, need to be set according to your actual data)
        self.vector_sizes = {
            'ecfp': 2048,
            'ecfp2': 2048,
            'ecfp3': 2048,
            'maccs': 167,
            'Avalon': 512,
            'RDKFingerprint': 2048, 
            'Torsion': 2048,
            'sum_mol_vectors': 300,
            'mean_mol_vectors': 300
        }
        

    def __len__(self):
        return int((len(self.ids) // self.batch_size)) + 1

    def __getitem__(self, index):
     
        batch_ids = self.ids[index*self.batch_size: index*self.batch_size + self.batch_size]
        query = {'id': {'$in': batch_ids}}
        pipeline = [
            {
                '$match': query  # Your existing filter query if any
            },
            # {'$lookup': {
            #     'from': 'test_metadata',
            #     'localField': 'id',
            #     'foreignField': 'id',
            #     'as': 'metadata'
            # }},
            # {'$unwind': '$metadata'},
            {
                '$project': {
                    '_id': 0,
                    'id': 1,
                    'protein_name': 1,
                    'ecfp': 1,
                    'ecfp2': 1,
                    'ecfp3': 1,
                    'maccs': 1,
                    'RDKFingerprint': 1,
                    'Torsion': 1,
                    'Avalon': 1,
                    'sum_mol_vectors': 1,
                    'mean_mol_vectors': 1,
                }
            }
        ]

        batch_data = list(self.full_molecules_test.aggregate(pipeline).batch_size(1000))

        tower_tensors = [torch.zeros((len(batch_data), size)) for size in self.vector_sizes.values()]
        binds = torch.zeros(len(batch_data), dtype=torch.int) 
        proteins = torch.zeros((len(batch_data), 3))

        for idx, item in enumerate(batch_data):
            binds[idx] = int(item['id'])
            proteins[idx,self.protein_map[item['protein_name']]] = 1 
            for tower, key in zip(tower_tensors, self.vector_sizes.keys()):
                tower[idx] = torch.tensor(pickle.loads(item[key]), dtype=torch.float32)
        del batch_data
                

        return tower_tensors[0],tower_tensors[1],tower_tensors[2],tower_tensors[3],tower_tensors[4],tower_tensors[5],tower_tensors[6],tower_tensors[7],tower_tensors[8], proteins, binds
