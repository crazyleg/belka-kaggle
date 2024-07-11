import requests
from torch.utils.data import Dataset
import torch
import pickle
from tqdm.auto import tqdm
import random
import msgpack
import numpy as np
import random 
import torch_geometric.utils

from torch_geometric.data import Data

class APIBasedMoleculeDataset(Dataset):
    def __init__(self, is_test=False, dataset_factor=0.1):
        self.is_test = is_test
        self.dataset_factor = dataset_factor
        self.protein_map = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
        random.seed(10)
        all_ids = list(range(1, 33601111))
        train_ids = random.sample(all_ids, int(33601111*0.95))
        test_ids = list(set(all_ids) - set(train_ids))
        self.ids = train_ids if is_test==False else test_ids 

    def __len__(self):
        return int(len(self.ids) * self.dataset_factor)

    def __getitem__(self, index):
        result = list(self.full_molecules.find({'seq_id':self.ids[index]},{
                    '_id': 0,
                    'protein_name': 1,
                    'binds': 1,
                    'data_g': 1,
                    # 'adjacency2': 1,
                    # 'molecular_size': 1

                }))
        result = result[0] 
        # fingerprints = torch.tensor(pickle.loads(result['fingerprint2']),dtype=torch.int) 
        # adjacency = pickle.loads(result['adjacency2'])
        # adjacency_t = torch.tensor(adjacency, dtype=torch.long).nonzero().t().contiguous() 
        target = result['binds']
        protein_indicator =  torch.zeros(3).float()
        protein_indicator[self.protein_map[result['protein_name']]] = 1
        data = pickle.loads(result['data_g'])
        data.validate(raise_on_error=True)

        return data, protein_indicator, target 


class APIBasedMoleculePredictDataset(Dataset):
    def __init__(self, ):
        
        self.protein_map = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
        with open('unique_test_ids.pkl','rb') as f:
            self.ids = pickle.load(f) 

    def __len__(self):
        return int(len(self.ids) )

    def __getitem__(self, index):
        result = list(self.raw_data.find({'id':self.ids[index]},{
                    '_id': 0,
                    'id' : 1,
                    'protein_name': 1,
                    'data_g': 1,
                    # 'adjacency2': 1,
                    # 'molecular_size': 1

                }))
        result = result[0] 
        # fingerprints = torch.tensor(pickle.loads(result['fingerprint2']),dtype=torch.int) 
        # adjacency = pickle.loads(result['adjacency2'])
        # adjacency_t = torch.tensor(adjacency, dtype=torch.long).nonzero().t().contiguous() 
        target = result['id']
        protein_indicator =  torch.zeros(3).float()
        protein_indicator[self.protein_map[result['protein_name']]] = 1
        data = pickle.loads(result['data_g'])
        data.validate(raise_on_error=True)

        return data, protein_indicator, target 