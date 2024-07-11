import requests
from torch.utils.data import Dataset
import torch
import pickle
from tqdm.auto import tqdm
import random
import msgpack
import numpy as np

class SubMoleculeDatasetMongoBatched(Dataset):
    def __init__(self, device, batch_size, dataset_factor=0.1, is_test_unique=False, is_test_random=False):
        self.device = device
        self.batch_size = batch_size
        self.is_test_unique = is_test_unique
        self.is_test_random = is_test_random
        self.dataset_factor = dataset_factor
        self.protein_map = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
        if (not is_test_unique) and (not is_test_random):
            with open('ids_train.pkl', 'rb') as f:
                self.idx = pickle.load(f)
        if is_test_unique:
            with open('ids_test_u.pkl', 'rb') as f:
                self.idx = pickle.load(f)
        if is_test_random:
            with open('ids_test1.pkl', 'rb') as f:
                self.idx = pickle.load(f)


    def __len__(self):
        return int(len(self.idx)*self.dataset_factor) // self.batch_size + (1 if len(self.idx) % self.batch_size != 0 else 0)
    
    def __getitem__(self, index):
        batch_ids = random.sample(self.idx, self.batch_size) 
        query = {'id': {'$in': batch_ids}}


        # Aggregation pipeline
        pipeline = [
            {
                '$match': query  # Your existing filter query if any
            },
            {'$lookup': {
                'from': 'molecules_collection',
                'localField': 'buildingblock1_smiles',
                'foreignField': 'SMILES',
                'as': 'molecule1_details'
            }},
            {'$unwind': '$molecule1_details'},
            {'$lookup': {
                'from': 'molecules_collection',
                'localField': 'buildingblock2_smiles',
                'foreignField': 'SMILES',
                'as': 'molecule2_details'
            }},
            {'$unwind': '$molecule2_details'},
            {'$lookup': {
                'from': 'molecules_collection',
                'localField': 'buildingblock3_smiles',
                'foreignField': 'SMILES',
                'as': 'molecule3_details'
            }},
            {'$unwind': '$molecule3_details'},
            {
                '$project': {
                    '_id': 0,
                    'protein_name': 1,
                    'binds': 1,
                    'mol1': '$molecule1_details',
                    'mol2': '$molecule2_details',
                    'mol3': '$molecule3_details',
                }
            }
        ]

        batch_data = self.raw_data.aggregate(pipeline).batch_size(1000)

        features = torch.zeros((self.batch_size, 3, 6876), dtype=torch.float32)
        targets = torch.zeros(self.batch_size, dtype=torch.float32)
        mand_fs = ['LogP', 'TPSA','Asphericity','Eccentricity','Inertial_Shape_Factor','Molecular_Weight','NPR1','NPR2','Num_Rotatable_Bonds','PMI1','PMI2','PMI3','Radius_of_Gyration','Spherocity_Index','LogD']
        mand_fs_l = len(mand_fs)
        sec_fs = ['MW', '#Heavy atoms', '#Aromatic heavy atoms', 'Fraction Csp3', '#Rotatable bonds', '#H-bond acceptors', '#H-bond donors', 'MR', 'TPSA', 'iLOGP', 'XLOGP3', 'WLOGP', 'MLOGP', 'Silicos-IT Log P', 'Consensus Log P', 'ESOL Log S', 'ESOL Solubility (mg/ml)', 'ESOL Solubility (mol/l)', 'Ali Log S', 'Ali Solubility (mg/ml)', 'Ali Solubility (mol/l)',  'Silicos-IT LogSw', 'Silicos-IT Solubility (mg/ml)', 'Silicos-IT Solubility (mol/l)',   'log Kp (cm/s)', 'Lipinski #violations', 'Ghose #violations', 'Veber #violations', 'Egan #violations', 'Muegge #violations', 'Bioavailability Score', 'PAINS #alerts', 'Brenk #alerts', 'Leadlikeness #violations', 'Synthetic Accessibility']
        vectors = ['ECFP','MACCS','Avalon','RDK','Torsion']
        for i, raw_data in enumerate(batch_data):
            targets[i] = int(raw_data['binds']) 
            for tower_i, pointer in enumerate(['mol1','mol2','mol3']):
                counter = 0
                features[i, tower_i, self.protein_map[raw_data['protein_name']]] = 1
                for f_i, feature in enumerate(mand_fs):
                    features[i, tower_i, f_i] = raw_data[pointer][feature]
                    counter += 1
                for f_i, feature in enumerate(sec_fs):
                    features[i, tower_i, f_i + mand_fs_l] = float(raw_data[pointer]['properties'][feature])
                    counter += 1
                for v in vectors:
                    features[i, tower_i, counter:counter+len(self.submolecules_data[raw_data[pointer]['SMILES']][v])] = torch.tensor(self.submolecules_data[raw_data[pointer]['SMILES']][v])
                    counter += len (self.submolecules_data[raw_data[pointer]['SMILES']][v])

        return features[:,0,:], features[:,1,:], features[:,2,:], targets
