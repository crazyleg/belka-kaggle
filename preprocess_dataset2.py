from multiprocessing import Pool
import pyarrow.parquet as pq
import pymongo
from pymongo import MongoClient
from rdkit import Chem
import multiprocessing
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm
from simple_gnn.preprocess import create_atoms, create_ijbonddict, extract_fingerprints
import numpy as np
# from gensim.models import word2vec
from bson.binary import Binary
import pickle
import sys
# from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
import random
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import pandas as pd
# Define dictionaries
atom_dict = {'C': 0, 'N': 1, ('C', 'aromatic'): 2, ('N', 'aromatic'): 3, 'O': 4, 'H': 5, ('S', 'aromatic'): 6, 'Cl': 7, 'S': 8, ('O', 'aromatic'): 9, 'Br': 10, 'F': 11, 'Si': 12, 'B': 13, 'I': 14, 'Dy': 15}
bond_dict = {'TRIPLE': 0, 'SINGLE': 1, 'AROMATIC': 2, 'DOUBLE': 3}
fingerprint_dict = {i: i for i in range(16)}
edge_dict = {}
TRAIN = False 
# model = word2vec.Word2Vec.load('model_300dim.pkl')

with open('dicts1.pickle', 'rb') as handle:
    atom_dict1, bond_dict1, fingerprint_dict1, edge_dict1 = pickle.load(handle)

with open('dicts2.pickle', 'rb') as handle:
    atom_dict2, bond_dict2, fingerprint_dict2, edge_dict2 = pickle.load(handle)

with open('unique_binds.pkl', 'rb') as handle:
    unique_binds = pickle.load(handle)

def mol_to_graph_data(mol):
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    num_features = 3  # example: atomic number, degree, implicit valence
    features = torch.zeros(num_atoms, num_features)

    for i, atom in enumerate(atoms):
        features[i, 0] = atom.GetAtomicNum()
        features[i, 1] = atom.GetDegree()
        features[i, 2] = atom.GetImplicitValence()

    # Get adjacency info
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(adj.nonzero()).type(torch.long)

    # Create edge features if necessary
    # For simplicity, this example will not include edge features

    return Data(x=features, edge_index=edge_index)

def process_batch(batch):
    batch_df = batch

    # with open('split.pickle', 'rb') as handle:
    #     (l1, l2, l3) = pickle.load(handle)
    # Convert batch to pandas DataFrame
    # positives = batch[batch['molecule_smiles'].isin(unique_binds)] 
    # random = batch.sample(frac=0.1, replace=False, random_state=1)
    # batch_df = pd.concat([positives, random]).drop_duplicates()


    # Perform chemical transformations and feature extraction
    mol = [Chem.MolFromSmiles(x.replace("Dy","C")) for x in batch_df['molecule_smiles']]
    # atoms = [create_atoms(x, atom_dict) for x in mol]
    # molecular_size = [len(x) for x in atoms]
    # i_jbond_dict = [create_ijbonddict(x, bond_dict) for x in mol]

    # atoms1 = [create_atoms(x, atom_dict1) for x in mol]
    # i_jbond_dict1 = [create_ijbonddict(x, bond_dict1) for x in mol]

    # atoms2 = [create_atoms(x, atom_dict2) for x in mol]
    # i_jbond_dict2 = [create_ijbonddict(x, bond_dict2) for x in mol]

    # fingerprints = [Binary(pickle.dumps(extract_fingerprints(0, x, y, fingerprint_dict, edge_dict).astype(np.uint8), protocol=pickle.HIGHEST_PROTOCOL)) for (x, y) in zip(atoms, i_jbond_dict)]
    # adjacency = [Binary(pickle.dumps(Chem.GetAdjacencyMatrix(x).astype(bool), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]

    # fingerprints1 = [Binary(pickle.dumps(extract_fingerprints(1, x, y, fingerprint_dict1, edge_dict1).astype(np.uint8), protocol=pickle.HIGHEST_PROTOCOL)) for (x, y) in zip(atoms1, i_jbond_dict1)]
    # adjacency1 = [Binary(pickle.dumps(Chem.GetAdjacencyMatrix(x).astype(bool), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    
    # fingerprints2 = [Binary(pickle.dumps(extract_fingerprints(2, x, y, fingerprint_dict2, edge_dict2).astype(np.uint8), protocol=pickle.HIGHEST_PROTOCOL)) for (x, y) in zip(atoms2, i_jbond_dict2)]
    # adjacency2 = [Binary(pickle.dumps(Chem.GetAdjacencyMatrix(x).astype(bool), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    data_g = [Binary(pickle.dumps(mol_to_graph_data(x), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    
    # ecfp = [Binary(pickle.dumps(list(map(bool, AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    # ecfp2 = [Binary(pickle.dumps(list(map(bool, AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=2048).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    # ecfp3 = [Binary(pickle.dumps(list(map(bool, AllChem.GetMorganFingerprintAsBitVect(x, radius=4, nBits=2048).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    # maccs = [Binary(pickle.dumps(list(map(bool,MACCSkeys.GenMACCSKeys(x).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    # RDKFingerprint = [Binary(pickle.dumps(list(map(bool, Chem.rdmolops.RDKFingerprint(x).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]
    # Torsion = [Binary(pickle.dumps(list(map(bool, Chem.rdMolDescriptors.GetHashedTopologicalTorsionFingerprint(x).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol] 
    # Avalon = [Binary(pickle.dumps(list(map(bool, GetAvalonFP(x).ToList())), protocol=pickle.HIGHEST_PROTOCOL)) for x in mol]  


    # mols = [ Chem.MolFromSmiles(i) for i in batch_df['molecule_smiles']]
    # sentences = [mol2alt_sentence(mol, 1) for mol in mols]
    # sum_mol_vectors = [Binary(pickle.dumps(list(np.array([model.wv.get_vector(i) for i in sentence if i in model.wv.key_to_index.keys()]).sum(axis=0).astype(np.float16)))) for sentence in sentences]
    # mean_mol_vectors = [Binary(pickle.dumps(list(np.array([model.wv.get_vector(i) for i in sentence if i in model.wv.key_to_index.keys()]).mean(axis=0).astype(np.float16)))) for sentence in sentences]

    # mol = [Chem.AddHs(Chem.MolFromSmiles(x.replace("Dy","C"))) for x in batch_df['molecule_smiles']]

    # # Generate and optimize conformation
    # [AllChem.EmbedMolecule(x) for x in mol]
    # [AllChem.MMFFOptimizeMolecule(x) for x in mol]

    # # Calculate properties
    # mol_weight = [Descriptors.MolWt(x) for x in mol]
    # log_p = [Descriptors.MolLogP(x) for x in mol]
    # tpsa = [rdMolDescriptors.CalcTPSA(x) for x in mol]

    # Add features to DataFrame
    # batch_df['fingerprints'] = fingerprints
    # batch_df['fingerprints1'] = fingerprints1
    # batch_df['fingerprint2'] = fingerprints2
    # batch_df['adjacency'] =  adjacency
    # batch_df['adjacency1'] =  adjacency1
    # batch_df['adjacency2'] =  adjacency2
    # batch_df['molecular_size'] = molecular_size
    # batch_df['molecular_size1'] = molecular_size1
    # batch_df['molecular_size2'] = molecular_size2
    # batch_df['ecfp'] = ecfp
    # batch_df['ecfp2'] = ecfp2
    # batch_df['ecfp3'] = ecfp3

    # batch_df['maccs'] = maccs 
    # batch_df['RDKFingerprint'] = RDKFingerprint 
    # batch_df['Torsion'] = Torsion 
    # batch_df['Avalon'] = Avalon 
    batch_df['data_g'] = data_g

    # batch_df['sum_mol_vectors'] = sum_mol_vectors 
    # batch_df['mean_mol_vectors'] = mean_mol_vectors 

    # batch_df['mol_weight'] = mol_weight 
    # batch_df['log_p'] = log_p 
    # batch_df['tpsa'] = tpsa 

    # Return the processed DataFrame as dictionary records for insertion
    return batch_df.to_dict('records')

def insert_into_db(data_records):
    # MongoDB Setup
    client = MongoClient('localhost', 27017)  # Adjust host and port as necessary
    db = client.belka
    collection = db.full_molecules_fingerprints_test
    collection.insert_many(data_records)

def process_and_insert(batch):
    batch_df = batch.to_pandas()
    if len(batch_df)==0: return 0
    processed_data = process_batch(batch_df)
    insert_into_db(processed_data)
    return len(processed_data)  # For example, returning the number of processed records

if __name__ == '__main__':
    # Open the Parquet file

    if TRAIN:
        parquet_file = pq.ParquetFile('train.parquet')
    else:
        parquet_file = pq.ParquetFile('test.parquet') 

    total_records = parquet_file.metadata.num_rows
    batch_size = 8000
    total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
    
    
    # Create a pool of workers
    with Pool(processes=1) as pool:  # Adjust the number of processes according to your system
        # Use imap_unordered to process data as it is read
        results = list(tqdm(pool.imap_unordered(process_and_insert, parquet_file.iter_batches(batch_size=batch_size)),
                            total=total_batches, desc="Processing batches"))

        # Display total processed batches
        total_processed = sum(results)
        print(f'Total processed batches: {total_processed}')
