from multiprocessing import Pool
import pyarrow.parquet as pq
import pymongo
from pymongo import MongoClient
from rdkit import Chem
import multiprocessing
from rdkit.Chem import rdmolops
import torch
from torch_geometric.data import Data, Batch
from tqdm.auto import tqdm
import numpy as np
from gensim.models import word2vec
from bson.binary import Binary
import pickle
import sys
import torch
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
import random
from rdkit import Chem
import uuid
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding
# Define dictionaries
permitted_list_of_atoms = ['C','N','O','H','S','Cl','Br','F','Si','B','I','Dy']
permitted_list_of_atoms_set = set(['C','N','O','H','S','Cl','Br','F','Si','B','I','Dy'])
fingerprint_dict = {i: i for i in range(16)}
edge_dict = {}

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)


def mol_to_graph_data(mol, b, mfpgen, torsiongen,rdkitgen, smiles):

    n_edges = 2*mol.GetNumBonds()
    atoms = mol.GetAtoms()
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    

    atoms_list = []
    for i, atom in enumerate(atoms):
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        is_in_a_ring_enc = [int(atom.IsInRing())]
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                        
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
        atoms_list.append(torch.tensor(atom_feature_vector))
    adj = rdmolops.GetAdjacencyMatrix(mol)
    (rows, cols) = np.nonzero(adj)
    n_edge_features = len(get_bond_features(mol.GetBondBetweenAtoms(0,1)))
    features = torch.stack(atoms_list)
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    # Get adjacency info

    edge_index = torch.tensor(np.array(adj.nonzero())).type(torch.int)

    return Data(x=features.to(torch.float16), edge_index=edge_index, edge_attr = torch.tensor(EF).to(torch.bool), 
                  smiles = smiles

                #   y = torch.tensor(b)[1:].unsqueeze(dim=0)
                )

def process_batch(batch):
    batch_df = batch
    from rdkit.Chem import rdFingerprintGenerator
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    torsiongen = rdFingerprintGenerator.GetTopologicalTorsionGenerator()
    rdkitgen = rdFingerprintGenerator.GetRDKitFPGenerator()

    # Perform chemical transformations and feature extraction
    mol = [Chem.MolFromSmiles(x) for x in batch_df['molecule_smiles']]
    data_g = Batch.from_data_list([mol_to_graph_data(x, None,  mfpgen, torsiongen, rdkitgen, smiles) for x,smiles in zip(mol, batch_df['molecule_smiles'])])
    # data_g = Batch.from_data_list([mol_to_graph_data(x,b,mfpgen,torsiongen,rdkitgen) for x, b in zip(mol, batch_df[['binds_BRD4','binds_HSA','binds_sEH']].itertuples() )])
    torch.save(data_g,f'{FOLDER_TO_SAVE}/{str(uuid.uuid4())}.pt')
    return len(data_g)  


def process_and_insert(batch):
    batch_df = batch.to_pandas()
    if len(batch_df)==0: return 0
    processed_data = process_batch(batch_df)
    return processed_data  # For example, returning the number of processed records

if __name__ == '__main__':

    TRAIN_FILE = '/home/anonymous/belka/systematic/train_data/splits/s2_train.parquet'
    TEST_RANDOM_FILE = '/home/anonymous/belka/systematic/train_data/splits/s2_random.parquet'
    TEST_UNIQUE_FILE = '/home/anonymous/belka/systematic/train_data/splits/s2_unique.parquet'
    FOLDER_TO_SAVE = '/mnt/fastssd/datasets/belka_test_is23'
    TEST_MAIN = '/home/anonymous/belka/systematic/test_data/test_shuffled_is23.parquet'
    TRAIN_MAIN = '/home/anonymous/belka/systematic/train_data/train_shuffled.parquet'
    BATCH_SIZE = 4096

    
    parquet_file = pq.ParquetFile(TEST_MAIN)
    total_records = parquet_file.metadata.num_rows
    batch_size = BATCH_SIZE
    total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
    
    # Create a pool of workers
    with Pool(processes=20) as pool:  # Adjust the number of processes according to your system
        # Use imap_unordered to process data as it is read
        results = list(tqdm(pool.imap_unordered(process_and_insert, parquet_file.iter_batches(batch_size=batch_size)),
                            total=total_batches, desc="Processing batches"))

        # Display total processed batches
        total_processed = sum(results)
        print(f'Total processed batches: {total_processed}')
