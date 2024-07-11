from multiprocessing import Pool
import pyarrow.parquet as pq
import pymongo
from pymongo import MongoClient
from rdkit import Chem
import multiprocessing
from tqdm.auto import tqdm
from simple_gnn.preprocess import create_atoms, create_ijbonddict, extract_fingerprints
import numpy as np
from bson.binary import Binary
import pickle
import sys
import random
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

TRAIN = True 

def process_batch(batch):
    
    with open('block_splits.pkl', 'rb') as f:
        ((split1_block1, split2_block1, split3_block1, split4_block1, split5_block1), (split1_block23, split2_block23, split3_block23, split4_block23, split5_block23)) = pickle.load(f)
    # Convert batch to pandas DataFrame
    x = batch.copy()
    # Add features to DataFrame

    if TRAIN:
        for i,(block1, block23) in enumerate([(split1_block1,split1_block23),(split2_block1,split2_block23),(split3_block1,split3_block23),(split4_block1,split4_block23),(split5_block1,split5_block23)]):
        # Mark entries for test1_unique based on the loaded splits
            x.loc[:,f'test{i}_unique'] = 0
            x.loc[:,f'test{i}_random'] = 0
            x.loc[:,f'train{i}'] = 0
            # ALL = test unique + edge + train unique
            mask_unique = (
                x['buildingblock1_smiles'].isin(block1) &
                x['buildingblock2_smiles'].isin(block23) &
                x['buildingblock3_smiles'].isin(block23)
            )
            if mask_unique.sum()>0:
                pass
            x.loc[mask_unique,f'test{i}_unique'] = 1

            mask_edge = (
                (x['buildingblock1_smiles'].isin(block1) |
                x['buildingblock2_smiles'].isin(block23) |
                x['buildingblock3_smiles'].isin(block23)) & (x[f'test{i}_unique']==0)
            )
            
            train_index = x[~(mask_edge | mask_unique)]
            random_split_mask = train_index.sample(frac=0.05, random_state=1).index
            x.loc[x.index.isin(random_split_mask), f'test{i}_random'] = 1
            x.loc[x.index.isin(train_index.index) & x[f'test{i}_random']==0 , f'train{i}'] = 1
        
    # Return the processed DataFrame as dictionary records for insertion
    return x.to_dict('records')

def insert_into_db(data_records):
    # MongoDB Setup
    client = MongoClient('localhost', 27017)  # Adjust host and port as necessary
    db = client.belka
    collection = db.train_metadata
    collection.insert_many(data_records)

def process_and_insert(batch):
    batch_df = batch.to_pandas()
    # batch_df = batch_df[batch_df['id']>9166999] 
    if len(batch_df)==0: return 0
    processed_data = process_batch(batch_df)
    if len(processed_data)>0:
        insert_into_db(processed_data)
    return len(processed_data)  # For example, returning the number of processed records

if __name__ == '__main__':
    # Open the Parquet file

    if TRAIN:
        parquet_file = pq.ParquetFile('train.parquet')
    else:
        parquet_file = pq.ParquetFile('test.parquet') 

    total_records = parquet_file.metadata.num_rows
    batch_size = 100000
    total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division
    count=0
 
    
    # Create a pool of workers
    with Pool(processes=28) as pool:  # Adjust the number of processes according to your system
        # Use imap_unordered to process data as it is read
        results = list(tqdm(pool.imap_unordered(process_and_insert, parquet_file.iter_batches(batch_size=batch_size)),
                            total=total_batches, desc="Processing batches"))

        # Display total processed batches
        total_processed = sum(results)
        print(f'Total processed batches: {total_processed}')
