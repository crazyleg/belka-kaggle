import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import pyarrow.compute as pc
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack, save_npz, load_npz
from rdkit.Chem import rdFingerprintGenerator
import uuid
import pickle
from sklearn.model_selection import train_test_split

with open('systematic/split_generation/block_splits.pkl', 'rb') as f:
    ((split1_block1, split2_block1, split3_block1, split4_block1, split5_block1), (split1_block23, split2_block23, split3_block23, split4_block23, split5_block23)) = pickle.load(f)


ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

def process_df_and_save(df, prefix):
    if len(df)==0: return
    mol = [Chem.MolFromSmiles(x.replace("Dy","C")) for x in df['molecule_smiles']]
    ecfp = [ecfp_gen.GetFingerprint(x).ToList() for x in mol]
    result = csr_matrix(ecfp)
    one_hot = np.zeros((len(df), 3))
    one_hot[df['protein_name']=="BRD4",0] = 1
    one_hot[df['protein_name']=="HSA",1] = 1
    one_hot[df['protein_name']=="sEH",2] = 1
    result = hstack([result, one_hot])
    uuid_id = str(uuid.uuid4())
    save_npz(f'/mnt/fastssd/belka_data/train_full/{prefix}_{uuid_id}.npz', result.tocsr())
    np.save(f'/mnt/fastssd/belka_data/train_full/{prefix}_{uuid_id}.npl', np.array(df['binds']))


def process_split(x):
    x = x.to_pandas()
    # if x has no test blocks - it is train
    # x_train = x[~(x['buildingblock1_smiles'].isin(split1_block1) | x['buildingblock2_smiles'].isin(split1_block23) | x['buildingblock3_smiles'].isin(split1_block23))] 
    x_train = x
    x_train = pd.concat([x_train[x_train['binds']==1], x_train[x_train['binds']==0].sample(frac=0.15, replace=False, random_state=1)]).drop_duplicates()

    # x_test unique
    # x_test_unique = x[x['buildingblock1_smiles'].isin(split1_block1) & x['buildingblock2_smiles'].isin(split1_block23) & x['buildingblock3_smiles'].isin(split1_block23)]  

    # x_test random 
    # Splitting the DataFrame into 90% train and 10% test
    # if len(x_train)>10:
        # x_train, x_test_random = train_test_split(x_train, test_size=0.1, random_state=42)  # random_state ensures reproducibility
    process_df_and_save(x_train, 'train_full')
        # process_df_and_save(x_test_random, 'random_test')
    # if len(x_test_unique)>1:    
    #     process_df_and_save(x_test_unique, 'unique_test')
    
    return len(x)
   

if __name__ == '__main__':
  
    parquet_file = pq.ParquetFile('train.parquet')
    total_records = parquet_file.metadata.num_rows
    batch_size = 200000
    total_batches = (total_records + batch_size - 1) // batch_size  # Ceiling division  


    with Pool(processes=20) as pool:  # Adjust the number of processes according to your system
        # Use imap_unordered to process data as it is read
        results = list(tqdm(pool.imap_unordered(process_split, parquet_file.iter_batches(batch_size=batch_size)),
                            total=total_batches, desc="Processing batches"))

        # Display total processed batches
        total_processed = sum(results)
        print(f'Total processed batches: {total_processed}')