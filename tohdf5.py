import h5py
import torch
import pickle
from pymongo import MongoClient
from tqdm import tqdm
from itertools import islice

def batched_cursor(cursor, batch_size):
    while True:
        batch = list(islice(cursor, batch_size))
        if not batch:
            break
        yield batch
def fetch_and_process_data(batch_size=1000):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['belka']
    raw_data = db['train_metadata']
    full_molecules = db['full_molecules']

    # Create a cursor for raw_data with batch size control
    cursor = raw_data.find({}, {'_id': 0, 'molecule_smiles': 1, 'protein_name': 1, 'binds': 1}).batch_size(batch_size)
    protein_map = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
    total_docs=1000000
    with h5py.File('data.hdf5', 'w') as f:
        # Placeholder shapes, adjust these based on your specific data structures
        data_shape = (total_docs, 3 + 1 + 2048)  # Example: 3 protein features, 1 molecular size, 2048 ECFP features
        data_dset = f.create_dataset('data', shape=data_shape, dtype='f', maxshape=(None, data_shape[1]))
        targets_dset = f.create_dataset('targets', shape=(total_docs,), dtype='f', maxshape=(None,))

        idx = 0
        for batch in tqdm(batched_cursor(cursor, batch_size), total=(total_docs // batch_size) + 1):
            smiles_list = [x['molecule_smiles'] for x in batch]
            molecules = list(full_molecules.find({'molecule_smiles': {'$in': smiles_list}}, {'_id': 0, 'ecfp': 1, 'molecular_size': 1, 'molecule_smiles': 1}))
            molecules_dict = {mol['molecule_smiles']: mol for mol in molecules}

            for raw_data in batch:
                molecule = molecules_dict.get(raw_data['molecule_smiles'], None)
                if molecule:
                    one_hot_protein = [0] * 3
                    if raw_data['protein_name'] in protein_map:
                        one_hot_protein[protein_map[raw_data['protein_name']]] = 1

                    ecfp = pickle.loads(molecule['ecfp'])
                    molecular_size = [molecule['molecular_size'] / 100]

                    # Combine into one tensor before writing
                    combined_data = torch.cat([
                        torch.tensor(one_hot_protein).float(),
                        torch.tensor(molecular_size).float(),
                        torch.tensor(ecfp).float()
                    ])

                    if idx >= len(data_dset):
                        data_dset.resize((idx + batch_size, data_shape[1]))
                        targets_dset.resize((idx + batch_size,))

                    data_dset[idx] = combined_data.numpy()
                    targets_dset[idx] = int(raw_data['binds'])
                    idx += 1

if __name__ == '__main__':
    fetch_and_process_data()
