import pymongo
from pymongo import MongoClient
from rdkit import Chem
import multiprocessing
from tqdm.auto import tqdm
from simple_gnn.preprocess import create_atoms, create_ijbonddict, extract_fingerprints
import numpy as np
from bson.binary import Binary
import pickle

# Define dictionaries
atom_dict = {'C': 0, 'N': 1, ('C', 'aromatic'): 2, ('N', 'aromatic'): 3, 'O': 4, 'H': 5, ('S', 'aromatic'): 6, 'Cl': 7, 'S': 8, ('O', 'aromatic'): 9, 'Br': 10, 'F': 11, 'Si': 12, 'B': 13, 'I': 14}
bond_dict = {'TRIPLE': 0, 'SINGLE': 1, 'AROMATIC': 2, 'DOUBLE': 3}
fingerprint_dict = {i: i for i in range(15)}
edge_dict = {}

# MongoDB Setup
client = MongoClient('localhost', 27017)  # Adjust host and port as necessary
db = client.belka
collection = db.bindings_raw_test

# Initialize the statuses
def process_document(doc, train=False):
    try:
        if train:
            smiles, property = doc['molecule_smiles'].replace("Dy","C"), doc['binds']
        else:
            smiles  = doc['molecule_smiles'].replace("Dy","C")
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(0, atoms, i_jbond_dict, fingerprint_dict, edge_dict)
        adjacency = Chem.GetAdjacencyMatrix(mol)

        # Update document with results
        update_fields = {
            'fingerprints': fingerprints.tolist(),
            'adjacency': Binary(pickle.dumps(adjacency.astype(np.uint8), protocol=2), subtype=128 ),
            'molecular_size': molecular_size,
            'status': 'processed'
        }
        collection.update_one({'id': doc['id']}, {'$set': update_fields})
    except Exception as e:
        # Update document status to failed
        collection.update_one({'id': doc['id']}, {'$set': {'status': 'failed', 'error': str(e)}})

# Batch processing with multiprocessing
def fetch_and_process_batch():
    batch_size = 1_000_00  # Set the batch size as per your needs
    while True:
        # Fetch a batch of documents
        docs = collection.find({'status': 'awaiting processing'}).limit(batch_size)
        docs = list(docs)
        if len(docs) == 0:
            break
        collection.update_many({'id': {'$in': [doc['id'] for doc in docs]}}, {'$set': {'status': 'processing'}})

        # Process documents in parallel
        pool = multiprocessing.Pool(processes=16)
        list(tqdm(pool.imap_unordered(process_document, docs), total=len(docs)))
        pool.close()
        pool.join()

# Run batch processing
fetch_and_process_batch()