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
import os

from torch_geometric.data import Data


class FileMegaSet(Dataset):
    def __init__(self, folder, ratio=1):
        self.folder = folder
        self.files = os.listdir(folder)
        self.files = random.sample(self.files, int(len(self.files)*ratio))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = torch.load(self.folder+'/'+self.files[index])
        
        return data 


class FileMegaSetMultiDir(Dataset):
    def __init__(self, folders):
        self.files = []
        self.file_paths = []

        for folder in folders:
            folder_files = os.listdir(folder)
            self.files.extend(folder_files)
            self.file_paths.extend([os.path.join(folder, file) for file in folder_files])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        return data