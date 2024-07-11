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
import random

from torch_geometric.data import Data, Batch


class FileMegaSet(Dataset):
    def __init__(self, folder, ratio=0.5, ratio_samples=1):
        self.folder = folder
        self.files = os.listdir(folder)
        self.files = random.sample(self.files, int(len(self.files)))
        self.ratio = ratio
        self.ratio_samples = ratio_samples

    def __len__(self):
        return int(len(self.files)*self.ratio_samples)

    def __getitem__(self, index):
        data = torch.load(self.folder+'/'+self.files[index])
        required_samples = int(4096 * self.ratio)
        available_samples = len(data)
        if available_samples >= required_samples:
            random_indices = random.sample(range(available_samples), required_samples)
        else:
            random_indices = random.choices(range(available_samples), k=required_samples)

        return Batch.from_data_list(data.index_select(random_indices))


class FileMegaSetMultiDir(Dataset):
    def __init__(self, folders, ratio=0.5, ratio_samples=1):
        self.files = []
        self.file_paths = []
        self.ratio = ratio
        self.ratio_samples = ratio_samples

        for folder in folders:
            folder_files = os.listdir(folder)
            self.files.extend(folder_files)
            self.file_paths.extend([os.path.join(folder, file) for file in folder_files])

    def __len__(self):
        return int(len(self.files)*self.ratio_samples)

    def __getitem__(self, index):
        data = torch.load(self.file_paths[index])
        required_samples = int(4096 * self.ratio)
        available_samples = len(data)
        if available_samples >= required_samples:
            random_indices = random.sample(range(available_samples), required_samples)
        else:
            random_indices = random.choices(range(available_samples), k=required_samples)
        return Batch.from_data_list(data.index_select(random_indices))