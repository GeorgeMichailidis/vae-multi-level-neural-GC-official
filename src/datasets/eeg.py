"""
Customized Dataset class for loading the trajectories
"""

import os
import time
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

class EEGOneSub(Dataset):

    def __init__(
        self,
        size,
        ds_name = 'EEG_EC', ## EEG_EC or EEG_EO
        file_type = 'train',
        subject_id = 0
    ):
        self.data_prefix = f'data_real/{ds_name}/{file_type}/data_'
        self.sample_size = size
        self.list_of_ids = list(range(size))
        self.subject_id = subject_id
        
    def __len__(self):
        return len(self.list_of_ids)

    def __getitem__(self, index):
        idx = self.list_of_ids[index]
        data = np.load(self.data_prefix + str(idx) + '.npy')[self.subject_id]
        data = torch.from_numpy(data).to(dtype=torch.float32).unsqueeze(0)
        
        return data

class EEGMultiSub(Dataset):
    def __init__(
        self,
        size,
        ds_name = 'EEG_EC',
        file_type = 'train',
    ):
        self.data_prefix = f'data_real/{ds_name}/{file_type}/data_'
        self.list_of_ids = list(range(size))
        
    def __len__(self):
        return len(self.list_of_ids)

    def __getitem__(self, index):
        idx = self.list_of_ids[index]
        data = np.load(self.data_prefix + str(idx) + '.npy')
        data = torch.from_numpy(data).to(dtype=torch.float32)
        return data
