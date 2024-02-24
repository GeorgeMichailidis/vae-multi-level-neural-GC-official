
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from .datasets import SimDatasetOneSubTS, SimDatasetOneSubSprings
from .datasets import SimDatasetMultiSubTS, SimDatasetMultiSubSprings

class simDL():
    
    def __init__(self, configs):
        super(simDL, self).__init__()
        self.configs = configs

        if 'Springs' in self.configs['ds_str']:
            self.dsClass = SimDatasetMultiSubSprings
        else:
            self.dsClass = SimDatasetMultiSubTS
        
    def train_dataloader(self):
        trainset = self.dsClass(ds_str=self.configs['ds_str'],
                        size=self.configs['train_size'],
                        file_type='train',
                        graph_key=self.configs['graph_key'],
                        include_diag=self.configs['include_diag'],
                        seed=self.configs.get('seed',0))
        return DataLoader(trainset,
                          batch_size=self.configs['batch_size'],
                          shuffle = self.configs.get('shuffle',True),
                          num_workers=min(self.configs.get('num_workers',4),os.cpu_count()-1),
                          persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        valset = self.dsClass(ds_str=self.configs['ds_str'],
                        size=self.configs['val_size'],
                        file_type='val',
                        graph_key=self.configs['graph_key'],
                        include_diag=self.configs['include_diag'],
                        seed=self.configs.get('seed',0))
        return DataLoader(valset,
                          batch_size=self.configs['batch_size'],
                          shuffle=False,
                          num_workers=min(self.configs.get('num_workers',4),os.cpu_count()-1),
                          persistent_workers=True, drop_last=True)

    def test_dataloader(self, file_type='test', batch_size=None):
        testset = self.dsClass(ds_str=self.configs['ds_str'],
                        size=self.configs['test_size'],
                        file_type=file_type,
                        graph_key=self.configs['graph_key'],
                        include_diag=self.configs['include_diag'],
                        seed=self.configs.get('seed',0))
        return DataLoader(testset,
                          batch_size=batch_size or self.configs['batch_size'],
                          shuffle=False,
                          num_workers=min(self.configs.get('num_workers',4),os.cpu_count()-1),
                          persistent_workers=False)

class simDLOne():
    def __init__(self, configs):
        super(simDLOne, self).__init__()
        self.configs = configs

        if 'Springs' in self.configs['ds_str']:
            self.dsClass = SimDatasetOneSubSprings
        else:
            self.dsClass = SimDatasetOneSubTS
        
    def train_dataloader(self):
        trainset = self.dsClass(ds_str=self.configs['ds_str'],
                        size=self.configs['train_size'],
                        file_type='train',
                        graph_key = self.configs['graph_key'],
                        include_diag=self.configs['include_diag'],
                        seed=self.configs.get('seed',0),
                        subject_id=self.configs['subject_id'])

        return DataLoader(trainset,
                          batch_size=self.configs['batch_size'],
                          shuffle = self.configs.get('shuffle',True),
                          num_workers=min(self.configs.get('num_workers',4),os.cpu_count()-1),
                          persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        valset = self.dsClass(ds_str=self.configs['ds_str'],
                        size=self.configs['val_size'],
                        file_type='val',
                        graph_key=self.configs['graph_key'],
                        include_diag=self.configs['include_diag'],
                        seed=self.configs.get('seed',0),
                        subject_id=self.configs['subject_id'])
        return DataLoader(valset,
                          batch_size=self.configs['batch_size'],
                          shuffle=False,
                          num_workers=min(self.configs.get('num_workers',4),os.cpu_count()-1),
                          persistent_workers=True, drop_last=True)

    def test_dataloader(self, file_type='test', batch_size=None):
        testset = self.dsClass(ds_str=self.configs['ds_str'],
                        size=self.configs['test_size'],
                        file_type=file_type,
                        graph_key=self.configs['graph_key'],
                        include_diag=self.configs['include_diag'],
                        seed=self.configs.get('seed',0),
                        subject_id=self.configs['subject_id'])

        return DataLoader(testset,
                          batch_size= batch_size or self.configs['batch_size'],
                          shuffle=False,
                          num_workers=min(self.configs.get('num_workers',4),os.cpu_count()-1),
                          persistent_workers=False)
