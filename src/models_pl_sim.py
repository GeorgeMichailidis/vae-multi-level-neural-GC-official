import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics.classification import BinaryAccuracy

from .models_pl import MultiSubVAE, OneSubVAE
from .modules import NumericalAccuracy
from .datasets import SimDatasetOneSubTS, SimDatasetOneSubSprings
from .datasets import SimDatasetMultiSubTS, SimDatasetMultiSubSprings

_EPS = 1.0e-4

class simMultiSubVAE(MultiSubVAE):

    def __init__(self, configs):
        super(simMultiSubVAE, self).__init__(configs)
        self.dsClass = SimDatasetMultiSubSprings if 'Springs' in self.configs['ds_str'] else SimDatasetMultiSubTS
        if self.configs['graph_type'] == 'binary':
            self.train_meter = BinaryAccuracy(multidim_average='global')
            self.val_meter = BinaryAccuracy(multidim_average='global')
            self.train_meter_bar = NumericalAccuracy(normalize=False)
            self.val_meter_bar = NumericalAccuracy(normalize=False)
        else:
            self.train_meter = NumericalAccuracy()
            self.val_meter = NumericalAccuracy()
            self.train_meter_bar = NumericalAccuracy(normalize=True)
            self.val_meter_bar = NumericalAccuracy(normalize=True)
    
    def log_graph_performance(self, graph_by_subject, graph_bar, mean_zq, param1_zbarq, param2_zbarq, mode='train'):
        
        if self.configs['graph_type'] == 'binary':
            graph_by_subject = 1 * (graph_by_subject!=0)
        
        subject_meter = getattr(self, f'{mode}_meter', None)
        subject_meter.update(mean_zq.view(-1,mean_zq.shape[-1]), graph_by_subject.view(-1,graph_by_subject.shape[-1]))
        self.log(f'{mode}_acc', subject_meter, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        bar_mean = param1_zbarq if self.configs['graph_type'] == 'numeric' else param1_zbarq/(param1_zbarq + param2_zbarq + _EPS)
        bar_meter = getattr(self, f'{mode}_meter_bar', None)
        bar_meter.update(bar_mean.view(-1,bar_mean.shape[-1]), graph_bar.view(-1,graph_bar.shape[-1]))
        self.log(f'{mode}_acc_bar', bar_meter, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return
    
    def training_step(self, batch_data, batch_idx):

        data, graph_by_subject, graph_bar = batch_data
        
        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        
        targets = self.get_targets_from_data(data)
        nll_loss, kl1, kl2 = self.loss_fn(targets, mu_x, var_x,
                                          mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp)
        beta = self.get_beta()
        loss = nll_loss + beta * self.kl_multiplier * (kl1 + kl2)
        
        self.log_dict({'train_nll_loss' : nll_loss,
                       'train_kl1': kl1,
                       'train_kl2': kl2,
                       'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_graph_performance(graph_by_subject, graph_bar, mean_zq, param1_zbarq, param2_zbarq, mode='train')
        
        return loss

    def validation_step(self, batch_data, batch_idx):

        data, graph_by_subject, graph_bar = batch_data
        
        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        
        targets = self.get_targets_from_data(data)
        nll_loss, kl1, kl2 = self.loss_fn(targets, mu_x, var_x,
                                          mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp)

        beta = self.get_beta()
        loss = nll_loss + beta * self.kl_multiplier * (kl1 + kl2)
        self.log_dict({'val_nll_loss' : nll_loss,
                       'val_kl1': kl1,
                       'val_kl2': kl2,
                       'val_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                       
        self.log_graph_performance(graph_by_subject, graph_bar, mean_zq, param1_zbarq, param2_zbarq, mode='val')
        return
        
    def predict_step(self, batch_data, batch_idx):
        
        data, graph_by_subject, graph_bar = batch_data
        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        mean_zbarq = param1_zbarq if self.configs['graph_type'] == 'numeric' else param1_zbarq/(param1_zbarq + param2_zbarq + _EPS)
        
        return {'mu_x': mu_x, 'mean_zq': mean_zq, 'mean_zbarq': mean_zbarq,
                'target_x': self.get_targets_from_data(data)}

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


class simOneSubVAE(OneSubVAE):

    def __init__(self, configs):
        super(simOneSubVAE, self).__init__(configs)
        self.dsClass = SimDatasetOneSubSprings if 'Springs' in self.configs['ds_str'] else SimDatasetOneSubTS
        if self.configs['graph_type'] == 'binary':
            self.train_meter = BinaryAccuracy(multidim_average='global')
            self.val_meter = BinaryAccuracy(multidim_average='global')
        else:
            self.train_meter = NumericalAccuracy()
            self.val_meter = NumericalAccuracy()
    
    def log_graph_performance(self, graph, mean_zq, mode='train'):
        
        if self.configs['graph_type'] == 'binary':
            graph = 1 * (graph!=0)

        meter_in_use = getattr(self, f'{mode}_meter', None)
        meter_in_use.update(mean_zq.view(-1,mean_zq.shape[-1]), graph.view(-1,graph.shape[-1]))
        self.log(f'{mode}_acc', meter_in_use, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return

    def training_step(self, batch_data, batch_idx):

        data, graph = batch_data

        mu_x, var_x, mean_zq, var_zq = self.forward(data)

        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)
        
        targets = self.get_targets_from_data(data)
        nll_loss, kl = self.loss_fn(targets, mu_x, var_x, mean_zq, var_zq)
        beta = self.get_beta()
        loss = nll_loss + beta * self.kl_multiplier * kl

        self.log_dict({'train_nll_loss' : nll_loss,
                       'train_kl': kl,
                       'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_graph_performance(graph, mean_zq, mode='train')
        return loss

    def validation_step(self, batch_data, batch_idx, dataloader_idx=0):

        data, graph = batch_data

        mu_x, var_x, mean_zq, var_zq = self.forward(data)
        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)

        targets = self.get_targets_from_data(data)
        nll_loss, kl = self.loss_fn(targets, mu_x, var_x, mean_zq, var_zq)

        beta = self.get_beta()
        loss = nll_loss + beta * self.kl_multiplier * kl
        
        self.log_dict({'val_nll_loss' : nll_loss,
                       'val_kl': kl,
                       'val_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_graph_performance(graph, mean_zq, mode='val')
        return

    def predict_step(self, batch_data, batch_idx):

        data, graph = batch_data
        mu_x, var_x, mean_zq, var_zq = self.forward(data)
        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)
            
        return {'mu_x': mu_x, 'mean_zq': mean_zq, 'target_x': self.get_targets_from_data(data)}

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
