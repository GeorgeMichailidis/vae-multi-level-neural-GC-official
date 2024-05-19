from abc import ABC, abstractmethod
import os

import numpy as np
import pytorch_lightning as pl
import torch

from .networks import OneLayer, TwoLayer
from .loss import OneSubELBOLoss, MultiSubELBOLoss

_EPS = 1.0e-4

class _baseModel(ABC, pl.LightningModule):
    """
    abstract base class for pl-based pipeline
    """
    def __init__(self, configs):
        super(_baseModel, self).__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.network = None
        self.loss_fn = None
        
    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        self._configs = configs
    
    @abstractmethod
    def training_step(self, batch_data, batch_idx):
        pass
    
    @abstractmethod
    def validation_step(self, batch_data, batch_idx):
        pass
    
    @abstractmethod
    def predict_step(self, batch_data, batch_idx):
        pass
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.configs['learning_rate'],
                                     weight_decay=self.configs.get('weight_decay',1e-6))

        if self.configs['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                            mode='min',
                            factor=self.configs['factor'],
                            patience=self.configs['patience'],
                            verbose=True)
            lr_scheduler = {'scheduler':scheduler,'monitor':'val_loss'}
        elif self.configs['scheduler_type'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                            step_size=self.configs['step_size'],
                            gamma=self.configs['gamma'],
                            verbose=False)
            lr_scheduler = {'scheduler':scheduler,'interval':self.configs['interval']}
        elif self.configs['scheduler_type'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                            milestones=self.configs['milestones'],
                            gamma=self.configs['gamma'],
                            verbose=False)
            lr_scheduler = {'scheduler':scheduler,'interval':self.configs['interval']}
        else:
            raise ValueError('unrecognized scheduler_type')

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class MultiSubVAE(_baseModel):

    def __init__(self,configs):
        super(MultiSubVAE, self).__init__(configs)
        self.network = TwoLayer(self.configs)
        self.loss_fn = MultiSubELBOLoss(graph_type = self.configs['graph_type'],
                                        loss_type = self.configs['loss_type'])
        self.warmup_epochs = self.configs['warmup_epochs']
        self.kl_multiplier = self.configs['kl_multiplier']
        
    def forward(self, data):
        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.network(data)
        return mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp

    def get_targets_from_data(self, data):
        targets = data[:, :, 1:, :]
        if targets.ndim == 4:
            targets = targets.unsqueeze(-1)
        return targets
    
    def get_beta(self):
        beta = min(self.warmup_epochs, self.current_epoch+1)/self.warmup_epochs if self.warmup_epochs is not None else 1
        return beta
        
    def training_step(self, data, batch_idx):

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
        return loss

    def validation_step(self, data, batch_idx):

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
                       
        return
            
    def predict_step(self, data, batch_idx):
        
        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        mean_zbarq = param1_zbarq if self.configs['graph_type'] == 'numeric' else param1_zbarq/(param1_zbarq + param2_zbarq + _EPS)
        
        return {'mu_x': mu_x, 'mean_zq': mean_zq, 'mean_zbarq': mean_zbarq, 'target_x': self.get_targets_from_data(data)}


class OneSubVAE(_baseModel):

    def __init__(self,configs):
        super(OneSubVAE, self).__init__(configs)
        self.network = OneLayer(self.configs)
        self.loss_fn = OneSubELBOLoss(graph_type = self.configs['graph_type'],
                                      loss_type = self.configs['loss_type'])
        self.warmup_epochs = self.configs['warmup_epochs']
        self.kl_multiplier = self.configs['kl_multiplier']
    
    def forward(self, data):
        mu_x, var_x, mean_zq, var_zq = self.network(data)
        return mu_x, var_x, mean_zq, var_zq
    
    def get_targets_from_data(self, data):
        targets = data[:, :, 1:, :]
        if targets.ndim == 4:
            targets = targets.unsqueeze(-1)
        return targets
    
    def get_beta(self):
        beta = min(self.warmup_epochs, self.current_epoch+1)/self.warmup_epochs if self.warmup_epochs is not None else 1
        return beta
    
    def training_step(self, data, batch_idx):

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

        return loss

    def validation_step(self, data, batch_idx):

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
        return
        
    def predict_step(self, data, batch_idx):
        
        mu_x, var_x, mean_zq, var_zq = self.forward(data)
        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)
        
        return {'mu_x': mu_x, 'mean_zq': mean_zq, 'target_x': self.get_targets_from_data(data)}
