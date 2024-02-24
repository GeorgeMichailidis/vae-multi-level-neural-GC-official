
import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy

from .encoder import Encoder
from .decoder import SimDecoder
from .loss import MultiSubELBOLoss
from ._simdl import simDL
from .modules.utils import NumericalAccuracy

_EPS=1e-4

class simMultiSubVAE(simDL, pl.LightningModule):
    def __init__(self, configs):
    
        super(simMultiSubVAE, self).__init__(configs)
        self.save_hyperparameters()
        self.configs = configs

        self.encoder = Encoder(self.configs)
        self.decoder = SimDecoder(self.configs)

        self.loss_fn = MultiSubELBOLoss(graph_type = self.configs['graph_type'], loss_type = self.configs['loss_type'])
        self.warmup_epochs = self.configs['warmup_epochs']
        self.kl_multiplier = self.configs['kl_multiplier']
        
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
        

    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        self._configs = configs

    def forward(self, data):
        
        if data.ndim == 4:
            data = data[:,:,:,:,None]

        mean_zq, var_zq, param1_zbarq, param2_zbarq = self.encoder(data)
        mean_zp, var_zp, mu_x, var_x = self.decoder(data, param1_zbarq, param2_zbarq, mean_zq, var_zq)

        return mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp

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

    def training_step(self, batch_data, batch_idx):

        data, graph_by_subject, graph_bar = batch_data
        
        if data.ndim == 4:
            data = data[:,:,:,:,None]
        targets = data[:, :, 1:, :, :]

        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        nll_loss, kl1, kl2 = self.loss_fn(targets, mu_x, var_x,
                                          mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp)

        beta = min(self.warmup_epochs, self.current_epoch+1)/self.warmup_epochs if self.warmup_epochs is not None else 1
        loss = nll_loss + beta * self.kl_multiplier * (kl1 + kl2)
        
        self.log_dict({'train_nll_loss' : nll_loss,
                       'train_kl1': kl1,
                       'train_kl2': kl2,
                       'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.configs['graph_type'] == 'binary':
            graph_by_subject = 1 * (graph_by_subject!=0)
        
        self.train_meter.update(mean_zq.view(-1,mean_zq.shape[-1]), graph_by_subject.view(-1,graph_by_subject.shape[-1]))
        self.log('train_acc', self.train_meter, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        bar_mean = param1_zbarq if self.configs['graph_type'] == 'numeric' else param1_zbarq/(param1_zbarq + param2_zbarq + _EPS)
        self.train_meter_bar.update(bar_mean.view(-1,bar_mean.shape[-1]), graph_bar.view(-1,graph_bar.shape[-1]))
        self.log('train_acc_bar', self.train_meter_bar, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch_data, batch_idx):

        data, graph_by_subject, graph_bar = batch_data
        
        if data.ndim == 4:
            data = data[:,:,:,:,None]
        targets = data[:, :, 1:, :, :]

        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        nll_loss, kl1, kl2 = self.loss_fn(targets, mu_x, var_x,
                                          mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp)

        beta = min(self.warmup_epochs, self.current_epoch+1)/self.warmup_epochs if self.warmup_epochs is not None else 1
        loss = nll_loss + beta * self.kl_multiplier * (kl1 + kl2)
        self.log_dict({'val_nll_loss' : nll_loss,
                       'val_kl1': kl1,
                       'val_kl2': kl2,
                       'val_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                       
        if self.configs['graph_type'] == 'binary':
            graph_by_subject = 1 * (graph_by_subject!=0)
        
        self.val_meter.update(mean_zq.view(-1,mean_zq.shape[-1]), graph_by_subject.view(-1,graph_by_subject.shape[-1]))
        self.log('val_acc', self.val_meter, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        bar_mean = param1_zbarq if self.configs['graph_type'] == 'numeric' else param1_zbarq/(param1_zbarq + param2_zbarq + _EPS)
        self.val_meter_bar.update(bar_mean.view(-1,bar_mean.shape[-1]), graph_bar.view(-1,graph_bar.shape[-1]))
        self.log('val_acc_bar', self.val_meter_bar, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
    def predict_step(self, batch_data, batch_idx):
        
        data, graph_by_subject, graph_bar = batch_data
        if data.ndim == 4:
            data = data[:,:,:,:,None]
        
        mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp = self.forward(data)
        mean_zbarq = param1_zbarq if self.configs['graph_type'] == 'numeric' else param1_zbarq/(param1_zbarq + param2_zbarq + _EPS)
        
        return {'target_x': data[:,:,1:,:],'mu_x': mu_x, 'mean_zq': mean_zq, 'mean_zbarq': mean_zbarq}
