
import os
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .encoder import Traj2GraphEncoder
from .loss import OneSubELBOLoss
from .modules.utils import binary_edge_sampler

class OneSubVAE(pl.LightningModule):

    def __init__(self, configs):
        super(OneSubVAE, self).__init__()

        self.save_hyperparameters()
        self.configs = configs
        self.loss_fn = OneSubELBOLoss(graph_type = self.configs['graph_type'], loss_type = self.configs['loss_type'])
        self.warmup_epochs = self.configs['warmup_epochs']

        self.encoder = Traj2GraphEncoder(self.configs)

        assert self.configs['decoder_style'] in ['GCEdge','GCNode']
        available_decoders = importlib.import_module('src.decoder')
        self.decoderClass = getattr(available_decoders, f'Graph2Traj{self.configs["decoder_style"]}Decoder')
        self.decoder = self.decoderClass(configs)
        
        self.MC_samples = self.configs.get('MC_samples',1)
        
    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        self._configs = configs

    def _sampler(self, mean_zq, var_zq):
        
        batch_size, num_subjects, num_edges = mean_zq.shape
        if self.configs['graph_type'] == 'numeric':
            zs_mc_shape = (self.MC_samples, batch_size, num_subjects, num_edges)
            zs_mc = mean_zq + (var_zq**0.5)*torch.randn(zs_mc_shape, device=mean_zq.device)
            subj_graph = torch.mean(zs_mc, dim=0, keepdim=False)
        else:
            logits_01 = torch.stack([-mean_zq, mean_zq], dim=-1)
            subj_graph = binary_edge_sampler(logits_01, self.configs['gumbel_tau'], self.configs['sample_hard'])

        return subj_graph

    def forward(self, data):
        if data.ndim == 4:
            data = data[:,:,:,:,None]

        mean_zq, var_zq = self.encoder(data)
        subj_graph = self._sampler(mean_zq, var_zq)
        mu_x, var_x = self.decoder(data, subj_graph)

        return mu_x, var_x, mean_zq, var_zq

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

    def training_step(self, data, batch_idx):

        if data.ndim == 4:
            data = data[:,:,:,:,None]

        targets = data[:, :, 1:, :]

        mu_x, var_x, mean_zq, var_zq = self.forward(data)
        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)

        nll_loss, kl = self.loss_fn(targets, mu_x, var_x, mean_zq, var_zq)

        beta = min(self.warmup_epochs, self.current_epoch+1)/self.warmup_epochs if self.warmup_epochs is not None else 1
        loss = nll_loss + beta * self.configs['kl_multiplier'] * kl

        self.log_dict({'train_nll_loss' : nll_loss,
                       'train_kl': kl,
                       'train_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, data, batch_idx):

        if data.ndim == 4:
            data = data[:,:,:,:,None]

        mu_x, var_x, mean_zq, var_zq = self.forward(data)
        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)

        targets = data[:,:,1:,:]
        nll_loss, kl = self.loss_fn(targets, mu_x, var_x, mean_zq, var_zq)

        beta = min(self.warmup_epochs, self.current_epoch+1)/self.warmup_epochs if self.warmup_epochs is not None else 1
        loss = nll_loss + beta * self.configs['kl_multiplier'] * kl
        
        self.log_dict({'val_nll_loss' : nll_loss,
                       'val_kl': kl,
                       'val_loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return
        
    def predict_step(self, data, batch_idx):
        if data.ndim == 4:
            data = data[:,:,:,:,None]

        mu_x, var_x, mean_zq, var_zq = self.forward(data)
        if self.configs['graph_type'] == 'binary':
            mean_zq = torch.sigmoid(mean_zq)
        return {'mu_x': mu_x, 'mean_zq': mean_zq, 'target_x': data[:,:,1:]}
