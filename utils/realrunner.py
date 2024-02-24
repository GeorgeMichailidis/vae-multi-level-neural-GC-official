
import importlib
import os
import warnings
import datetime
import time

import torch
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, Callback, LearningRateMonitor, ModelCheckpoint

class Printer(Callback):

    def __init__(self):
        pass
    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.time()
    def on_train_epoch_end(self, trainer, pl_module):

        t1 = time.time()
        elogs = trainer.logged_metrics
        lr = pl_module.lr_schedulers().optimizer.param_groups[0]['lr']

        print(f">> EPOCH={pl_module.current_epoch:03d}, time elapsed={t1-self.t0:.0f}s; train loss={elogs['train_loss_epoch']:.4f}, val loss={elogs['val_loss_epoch']:.4f}; lr={lr:.6f}")

        if 'train_nll_loss_epoch' in elogs.keys():
            print(f" * train_nll_loss:{elogs['train_nll_loss_epoch']:.4f}, val_nll_loss:{elogs['val_nll_loss_epoch']:.4f}")
        if 'train_kl_epoch' in elogs.keys():
            print(f" * train_kl:{elogs['train_kl_epoch']:.4f}, val_kl:{elogs['val_kl_epoch']:.4f}")
        if 'train_kl1_epoch' in elogs.keys():
            print(f" * train_kl1:{elogs['train_kl1_epoch']:.4f}, val_kl1:{elogs['val_kl1_epoch']:.4f}")
        if 'train_kl2_epoch' in elogs.keys(): 
            print(f" * train_kl2:{elogs['train_kl2_epoch']:.4f}, val_kl2:{elogs['val_kl2_epoch']:.4f}")

class realRunner():
    
    def __init__(self, args):

        self.args = args
        self.populate_model_configs()
        
        ## set up model class
        available_models = importlib.import_module('src')
        self.modelClass = getattr(available_models, self.model_configs['model_type'])
        
        ## set up dataset class
        available_datasets = importlib.import_module('src.datasets')
        if self.model_configs['model_type'] == 'MultiSubVAE':
            dsClassName = self.model_configs['ds_str'] + 'MultiSub'
        else:
            dsClassName = self.model_configs['ds_str'] + 'OneSub'
        self.dsClass = getattr(available_datasets, dsClassName)

    def populate_model_configs(self):
        
        model_configs = {'ds_str': self.args.ds_str, 'train_size': self.args.train_size}
        model_configs['ds_name'] = self.args.raw_data_specs['ds_name']

        for section_key in ['data_params', 'network_params', 'optimizer', 'train_params']:
            model_configs.update(getattr(self.args, section_key))
        assert model_configs['model_type'] in ['MultiSubVAE','OneSubVAE']

        model_configs['feature_dim'] = model_configs.get('feature_dim',1)
        model_configs['kl_multiplier'] = model_configs.get('kl_multiplier', 1.0)
        if model_configs['graph_type'] == 'numeric':
            model_configs['unit_variance'] = model_configs.get('unit_variance', True)
        else:
            model_configs['gumbel_tau'] = model_configs.get('gumbel_tau', 0.5)
        
        if model_configs['es_patience'] and model_configs['es_patience'] <= model_configs['warmup_epochs']:
            warnings.warn(f'es_patience = {model_configs["es_patience"]:.0f}, warmup_epochs = {model_configs["warmup_epochs"]:.0f}; es_patience should be set larger than warmup_epochs.')
        
        if model_configs['scheduler_type'] in ['StepLR','MultiStepLR']:
            model_configs['interval'] = model_configs.get('interval', 'epoch')

        setattr(self, 'model_configs', model_configs)
        print(f'================================')
        print(self.model_configs)
        print(f'================================')
        
        return

    def create_dataloader(self, file_type, mode=None):
        
        if mode is None:
            mode = file_type

        if self.model_configs['model_type'] == 'MultiSubVAE':
            dataset = self.dsClass(ds_name=self.model_configs['ds_name'],
                               size=self.model_configs[f'{file_type}_size'],
                               file_type=file_type)
        else:
            dataset = self.dsClass(ds_name=self.model_configs['ds_name'],
                               size=self.model_configs[f'{file_type}_size'],
                               file_type=file_type,
                               subject_id=self.model_configs['subject_id'])

        dataloader = DataLoader(dataset,
                              batch_size=self.model_configs['batch_size'],
                              shuffle = False if mode != 'train' else self.model_configs.get('shuffle',True),
                              num_workers=min(self.model_configs.get('num_workers',4),os.cpu_count()-1),
                              drop_last=True if mode != 'test' else False,
                              persistent_workers=True)
        return dataloader

    def create_trainer(self):

        lr_monitor = LearningRateMonitor(logging_interval='step')
        printer = Printer()

        ckpt_callback = ModelCheckpoint(dirpath=self.args.ckpt_dir, filename=f'ckpt-best', monitor='val_loss', save_last=True)
        ckpt_callback.CHECKPOINT_NAME_LAST = f'ckpt-last'

        callbacks = [lr_monitor, ckpt_callback, printer]
        if self.model_configs['es_patience']:
            early_stopper = EarlyStopping(monitor="val_loss",
                                          min_delta=1e-4,
                                          patience=self.model_configs['es_patience'],
                                          verbose=True,
                                          mode="min")
            callbacks.append(early_stopper)

        logger = TensorBoardLogger(save_dir=f'{self.args.output_dir}', name='lightning_logs')
        trainer = pl.Trainer(accelerator = self.args.accelerator,
                             devices = self.args.devices,
                             max_epochs = self.model_configs['max_epochs'],
                             callbacks = callbacks,
                             enable_progress_bar = False,
                             logger = logger,
                             gradient_clip_val = self.model_configs['gradient_clip_val'],
                             log_every_n_steps = 1,
                             num_sanity_val_steps = 0,
                             limit_val_batches = self.model_configs['limit_val_batches'],
                             precision = 32,
                             detect_anomaly = False, fast_dev_run = False)

        self.best_model_path = os.path.join(self.args.ckpt_dir, 'ckpt-best.ckpt')
        self.last_model_path = os.path.join(self.args.ckpt_dir, 'ckpt-last.ckpt')

        return trainer

    def train_network(self, train_dataloader, val_dataloader=None):

        model = self.modelClass(self.model_configs)
        trainer = self.create_trainer()

        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] model training starts')
        trainer.fit(model, train_dataloader, val_dataloader)
        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] model training finishes at epoch = {model.current_epoch}')

        self.tb_dir = trainer.logger.log_dir
        print(f'>> ckpt_path (best) = {self.best_model_path}')
        print(f'>> ckpt_path (last) = {self.last_model_path}')
        print(f'>> tb_dir = {self.tb_dir}')

        return trainer

    def model_eval(self, pl_trainer, dataloader, ckpt_type='last'):
        
        print(f'[{datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")}] performing model evaluation')

        if ckpt_type == 'last':
            model = self.modelClass.load_from_checkpoint(self.last_model_path)
            print(f'>> model loaded from {self.last_model_path}, model.device={model.device}')
        else:
            model = self.modelClass.load_from_checkpoint(self.best_model_path)
            print(f'>> model loaded from {self.best_model_path}, model.device={model.device}')

        out = pl_trainer.predict(model, dataloader)

        if self.model_configs['model_type'] == 'MultiSubVAE':

            target_xs = torch.concat([batch_out['target_x'] for batch_out in out],axis=0)
            pred_xs = torch.concat([batch_out['mu_x'] for batch_out in out],axis=0)
            subj_graphs = torch.concat([batch_out['mean_zq'] for batch_out in out],axis=0)
            bar_graphs = torch.concat([batch_out['mean_zbarq'] for batch_out in out],axis=0)

            returnDict = {'xTargets': target_xs.numpy(), 'xPredsMean': pred_xs.numpy(), 'subjGraphs': subj_graphs.numpy(), 'barGraphs': bar_graphs.numpy()}

        elif self.model_configs['model_type'] == 'OneSubVAE':

            target_xs = torch.concat([batch_out['target_x'] for batch_out in out],axis=0)
            pred_xs = torch.concat([batch_out['mu_x'] for batch_out in out],axis=0)
            subj_graphs = torch.concat([batch_out['mean_zq'] for batch_out in out],axis=0)

            returnDict = {'xTargets': target_xs.numpy(), 'xPredsMean': pred_xs.numpy(), 'subjGraphs': subj_graphs.numpy()}

        else:
            raise ValueError('not supported')

        return returnDict

    def end2end(self, eval_type='test'):

        train_dataloader = self.create_dataloader('train')
        val_dataloader = self.create_dataloader('val')

        pl_trainer = self.train_network(train_dataloader, val_dataloader)

        test_dataloader = self.create_dataloader(eval_type, mode='test')
        outDict = self.model_eval(pl_trainer, test_dataloader, ckpt_type='last')

        return outDict
