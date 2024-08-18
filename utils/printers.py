import time

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.model_summary import summarize

class RealPrinter(Callback):

    def __init__(self):
        pass
    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.monotonic()
    def on_train_epoch_end(self, trainer, pl_module):

        t1 = time.monotonic()
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


class SimPrinter(Callback):

    def __init__(self):
        pass
    def on_train_epoch_start(self, trainer, pl_module):
        self.t0 = time.monotonic()
    def on_train_epoch_end(self, trainer, pl_module):

        t1 = time.monotonic()
        elogs = trainer.logged_metrics
        lr = pl_module.lr_schedulers().optimizer.param_groups[0]['lr']

        acc_subj_str = f"train_acc:{elogs['train_acc_epoch']:.2f}, val_acc:{elogs['val_acc_epoch']:.2f};" if 'train_acc_epoch' in elogs.keys() else ''
        acc_bar_str = f"train_acc_bar:{elogs['train_acc_bar_epoch']:.2f}, val_acc_bar:{elogs['val_acc_bar_epoch']:.2f};" if 'train_acc_bar_epoch' in elogs.keys() else ''
        
        print(f">> EPOCH={pl_module.current_epoch:03d}, time elapsed={t1-self.t0:.0f}s; train loss={elogs['train_loss_epoch']:.4f}, val loss={elogs['val_loss_epoch']:.4f}; lr={lr:.6f}; {acc_subj_str} {acc_bar_str}")
        if 'train_nll_loss_epoch' in elogs.keys():
            print(f" * train_nll_loss:{elogs['train_nll_loss_epoch']:.4f}, val_nll_loss:{elogs['val_nll_loss_epoch']:.4f}")
        if 'train_kl_epoch' in elogs.keys():
            print(f" * train_kl:{elogs['train_kl_epoch']:.4f}, val_kl:{elogs['val_kl_epoch']:.4f}")
        if 'train_kl1_epoch' in elogs.keys():
            print(f" * train_kl1:{elogs['train_kl1_epoch']:.4f}, val_kl1:{elogs['val_kl1_epoch']:.4f}")
        if 'train_kl2_epoch' in elogs.keys():
            print(f" * train_kl2:{elogs['train_kl2_epoch']:.4f}, val_kl2:{elogs['val_kl2_epoch']:.4f}")


class ModelSummaryPrinter(Callback):
    
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
    
    def on_fit_start(self, trainer, pl_module):
        print('=========================')
        print(summarize(pl_module, max_depth=self.max_depth))
        print('=========================')
    
