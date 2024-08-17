import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from torchmetrics import Metric

class NumericalAccuracy(Metric):
    
    higher_is_better = True
    full_state_update = False
    def __init__(self, normalize=True):

        super().__init__()
        self.normalize = normalize
        self.add_state("corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
    def update(self, preds, target):

        assert preds.shape == target.shape
        assert preds.ndim <= 2
        
        if preds.ndim == 2:
            batch_size = preds.shape[0]
            if self.normalize:
                target = target/torch.max(torch.abs(target),dim=-1, keepdim=True)[0]
                preds = preds/torch.max(torch.abs(preds),dim=-1, keepdim=True)[0]

            pearson = PearsonCorrCoef(num_outputs=batch_size).to(preds.device)
            corr = pearson(preds.T, target.T)
        else:
            batch_size = 1
            if self.normalize:
                target = target/torch.max(torch.abs(target))
                preds = preds/torch.max(torch.abs(preds))
            pearson = PearsonCorrCoef().to(preds.device)
            corr = pearson(preds, target)
            
        self.corr += torch.sum(corr)
        self.total += batch_size

    def compute(self):
        return self.corr/self.total
