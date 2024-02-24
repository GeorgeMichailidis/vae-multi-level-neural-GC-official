
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from torchmetrics import Metric

def binary_edge_sampler(logits, gumbel_tau, hard):
    
    assert logits.ndim == 4
    assert logits.shape[-1] == 2

    batch_size, num_subjects, num_edges, num_class = logits.shape
    samples_onehot = F.gumbel_softmax(logits, tau=gumbel_tau, hard=hard)
    subj_graph = samples_onehot[:,:,:,1].view(batch_size, -1, num_edges)
    
    return subj_graph

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)

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
