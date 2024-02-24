import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class PeriodicEmbedder(nn.Module):
    def __init__(
        self,
        feature_dim,
        emb_dim,
        num_variables
    ):
        super().__init__()
        self.pi = torch.acos(torch.zeros(1)).item()*2
        
        embBases = []
        for _ in range(num_variables):
            layer = nn.Linear(feature_dim, emb_dim//2)
            nn.init.normal_(layer.weight, 0, 10)
            embBases.append(layer)
        self.embBases = nn.ModuleList(embBases)
        
    def forward(self, x):
        
        assert x.ndim == 3
        
        out = []
        for idx, embBasis in enumerate(self.embBases):
            bases = embBasis(x[:,idx,:])
            cosEmb, sinEmb = torch.cos(bases), torch.sin(bases)
            x_emb = torch.concat([cosEmb,sinEmb],axis=-1)
            out.append(x_emb)
        return torch.stack(out,axis=1)
