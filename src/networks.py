import torch
import torch.nn as nn

from .encoder import Traj2GraphEncoder, LadderEncoder
from .decoder import Graph2TrajGCEdgeDecoder, Graph2TrajGCNodeDecoder, LadderDecoder
from .sampler import Sampler

class OneLayer(nn.Module):
    """
    single-entity estimation via VAE
    """
    def __init__(self, configs):
        super(OneLayer, self).__init__()
        self.configs = configs
        self.encoder = Traj2GraphEncoder(self.configs)
        if self.configs['decoder_style'] == 'GCNode':
            self.decoder = Graph2TrajGCNodeDecoder(self.configs)
        else:
            self.decoder = Graph2TrajGCEdgeDecoder(self.configs)
        self.mc_samples = self.configs['MC_samples']
    
    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        assert configs['decoder_style'] in ['GCEdge','GCNode']
        assert configs['graph_type'] in ['numeric','binary']
        configs.setdefault('MC_samples',1)
        self._configs = configs
    
    def count_nn_params(self, model):
        return {'encoder': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                'decoder': sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}

    def forward(self, data):
        if data.ndim == 4:
            data = data[:,:,:,:,None]
        mean_zq, var_zq = self.encoder(data)
        if self.configs['graph_type'] == 'numeric':
            sampled_z = Sampler.sampler_normal(mean_zq, var_zq, self.mc_samples)
        else:
            logits_01 = torch.stack([-mean_zq, mean_zq], dim=-1)
            sampled_z = Sampler.sampler_binary_from_logits(logits_01,
                                                    gumbel_tau=self.configs['gumbel_tau'],
                                                    hard=self.configs['sample_hard'],
                                                    mc_samples=self.mc_samples)
        mu_x, var_x = self.decoder(data, sampled_z)
        return mu_x, var_x, mean_zq, var_zq


class TwoLayer(nn.Module):
    """
    multi-entity joint estimation with a layered structure
    """
    def __init__(self, configs):
        super(TwoLayer, self).__init__()
        self.configs = configs
        self.encoder = LadderEncoder(self.configs)
        self.decoder = LadderDecoder(self.configs)
    
    @property
    def configs(self):
        return self._configs

    @configs.setter
    def configs(self, configs):
        assert configs['decoder_style'] in ['GCEdge','GCNode']
        assert configs['graph_type'] in ['numeric','binary']
        configs.setdefault('MC_samples',1)
        self._configs = configs
    
    def count_nn_params(self, model):
        return {'encoder': sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            'decoder': sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}
    
    def forward(self, data):
        if data.ndim == 4:
            data = data[:,:,:,:,None]

        mean_zq, var_zq, param1_zbarq, param2_zbarq = self.encoder(data)
        mean_zp, var_zp, mu_x, var_x = self.decoder(data, param1_zbarq, param2_zbarq, mean_zq, var_zq)

        return mu_x, var_x, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp
