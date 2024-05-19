import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1.0e-4

class Sampler():
    
    def __init__(self):
        pass
        
    @classmethod
    def sampler_normal(cls, mean_zq, var_zq, mc_samples=1):
        """
        Argv:
        - mean_zq, var_zq: [batch_size, num_edges] that corresponds to the mean and variance of the corresponding edge
            * in the multi-subject case, input size is [batch_size, num_subjects, num_edges]
        - mc_samples: number of monte carlo samples generated, which later on will be averaged
        Return:
        - sampled_graph: same size as mean_zq, var_zq
        """
        if mean_zq.ndim == 2:
            batch_size, num_edges = mean_zq.shape
            num_subjects = 1
        else:
            batch_size, num_subjects, num_edges = mean_zq.shape
            mean_zq = mean_zq.view(-1, num_edges)
            var_zq = var_zq.view(-1, num_edges)
            
        ## reparametrization + multiple monte-carlo samples; here we leverage broadcasting
        zs_mc = mean_zq + (var_zq**0.5)*torch.randn((mc_samples, mean_zq.shape[0], num_edges), device=mean_zq.device)
        sampled_graph = torch.mean(zs_mc, dim=0, keepdim=False)
        if num_subjects > 1:
            sampled_graph = sampled_graph.reshape(batch_size, num_subjects, num_edges).contiguous()
            
        return sampled_graph

    @classmethod
    def sampler_binary_from_logits(cls, logits, gumbel_tau=0.5, hard=False, mc_samples=1):
        """
        Argv:
        - logits: num_classes = 2 (0/1)
            * single-subject: [batch_size, num_edges, num_classes]
            * multi-subject: [batch_size, num_subjects, num_edges, num_classes]
        - gumbel_tau: float
        - hard: boolean
        Return:
        - sampled_graph: sampled edge probabilities for class 1; [batch_size, num_edges]
        """
        if logits.ndim==3:
            assert logits.shape[-1]==2, f'logits.shape={logits.shape}: expecting (batch_size,num_edges,2)'
            batch_size, num_edges, _ = logits.shape
            num_subjects = 1
        else:
            assert (logits.ndim == 4) and (logits.shape[-1]==2), f'logits.shape={logits.shape}: expecting (batch_size,num_subjects,num_edges,2)'
            batch_size, num_subjects, num_edges, _ = logits.shape
            logits = logits.view(batch_size*num_subjects, -1, 2)
        
        logits_mc = logits.unsqueeze(0).repeat(mc_samples, 1, 1, 1).reshape(-1, num_edges, 2)
        ## samples_onehot.shape = [mc_samples*batch_size*num_subjects, num_edges, 2]
        samples_onehot = F.gumbel_softmax(logits_mc, tau=gumbel_tau, hard=hard)
        ## take only the one class, then average across MC sampels
        sampled_graph = torch.mean(samples_onehot[:,:,1].view(mc_samples, -1, num_edges), dim=0, keepdim=False)
        if num_subjects > 1:
            sampled_graph = sampled_graph.reshape(batch_size, num_subjects, num_edges).contiguous()
        
        return sampled_graph
    
    @classmethod
    def sampler_binary_from_probs(cls, probs, gumbel_tau=0.5, hard=False, mc_samples=1):
        """
        Argv:
        - probs: class-1 probability
        """
        p = torch.clamp(probs, min=_EPS, max=1-_EPS)
        logits_class1, logits_class0 = torch.log(p/(1.-p)), torch.log((1.-p)/p)
        logits_01 = torch.stack([logits_class0, logits_class1],axis=-1)
        return cls.sampler_binary_from_logits(logits_01, gumbel_tau=gumbel_tau, hard=hard, mc_samples=mc_samples)
    
    @classmethod
    def sampler_beta(cls, param1, param2, mc_samples=1):
        """
        Argv:
        - param1, param2: two parameters for Beta distribution; [batch_size, num_edges]
        """
        assert (param1.ndim == 2) and (param2.ndim==2)
        batch_size, num_edges = param1.shape
    
        param1_mc = param1.unsqueeze(0).repeat(mc_samples,1,1).view(mc_samples*batch_size, -1)
        param2_mc = param2.unsqueeze(0).repeat(mc_samples,1,1).view(mc_samples*batch_size, -1)
        
        sampled_graph_mc = torch.distributions.beta.Beta(param1_mc, param2_mc).rsample()
        sampled_graph = torch.mean(sampled_graph_mc.view(mc_samples, batch_size, num_edges), dim=0, keepdim=False)
        return sampled_graph
