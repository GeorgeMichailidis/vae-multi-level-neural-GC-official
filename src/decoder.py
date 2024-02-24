"""
Decoder modules: Common2Entity + Node-centric (Graph2TrajGCNodeDecoder)/Edge-centric (Graph2TrajGCEdgeDecoder) Graph2Traj decoders, and the SimDecoder object that combines the two modules which also includes the sampling step
The code in Graph2TrajGCEdgeDecoder() referenced and is modified based on https://github.com/ethanfetaya/NRI/blob/master/modules.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import importlib

from ._basegraph import _baseGraph
from .modules import MLP, PeriodicEmbedder

_EPS=1e-4

class _baseGraph2TrajGCDecoder(_baseGraph, nn.Module):
    
    def __init__(self, params):
        
        super(_baseGraph2TrajGCDecoder, self).__init__(params['num_nodes'], params['include_diag'])
        self.num_nodes = params['num_nodes']
        self.feature_dim = params['feature_dim']
        self.graph_type = params['graph_type']

        self.hidden_dim = params['decoder_hidden_dim']
        self.tf_interval = params['decoder_tf_interval']
        self.fix_var = params['decoder_fix_trajvar']

        self.do_prob = params.get('do_prob',0.1)
        self.min_var = params.get('min_var',1e-8)
        self.max_var = params.get('max_var',100)
        
        self.residual_flag = params.get('decoder_residual_flag',False)
        self.delta_flag = params.get('decoder_delta_flag',False)
        
    def single_step_forward(self, prev_step, graph):
        raise NotImplementedError
    
    def forward(self, data, graph):
    
        batch_size, num_subjects, num_timesteps, num_nodes, feature_dim = data.shape
        x = data.view(-1, num_timesteps, num_nodes, feature_dim)
        graph = graph.view(-1, graph.size(-1))

        prev_step = x[:,0::self.tf_interval, :, :]
        chunk_size = prev_step.size(1)

        mu_pred_raw, var_pred_raw = [], []
        for step in range(self.tf_interval):
            mu, var = self.single_step_forward(prev_step, graph)
            mu_pred_raw.append(mu)
            var_pred_raw.append(var)
            prev_step = mu

        mu_pred = torch.zeros((batch_size*num_subjects, chunk_size*self.tf_interval, num_nodes, feature_dim), device=x.device)
        var_pred = torch.zeros((batch_size*num_subjects, chunk_size*self.tf_interval, num_nodes, feature_dim), device=x.device)

        for i in range(self.tf_interval):
            mu_pred[:, i::self.tf_interval, :, :] = mu_pred_raw[i]
            var_pred[:, i::self.tf_interval, :, :] = var_pred_raw[i]
        mu_pred, var_pred = mu_pred[:, :(x.size(1)-1), :, :], var_pred[:, :(x.size(1)-1), :, :]

        mu_pred = mu_pred.view(batch_size, -1, mu_pred.size(1), num_nodes, feature_dim)
        var_pred = var_pred.view(batch_size, -1, var_pred.size(1), num_nodes, feature_dim)

        return mu_pred, var_pred
    
class Graph2TrajGCEdgeDecoder(_baseGraph2TrajGCDecoder):
    
    def __init__(self, params):

        super(Graph2TrajGCEdgeDecoder, self).__init__(params)
        self.msg_fc = MLP(input_dim = 2*self.feature_dim,
                          output_dim = self.hidden_dim,
                          hidden_dims = [self.hidden_dim],
                          dropout_rate = self.do_prob,
                          activation = 'relu',
                          batch_norm = False)
        
        fc_input_dim = self.hidden_dim if not self.residual_flag else self.hidden_dim + self.feature_dim
        self.mean_fc = MLP(input_dim = fc_input_dim,
                           output_dim = self.feature_dim,
                           hidden_dims = [self.hidden_dim, self.hidden_dim],
                           dropout_rate = self.do_prob,
                           activation = 'relu',
                           batch_norm = False)

        if self.fix_var is None:
            self.variance_fc = MLP(input_dim = fc_input_dim,
                          output_dim = self.feature_dim,
                          hidden_dims = [self.hidden_dim, self.hidden_dim],
                          activation = 'relu',
                          dropout_rate = self.do_prob,
                          batch_norm = False,
                          use_softplus = True,
                          clip = True,
                          min_val = self.min_var,
                          max_val = self.max_var)

    def single_step_forward(self, prev_step, graph):
    
        edge_hidden = self.node2edge(prev_step)
        batch_size, num_edges = graph.shape
        graph = graph.unsqueeze(1).expand([batch_size, prev_step.size(1), num_edges])
        graph = graph.unsqueeze(-1)

        msg_hidden = self.msg_fc(edge_hidden)
        msg_hidden *= graph

        msg_in_node = self.edge2node(msg_hidden, normalize=False)
        if self.residual_flag:
            msg_in_node = torch.cat([prev_step, msg_in_node],dim=-1)
        
        mu = self.mean_fc(msg_in_node)
        if self.delta_flag:
            mu = mu + prev_step
            
        if self.fix_var is None:
            var = self.variance_fc(msg_in_node)
        else:
            var = self.fix_var * torch.ones_like(mu,device=mu.device)

        return mu, var

class Graph2TrajGCNodeDecoder(_baseGraph2TrajGCDecoder):
    
    def __init__(self, params):

        super(Graph2TrajGCNodeDecoder, self).__init__(params)
        
        self.hidden_dim = params['decoder_hidden_dim']
        self.do_prob = params['do_prob']
        
        self.emb_dim = params.get('decoder_emb_dim',None)
        if self.emb_dim:
            self.embedder = PeriodicEmbedder(self.feature_dim, self.emb_dim, self.num_nodes)
        
        self.feature_dim_eff = self.emb_dim or self.feature_dim
        fc_input_dim = self.num_nodes * self.feature_dim_eff if not self.residual_flag else (self.num_nodes + 1) * self.feature_dim_eff
        self.mean_fc = MLP(input_dim = fc_input_dim,
                   output_dim = self.feature_dim,
                   hidden_dims = [2*self.hidden_dim, self.hidden_dim, self.hidden_dim//2],
                   dropout_rate = self.do_prob,
                   activation = 'relu',
                   batch_norm = False)
        
        if self.fix_var is None:
            self.variance_fc = MLP(input_dim = fc_input_dim,
                          output_dim = self.feature_dim,
                          hidden_dims = [self.hidden_dim, self.hidden_dim//2],
                          activation = 'relu',
                          dropout_rate = self.do_prob,
                          batch_norm = False,
                          use_softplus = True,
                          clip = True,
                          min_val = self.min_var,
                          max_val = self.max_var)

    def single_step_forward(self, prev_step, gc_graph):
        
        batch_size, chunk_size, num_nodes, feature_dim = prev_step.shape
        assert self.feature_dim == feature_dim
        
        _, num_edges = gc_graph.shape
        
        gc_graph = gc_graph.unsqueeze(1).expand([batch_size, chunk_size, num_edges]).contiguous()
        gc_graph_reshape = torch.zeros(batch_size, chunk_size, num_nodes, num_nodes, device=gc_graph.device)
        gc_graph_reshape[:,:,self.receivers, self.senders] = gc_graph
        
        prev_step_reshaped = prev_step.reshape([-1,num_nodes,feature_dim])
        prev_step_optional_emb = self.embedder(prev_step_reshaped) if self.emb_dim else prev_step_reshaped
        prev_step_expanded = prev_step_optional_emb.unsqueeze(1).expand([batch_size*chunk_size, num_nodes, num_nodes, self.feature_dim_eff]).contiguous()
        prev_step_expanded = prev_step_expanded.reshape([batch_size, chunk_size, num_nodes, num_nodes, -1])
        
        prev_step_msg = prev_step_expanded * gc_graph_reshape.unsqueeze(-1)
        prev_step_msg = prev_step_msg.flatten(3)
        
        if self.residual_flag:
            prev_step_msg = torch.cat([prev_step_optional_emb.reshape([batch_size, chunk_size, num_nodes, -1]), prev_step_msg],dim=-1)
            
        mu = self.mean_fc(prev_step_msg)
        if self.delta_flag:
            mu = mu + prev_step
        
        if self.fix_var is None:
            var = self.variance_fc(prev_step_msg)
        else:
            var = self.fix_var * torch.ones_like(mu,device=mu.device)
        return mu, var

class Common2EntityDecoder(nn.Module):
    
    def __init__(self, params):

        super(Common2EntityDecoder, self).__init__()
        self.graph_type = params['graph_type']
        self.num_nodes = params['num_nodes']
        self.num_subjects = params['num_subjects']
        self.unit_variance = params['unit_variance']

        self.do_prob = params.get('do_prob', 0.1)
        self.min_var = params.get('min_var',1e-8)
        self.max_var = params.get('max_var',100)
        self.num_edges = int(self.num_nodes*(self.num_nodes-1))

        if self.graph_type == 'numeric' and not self.unit_variance:
            if not isinstance(params['decoder_hidden_dims_gvarnet'], list):
                assert isinstance(params['decoder_hidden_dims_gvarnet'], int)
                params['decoder_hidden_dims_gvarnet'] = [params['decoder_hidden_dims_gvarnet']]
            self.var_fc = MLP(input_dim = self.num_edges,
                            output_dim = self.num_edges,
                            hidden_dims = params['decoder_hidden_dims_gvarnet'],
                            activation = 'elu',
                            dropout_rate = self.do_prob,
                            use_softplus = True,
                            clip = True,
                            min_val = self.min_var,
                            max_val = self.max_var)

    def forward(self, zbar):
        
        mu = zbar.unsqueeze(1).repeat(1, self.num_subjects, 1)
        if self.graph_type == 'numeric':
            if self.unit_variance:
                var = torch.ones_like(mu, device=mu.device)
            else:
                var = self.var_fc(zbar)
                var = var.unsqueeze(1).repeat(1, self.num_subjects, 1)
            return mu, var
        else:
            return mu, None

class SimDecoder(nn.Module):
    
    def __init__(self, params):

        super(SimDecoder, self).__init__()

        self.graph_type = params['graph_type']
        self.num_subjects = params['num_subjects']
        self.encoder_weight = params['encoder_weight']
        self.decoder_style = params['decoder_style']

        self.gumbel_tau = params.get('gumbel_tau',None)
        self.sample_hard = params.get('sample_hard',False)
        self.MC_samples = params.get('MC_samples',1)

        if self.decoder_style == 'GCEdge':
            self.graph2traj_net = Graph2TrajGCEdgeDecoder(params)
        elif self.decoder_style == 'GCNode':
            self.graph2traj_net = Graph2TrajGCNodeDecoder(params)
        else:
            raise ValueError(f'unrecognized decoder_style={self.decoder_style}')

        self.common2graph_net = Common2EntityDecoder(params)

    def _bernoulli_sampler(self, p, tau=1., hard=False):
    
        p = torch.clamp(p, min=_EPS, max=1-_EPS)
        logits_class1, logits_class0 = torch.log(p/(1.-p)), torch.log((1.-p)/p)
        logits = torch.stack([logits_class1, logits_class0],axis=-1)
        samples_onehot = F.gumbel_softmax(logits, tau=tau, hard=hard)
        return samples_onehot[:,:,0]

    def _merge_encoder_info(self, mean_q, var_q, mean_p, var_p):
        
        if self.graph_type == 'numeric':
            var_q_inv, var_p_inv = 1./var_q, 1./var_p
            var_adj = 1./(var_q_inv * self.encoder_weight + var_p_inv * (1 - self.encoder_weight))
            mean_adj = (mean_q * var_q_inv * self.encoder_weight + mean_p * var_p_inv * (1 - self.encoder_weight)) * var_adj
        else:
            mean_q, mean_p = torch.clamp(mean_q, min=_EPS, max=1-_EPS), torch.clamp(mean_p, min=_EPS, max=1-_EPS)
            mean_adj = 1./(self.encoder_weight/mean_q + (1 - self.encoder_weight)/mean_p)
            var_adj = var_p

        return mean_adj, var_adj

    def forward(self, data, param1_zbar, param2_zbar, mean_zq, var_zq):
        
        batch_size, num_edges = param1_zbar.shape

        zbar_mc_shape = (self.MC_samples, batch_size, num_edges)
        zs_mc_shape = (self.MC_samples, batch_size, self.num_subjects, num_edges)

        if self.graph_type == 'numeric':

            zbar_mc = param1_zbar + (param2_zbar**0.5)*torch.randn(zbar_mc_shape,device=param1_zbar.device)
            zbar = torch.mean(zbar_mc, dim=0, keepdim=False)

            mean_z, var_z = self.common2graph_net(zbar)
            mean_z_adj, var_z_adj = self._merge_encoder_info(mean_zq, var_zq, mean_z, var_z)

            zs_mc = mean_z_adj + (var_z_adj**0.5)*torch.randn(zs_mc_shape, device=mean_z.device)
            z_by_subject = torch.mean(zs_mc, dim=0, keepdim=False)

        else:

            param1_zbar_mc = param1_zbar.repeat(self.MC_samples,1,1).view(self.MC_samples * batch_size, -1)
            param2_zbar_mc = param2_zbar.repeat(self.MC_samples,1,1).view(self.MC_samples * batch_size, -1)
            zbar_mc = torch.distributions.beta.Beta(param1_zbar_mc, param2_zbar_mc).rsample()
            zbar = torch.mean(zbar_mc.view(zbar_mc_shape), dim=0, keepdim=False)

            mean_z, var_z = self.common2graph_net(zbar)
            mean_z_adj, var_z_adj = self._merge_encoder_info(mean_zq, var_zq, mean_z, var_z)

            z_samples = self._bernoulli_sampler(mean_z_adj.repeat(self.MC_samples,1,1,1).view(-1, num_edges), tau=self.gumbel_tau, hard=self.sample_hard)
            zs_mc = z_samples.view(zs_mc_shape)
            z_by_subject = torch.mean(zs_mc, dim=0, keepdim=False)

        mu_x, var_x = self.graph2traj_net(data, z_by_subject)

        return mean_z_adj, var_z_adj, mu_x, var_x
