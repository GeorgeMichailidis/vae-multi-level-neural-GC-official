"""
Encoder modules: Traj2Graph GNN-style (Traj2GraphEncoder) encoder + Entity2Common
Traj2GraphEncoder() is largely taken from https://github.com/ethanfetaya/NRI/blob/master/modules.py
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .basegraph import BaseGraph
from .modules import MLP

_EPS=1e-4

class Traj2GraphEncoder(BaseGraph):
    
    def __init__(self, params):

        super(Traj2GraphEncoder, self).__init__(params['num_nodes'], params['include_diag'])

        self.graph_type = params['graph_type']
        self.num_nodes = params['num_nodes']
        self.feature_dim = params['feature_dim']
        self.context_length = params['context_length']
        self.hidden_dim = params['encoder_hidden_dim']
        self.do_prob = params.get('do_prob',0.1)
        self.unit_variance = params.get('unit_variance', False)
        self.min_var = params.get('min_var',1e-8)
        self.max_var = params.get('max_var',100)

        self.msg_fc1 = MLP(input_dim = self.context_length * self.feature_dim,
                        output_dim = self.hidden_dim,
                        hidden_dims = [self.hidden_dim],
                        dropout_rate = self.do_prob,
                        activation = 'elu')

        self.msg_fc2 = MLP(input_dim = self.hidden_dim * 2,
                        output_dim = self.hidden_dim,
                        hidden_dims = [self.hidden_dim],
                        dropout_rate = self.do_prob,
                        activation = 'elu')

        self.msg_fc3 = MLP(input_dim = self.hidden_dim,
                         output_dim = self.hidden_dim,
                         hidden_dims = [self.hidden_dim],
                         dropout_rate = self.do_prob,
                         activation = 'elu')

        self.hidden_fc = MLP(input_dim = self.hidden_dim * 3,
                             output_dim = self.hidden_dim,
                             hidden_dims = [self.hidden_dim],
                             dropout_rate = self.do_prob,
                             activation = 'elu')

        self.mean_fc = nn.Linear(self.hidden_dim, 1)

        if self.graph_type == 'numeric' and not self.unit_variance:
            self.variance_fc = MLP(input_dim = self.hidden_dim,
                                   output_dim = 1,
                                   hidden_dims = None,
                                   use_softplus = True,
                                   clip = True,
                                   min_val = self.min_var,
                                   max_val = self.max_var)

    def forward(self, data):
        
        batch_size, num_subjects, context_length, num_nodes, feature_dim = data.shape
        assert context_length == self.context_length
        assert num_nodes == self.num_nodes

        inputs = data.permute(0,1,3,2,4).contiguous().view(-1, num_nodes, context_length*feature_dim)

        x = self.msg_fc1(inputs)
        x = self.node2edge(x)
        x = self.msg_fc2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.msg_fc3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=2)

        output = self.hidden_fc(x)

        mu = self.mean_fc(output)
        if self.graph_type == 'numeric':
            if not self.unit_variance:
                var = self.variance_fc(output)
            else:
                var = torch.ones_like(mu, device=mu.device)
            return mu.view(batch_size, -1, mu.size(1)), var.view(batch_size, -1, var.size(1))
        else:
            return mu.view(batch_size, -1, mu.size(1)), None

class Entity2CommonEncoder(nn.Module):
    def __init__(self, params):
        super(Entity2CommonEncoder, self).__init__()
        self.graph_type = params['graph_type']
        self.unit_variance = params.get('unit_variance', False)
        self.min_var = params.get('min_var',1e-8)
        self.max_var = params.get('max_var', 100)

    def forward(self, input):
        _, num_subjects, _ = input.shape
        mean = torch.mean(input, 1, False)

        if self.graph_type == 'numeric':
            if not self.unit_variance:
                var = torch.var(input, 1, False)
            else:
                var = torch.ones_like(mean, device=mean.device)
            prec = var ** (-1) * num_subjects
            mu = (prec + 1) ** (-1) * prec * mean
            var = (prec + 1) ** (-1)
            var = torch.clamp(var, self.min_var, self.max_var)
            return mu, var
        else:
            alpha = num_subjects * mean + 1
            beta = num_subjects - num_subjects * mean + 1
            return alpha, beta


class LadderEncoder(nn.Module):

    def __init__(self, params):
        
        super(LadderEncoder, self).__init__()
        self.traj2graph_net = Traj2GraphEncoder(params)
        self.graph2common_net = Entity2CommonEncoder(params)
        self.graph_type = params['graph_type']

    def forward(self, data):
        mean_z, var_z = self.traj2graph_net(data)
        if self.graph_type == 'binary':
            mean_z = torch.clamp(torch.sigmoid(mean_z), min=_EPS, max=1-_EPS)
            
        param1_zbar, param2_zbar = self.graph2common_net(mean_z)
        return mean_z, var_z, param1_zbar, param2_zbar
