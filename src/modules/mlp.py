import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """
    a simple feed-forward network
    consisting of a sequence of linear -> activation -> dropout layers
    """
    def __init__(
        self,
        input_dim,                  ## int, input dimension
        output_dim,                 ## int, output dimension
        hidden_dims,                ## None or list of ints, hidden layer dimensions
        dropout_rate=0.1,
        activation='relu',
        batch_norm=True,
        use_softplus=False,
        clip=False,
        min_val=1e-5,
        max_val=100,
    ):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        self.clip = clip
        self.min_val = min_val
        self.max_val = max_val
        
        self.use_softplus = use_softplus
        if use_softplus:
            self.softplus_gate = nn.Softplus(1)
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(self.output_dim)
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError('currenctly not implemented')
        
        if hidden_dims is None:
            self.ffnet = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(self.activation)
                layers.append(nn.Dropout(self.dropout_rate))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            self.ffnet = nn.Sequential(*layers)
        
    def apply_bn(self, inputs):
    
        if inputs.ndim == 2:
            return self.bn(x)
        else:
            x = self.bn(inputs.view(inputs.size(0) * inputs.size(1),-1))
            return x.view(inputs.size(0), inputs.size(1), -1)
        
    def forward(self, x):
        
        out = self.ffnet(x)
        if self.batch_norm:
            out = self.apply_bn(out)
        
        if self.use_softplus:
            out = self.softplus_gate(out) + float(self.min_val)
        if self.clip:
            out = torch.clamp(out, self.min_val, self.max_val)
        return out
