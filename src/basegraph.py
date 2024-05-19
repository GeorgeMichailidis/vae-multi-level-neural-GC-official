
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseGraph(nn.Module):
    def __init__(self, num_nodes, include_diag=False):
        
        super(BaseGraph, self).__init__()
        self.num_nodes = num_nodes
        self.include_diag = include_diag
        
        self._create_receiver_sender_indices()
        self._create_onehot_rel()
        
    def _encode_onehot(self, labels, classes):
        
        classes = set(classes)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),dtype=np.int32)
        return labels_onehot
    
    def _create_receiver_sender_indices(self):
        if self.include_diag:
            edge_indices = np.ones([self.num_nodes, self.num_nodes])
        else:
            edge_indices = np.ones([self.num_nodes, self.num_nodes]) - np.eye(self.num_nodes)
        self.receivers, self.senders = np.where(edge_indices)
        
    def _create_onehot_rel(self):
        
        classes = set(range(self.num_nodes))
        
        rel_rec = np.array(self._encode_onehot(self.receivers, classes))
        rel_send = np.array(self._encode_onehot(self.senders, classes))
        
        self.rel_rec = torch.tensor(rel_rec, dtype=torch.float32)
        self.rel_send = torch.tensor(rel_send, dtype=torch.float32)
    
    def node2edge(self, node_representation):
        
        receivers = torch.matmul(self.rel_rec.to(node_representation.device), node_representation)
        senders = torch.matmul(self.rel_send.to(node_representation.device), node_representation)
        return torch.cat([receivers, senders], dim=-1)
         
    def edge2node(self, x, normalize=True):
        incoming = torch.matmul(self.rel_rec.t().to(x.device), x)
        normalizer = 1.0 if not normalize else incoming.size(1)
        return incoming / normalizer
