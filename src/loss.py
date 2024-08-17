import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.kl import kl_divergence

class KL():
    def __init__(self):
        pass

    @staticmethod
    def reduction(x, reduction = 'mean'):
        if reduction is None:
            return x
        elif reduction == 'mean':
            return x.mean()
        elif reduction == 'sum':
            return x.sum()
        else:
            raise ValueError('unrecognized reduction method')

    @classmethod
    def normal(cls, mu_1, var_1, mu_2, var_2, reduction=None):
        p, q = Normal(mu_1,var_1**0.5), Normal(mu_2,var_2**0.5)
        kl = kl_divergence(p,q)
        return cls.reduction(kl, reduction = reduction)

    @classmethod
    def beta(cls, alpha_1, beta_1, alpha_2, beta_2, reduction=None):
        p, q = Beta(alpha_1,beta_1), Beta(alpha_2, beta_2)
        kl = kl_divergence(p,q)
        return cls.reduction(kl, reduction = reduction)

    @classmethod
    def bernoulli(cls, p_1, p_2, reduction=None):
        p, q = Bernoulli(p_1), Bernoulli(p_2)
        kl = kl_divergence(p,q)
        return cls.reduction(kl, reduction = reduction)

class MultiSubELBOLoss(nn.Module):

    def __init__(
        self,
        graph_type = 'numeric',
        loss_type = 'NLL'
    ):
        super(MultiSubELBOLoss, self).__init__()
        self.graph_type = graph_type
        self.loss_type = loss_type
        if loss_type == 'MSE':
            self.loss_obj = nn.MSELoss(reduction='mean')
        else:
            self.loss_obj = nn.GaussianNLLLoss(reduction='mean')

    def ladder_adjustment(self, mean_q, var_q, mean_p, var_p):
        if self.graph_type == 'numeric':
            var_q_inv, var_p_inv = 1./var_q, 1./var_p
            var_adj = 1./(var_q_inv + var_p_inv)
            mean_adj = (mean_q * var_q_inv + mean_p * var_p_inv) * var_adj
        else:
            mean_adj = 2./(1./mean_q + 1./mean_p)
            var_adj = None

        return mean_adj, var_adj

    def calc_reconstruction_err(self, targets, preds_mean, preds_var):
        _, _, timesteps, num_nodes, feature_dim = targets.shape
        targets = targets.view(-1, timesteps, num_nodes, feature_dim)
        preds_mean = preds_mean.view(-1, timesteps, num_nodes, feature_dim)
        preds_var = preds_var.view(-1, timesteps, num_nodes, feature_dim)

        if self.loss_type == 'NLL':
            return self.loss_obj(preds_mean, targets, preds_var)
        else:
            return self.loss_obj(preds_mean, targets)

    def calc_KL(self, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp):
        
        batch_size, num_subjects, num_edges = mean_zq.shape
        mean_zq_adj, var_zq_adj = self.ladder_adjustment(mean_zq, var_zq, mean_zp, var_zp)

        if self.graph_type == 'numeric':

            param1_zbarp, param2_zbarp = torch.zeros_like(param1_zbarq, device=param1_zbarq.device), torch.ones_like(param2_zbarq, device=param2_zbarq.device)
            KL1 = KL.normal(param1_zbarq.view(-1,1), param2_zbarq.view(-1,1), param1_zbarp.view(-1,1), param2_zbarp.view(-1,1), reduction='mean')

            mean_zq_adj, var_zq_adj = mean_zq_adj.view(-1,1), var_zq_adj.view(-1,1)
            mean_zp, var_zp = mean_zp.view(-1,1), var_zp.view(-1,1)
            KL2 = KL.normal(mean_zq_adj, var_zq_adj, mean_zp, var_zp, reduction='mean')

        elif self.graph_type == 'binary':

            param1_zbarp, param2_zbarp = torch.ones_like(param1_zbarq, device=param1_zbarq.device), torch.ones_like(param2_zbarq, device=param2_zbarq.device)
            KL1 = KL.beta(param1_zbarq.view(-1,1), param2_zbarq.view(-1,1), param1_zbarp.view(-1,1), param2_zbarp.view(-1,1), reduction='mean')

            mean_zq_adj = mean_zq_adj.view(-1,1)
            mean_zp  = mean_zp.view(-1,1)
            KL2 = KL.bernoulli(mean_zq_adj, mean_zp, reduction='mean')

        return KL1, KL2

    def forward(self, targets, preds_mean, preds_var, mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp):

        nll_loss = self.calc_reconstruction_err(targets, preds_mean, preds_var)
        kl1, kl2 = self.calc_KL(mean_zq, var_zq, param1_zbarq, param2_zbarq, mean_zp, var_zp)

        return nll_loss, kl1, kl2
        
class OneSubELBOLoss(nn.Module):

    def __init__(
        self,
        graph_type = 'numeric',
        loss_type = 'NLL',
        reduction = 'mean'
    ):
        super(OneSubELBOLoss, self).__init__()
        self.graph_type = graph_type
        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type == 'MSE':
            self.loss_obj = nn.MSELoss(reduction=self.reduction)
        else:
            self.loss_obj = nn.GaussianNLLLoss(reduction=self.reduction)

    def calc_reconstruction_err(self, targets, preds_mean, preds_var):
        
        _, _, timesteps, num_nodes, feature_dim = targets.shape

        targets = targets.view(-1, timesteps, num_nodes, feature_dim)
        preds_mean = preds_mean.view(-1, timesteps, num_nodes, feature_dim)
        preds_var = preds_var.view(-1, timesteps, num_nodes, feature_dim)

        if self.loss_type == 'NLL':
            return self.loss_obj(preds_mean, targets, preds_var)
        else:
            return self.loss_obj(preds_mean, targets)

    def calc_KL(self, mean_zq, var_zq):
        if self.graph_type == 'numeric':
            param1_zp, param2_zp = torch.zeros_like(mean_zq, device=mean_zq.device), torch.ones_like(var_zq, device=var_zq.device)
            kl = KL.normal(mean_zq.view(-1,1), var_zq.view(-1,1), param1_zp.view(-1,1), param2_zp.view(-1,1), reduction=self.reduction)
        elif self.graph_type == 'binary':
            param1_zp = 0.5*torch.ones_like(mean_zq, device=mean_zq.device)
            kl = KL.bernoulli(mean_zq.contiguous().view(-1,1), param1_zp.contiguous().view(-1,1), reduction=self.reduction)
        return kl

    def forward(self, targets, preds_mean, preds_var, mean_zq, var_zq):

        batch_size, _, timesteps, num_nodes, feature_dim = targets.shape
        if self.reduction == 'sum':
            normalizer = 1.0 * batch_size * num_nodes
        else:
            normalizer = 1.0

        nll_loss = self.calc_reconstruction_err(targets, preds_mean, preds_var)
        kl = self.calc_KL(mean_zq, var_zq)

        return nll_loss/normalizer, kl/normalizer
