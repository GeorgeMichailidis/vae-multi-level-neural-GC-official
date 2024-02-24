
import os
import time
import datetime
import numpy as np
import math
import random
import pickle
from scipy.linalg import toeplitz

from .utils_simVAR import gen_VAR_trs, gen_LinearVAR_data, gen_NonLinearVAR_data, get_specradius, _to_companion, scale_trsmtx, get_Gamma

class _baseVAR():
    """
    base class for simulating the trajectories of N regions for M subjects according to a VAR(q) system
    """
    def __init__(self, params):

        super().__init__()
        self.n_nodes = params['num_nodes']                  ## number of nodes
        self.m_subjects = params['num_subjects']            ## number of subjects
        self.dispersion = params['dispersion']              ## standard dev of the normal distribution from which pertubations are drawn for each subject
        self.q_lags = params['q_lags']                      ## number of lags of the VAR system
        self.sigLow = params['sigLow']                      ## lower bound for the entry (raw, subject to scaling)
        self.sigHigh = params['sigHigh']                    ## upper bound for the entry (raw, subject to scaling)
        self.sigDecay = params['sigDecay']                  ## decay of the lower and upper bounds over lags
        self.specr_common = params['specr_common']          ## target spectral radius of the common transition matrix (in its companion form)
        self.sparsity_common = params['sparsity_common']    ## sparsity of the common transition matrix
        self.specr_max = params['specr_max']                ## the spectral radius if the case where an individual transition matrix becomes explosive after pertubation
        self.sigma_obs = params['sigma_obs']                ## noise for the observations
        self.rho = params.get('rho', 0.0)                   ## decaying coefficient for Toeplitz covariance matrix
        self.is_diagonal = params.get('is_diagonal', False)       ## diagonal graph only
        self.perturb_support = params.get('perturb_support', False) ## whether the perturbation is at the support level

        self._compute_sigma_matrix(self.sigma_obs, self.rho, self.n_nodes)

        ## placeholder for saving
        self.VAR_params = None

    def gen_VAR_params(self, update_attr = True):
        """
        generate the parameters for the dgp, in this case, common transition matrices then the pertubations
        subject-specific transition matrices are obtained as common + pertubation
        """
        ## generate the common component
        Abar = gen_VAR_trs(self.n_nodes,
                           nlags = self.q_lags,
                           sigLow = self.sigLow,
                           sigHigh = self.sigHigh,
                           sigDecay = self.sigDecay,
                           targetSR= self.specr_common,
                           sparsity = self.sparsity_common,
                           is_diagonal = self.is_diagonal)
        Gammabar = get_Gamma(Abar, Sigma_u = self.sigma_matrix)

        if not self.perturb_support:
            As, Ds = self._perturb_magnitude(Abar)
        else: ## note that in this process, the initially generated Abar also changes as certain entries will be flipped to zero
            Abar, As, Ds = self._perturb_support(Abar)

        ## rescale As and update Ds accordingly to ensure stationarity
        self._rescale_specradius(As, Ds, Abar)

        ## calculate the contemporaneous covariance matrix
        Gammas = []
        for i in range(self.m_subjects):
            Gamma_i = get_Gamma(As[i], Sigma_u = self.sigma_matrix)
            Gammas.append(Gamma_i)

        if update_attr:
            setattr(self, 'VAR_params', {'Abar': Abar, 'As': As, 'Gammabar': Gammabar, 'Gammas': Gammas})
        return

    def _compute_sigma_matrix(self, sigma_obs, rho, n_nodes):
        """
        compute coveriance matrix for noise term
        Argvs:
        - sigma_obs: standard deviation for the observation noise
        - rho: for Toeplitz covariance matrix, specify off diagonal C_ij = rho^abs(i-j)
        - n_nodes: number of nodes
        """
        self.sigma_matrix = toeplitz([rho ** i for i in range(n_nodes)]) * sigma_obs
        return

    def _rescale_specradius(self, As, Ds, Abar):
        """
        rescale the spectral radius of the perturbed transition matrices (after converting the stacked ones to their companion form)
        Argvs:
        - As: list of (stacked) transition matrices
        - Ds: list of deltas to be updated; in the case where there is rescaling, the true delta needs to be recalculated
        """
        for i, A_i in enumerate(As):
            spec_radius = get_specradius(_to_companion(A_i))
            if spec_radius >= self.specr_max:
                print(f'[WARNING] spectral_radius = {spec_radius:.3f} for A_{i} exceeds {self.specr_max:.2f}; auto rescaling with target spectral radius set to {self.specr_max:.2f}')
                As[i] = scale_trsmtx(A_i,target=self.specr_max)
                Ds[i] = As[i] - Abar
        return

    def _perturb_magnitude(self, Abar):
        """
        given Abar, subject-level graphs are obtained by perturbing the magnitude of the entries
        """
        # generate the idiosyncratic components and subject-specific transition matrices, assuming additive pertubation
        As, Ds = [], []
        for i in range(self.m_subjects):
            D_i = self.dispersion * np.random.uniform(low=-1.0, high=1.0, size=Abar.shape)
            if self.is_diagonal:
                for i in range(D_i.shape[-1]):
                    D_i[:,:,i] = np.diag(np.diag(D_i[:,:,i]))
            Ds.append(D_i)
            As.append(Abar + D_i)
        
        return Abar, As, Ds

    def _perturb_support(self, Abar_init):
        """
        given Abar_init of size [p,p,q], subject-level graphs are obtained by perturbing the support
        - we first determine the "fixed" support, i.e., the (non-zero) skeleton that is shared by all subjects
        - the flipped support is the complement (i.e., support - fixed support)
        - free entries = originally zero entries, which will be filled with the values from the flipped support at random
        in this process, the sparsity level for each subject is preserved the same as the common one
        after the pertubation, the guaranteed commonality across subjects is the fixed support (non-zero) + flipped support (converted to zero)
        heterogeneity across subjects should be bounded by 2 * sparsity_common * num_nodes^2 * perturbation
        """
        print(f'adding support perturbation to {self.m_subjects} subjects')
        
        Abar = Abar_init.copy()
        As = [Abar_init.copy() for subject_id in range(self.m_subjects)]
        for q in range(self.q_lags):
            
            support, freeEntries = list(zip(*np.nonzero(Abar_init[:,:,q]))), list(zip(*np.where(Abar_init[:,:,q] == 0)))
            
            supportFix = random.sample(support, math.ceil((1-self.dispersion)*len(support)))
            supportFlip = list(set(support) - set(supportFix))
            print(f'>> applying support perturbation; q={q}, size(support)={len(support)}, size(supportFix)={len(supportFix)}, size(supportFlip)={len(supportFlip)}')
            valuesFloat = [Abar[i,j,q] for (i,j) in supportFlip]
            
            ## flip Abar
            Abar[:,:,q][tuple(zip(*supportFlip))] = 0
            ## flip subject-graph + reallocate
            for subject_id in range(self.m_subjects):
                ## set support flip to zero
                As[subject_id][:,:,q][tuple(zip(*supportFlip))] = 0
                ## randomly choose from freeEntries and fill
                entriesNZ = random.sample(freeEntries, len(valuesFloat))
                As[subject_id][:,:,q][tuple(zip(*entriesNZ))] = valuesFloat

        Ds = [A - Abar for A in As]
        return Abar, As, Ds

class simLinearVAR(_baseVAR):
    """
    simulate the trajectories of N regions for M subjects according to a linear VAR(q) system
    """
    def __init__(self, params):
        super(simLinearVAR, self).__init__(params)

    def sim_trajectory(self, T_obs):

        if not self.VAR_params:
            print(f'VAR_params is missing; generating transition matrices ...')
            self.gen_VAR_params(update_attr = True)

        As = self.VAR_params['As']

        data = {}
        for i in range(self.m_subjects):
            data[i] = gen_LinearVAR_data(As[i], T_obs = T_obs, sigma_matrix = self.sigma_matrix)
        return data
