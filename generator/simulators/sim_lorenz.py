
import os
import numpy as np

from .utils_lorenz import simulate_lorenz_96

class simLorenz():
    """
    Generate time series that follows dynamics of Lorenz model
    """
    def __init__(self, params):
        self.n_nodes = params['num_nodes']
        self.m_subjects = params['num_subjects']
        self.forceLow = params['forceLow']
        self.forceHigh = params['forceHigh']
        assert params['forceDistribution'] in ['uniform', 'grid']
        self.forceDistribution = params['forceDistribution']

        self._get_gc_matrix()

    def _get_gc_matrix(self):
        """
        set up Granger causality ground truth.
        """
        p = self.n_nodes
        GC = np.zeros((p, p), dtype=int)
        for i in range(p):
            GC[i, i] = 1
            GC[i, (i + 1) % p] = 1
            GC[i, (i - 1) % p] = 1
            GC[i, (i - 2) % p] = 1

        self.gc_matrix = GC

    def gen_forces(self):
        """
        generate heterogeneous forces for different subject
        """
        if self.forceDistribution == 'uniform':
            forces = np.random.uniform(self.forceLow, self.forceHigh, self.m_subjects)
        else:
            forces = np.linspace(self.forceLow, self.forceHigh, self.m_subjects)
            
        setattr(self, 'Lorenz_params', {'forces': forces, 'GC': self.gc_matrix})
        return

    def sim_trajectory(self, T_obs, T_delta, sigma_obs):
        """
        simulate trajectory (with noise) whose dynamics follows Lorenz model
        """
        data = {}
        for i in range(self.m_subjects):
            force = self.Lorenz_params['forces'][i]
            print(f"simulating subject {i} with force {force}")
            data[i] = simulate_lorenz_96(self.n_nodes, T=T_obs, F=force, delta_t=T_delta, sd=sigma_obs, burn_in=1000)
        return data
