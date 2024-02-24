"""
adapted from https://github.com/ethanfetaya/NRI/blob/master/data/synthetic_sim.py
"""

import os
import time
import datetime
import numpy as np

class simSprings():
    def __init__(self, params):

        super().__init__()
        self.n_nodes = params['num_nodes']
        self.loc_init = params['loc_init']
        self.vel_norm = params['vel_norm']
        self.interaction_strength = params['interaction_strength']
        self.noise_var = params['noise_var']
        self.box_size = params.get('box_size',5)

        self._spring_types = np.array([0.,1.])

        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        with np.errstate(divide='ignore'):
            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        '''
        Argvs:
        - loc: 2xN location at one time stamp
        - vel: 2xN velocity at one time stamp
        Returns:
        - location and velocity after hiting walls and returning after elastically colliding with walls
        '''

        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))

        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Argvs:
        - A np.array, Nxd
        - B np.array, Mxd
        Returns:
        - dist NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def generate_edges_from_beta(self, alpha=1, beta=1):
        """
        generate a weighted graph whose edges fall in between (0,1); edge weight are sampled from beta distribution
        """
        weights = np.random.beta(alpha,beta,size=(int(self.n_nodes*(self.n_nodes-1)/2),))
        edges = np.zeros((self.n_nodes, self.n_nodes))
        edges[np.tril_indices(self.n_nodes,-1)] = weights
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        return edges

    def generate_edges_from_prob(self, one_prob):
        """
        Generate binary edges based on the probability of one according to a potentially heterogeneous ER graph
        Argv:
        - one_prob: float in between [0,1], or np.array of size n_nodes*(n_nodes-1)/2
        """
        if isinstance(one_prob,float): ## homogeneous
            edges = np.random.choice(self._spring_types, size=(self.n_nodes, self.n_nodes), p=[1.0-one_prob, one_prob])
        else: ## heterogeneous
            assert len(one_prob) == int(self.n_nodes * (self.n_nodes-1)/2), f'len(one_prob)={len(one_prob)}; target_graph_size={int(self.n_nodes * (self.n_nodes-1)/2)}'
            edges = np.zeros((self.n_nodes, self.n_nodes))
            edges[np.tril_indices(self.n_nodes,-1)] = one_prob
            for i in range(self.n_nodes):
                for j in range(i):
                    edges[i,j] = np.random.choice(self._spring_types, size=1, p=[1.0-edges[i,j], edges[i,j]])

        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        return edges

    def perturb_edges(self, edges, epsilon=0.25):
        """
        flip the binary edges with epsilon - currently not used
        Argv:
        - epsilon: flipping probability
        """
        n_nodes = edges.shape[0]
        edges_copy = copy.deepcopy(edges)
        for i in range(n_nodes):
            for j in range(i):
                indicator = np.random.binomial(1, epsilon, size=1)
                if indicator == 1:
                    edges_copy[i,j] = 1 - edges_copy[i,j]

        edges_copy = np.tril(edges_copy) + np.tril(edges_copy, -1).T
        np.fill_diagonal(edges_copy, 0)
        return edges_copy

    def sample_one_trajectory(self, T_obs=10000, sample_freq=10, one_prob=0.5, fix_edges=None):
        """
        simulate a single trajectory based on either the specified graph (take priority) or the edge probability
        Argv:
        - T_obs: total number of steps forward
        - sample_freq: sampling freq for the final data
        - fix_edges: np.array of size [n_node, n_node] denoting the graph; symmetric with zero diagonal; if None, will be generated
        - one_prob: the probability of having an edge
        Returns:
        - loc, vel: size [T_save, 2, n_nodes], where T_save = int(T_obs / sample_freq - 1)
        - edges: the graph of size [n_node, n_node]

        see also https://github.com/ethanfetaya/NRI/blob/master/data/synthetic_sim.py#L73
        """

        assert T_obs % sample_freq == 0
        T_save = int(T_obs / sample_freq - 1)

        diag_mask = np.ones((self.n_nodes, self.n_nodes), dtype=bool)
        np.fill_diagonal(diag_mask, 0)

        counter = 0
        # sample edges
        if fix_edges is None:
            assert one_prob is not None
            edges = self.generate_edges_from_prob(one_prob)
        else:
            edges = fix_edges.copy()

        # Initialize location and velocity
        loc, loc_next = np.zeros((T_save, 2, self.n_nodes)), np.random.randn(2,self.n_nodes) * self.loc_init
        vel, vel_next = np.zeros((T_save, 2, self.n_nodes)), np.random.randn(2,self.n_nodes)

        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm/v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,0)  # self forces are zero (fixes division by zero)
            F = (forces_size.reshape(1, self.n_nodes, self.n_nodes) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, self.n_nodes, self.n_nodes),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, self.n_nodes, self.n_nodes)))).sum(axis=-1)
            F[F > self._max_F], F[F < -self._max_F] = self._max_F, -self._max_F

            vel_next += self._delta_T * F

            # run leapfrog
            for i in range(1, T_obs):

                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)
                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, self.n_nodes, self.n_nodes) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, self.n_nodes, self.n_nodes),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, self.n_nodes, self.n_nodes)))).sum(axis=-1)
                F[F > self._max_F], F[F < -self._max_F] = self._max_F, -self._max_F

                vel_next += self._delta_T * F

            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_nodes) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_nodes) * self.noise_var

            return loc, vel, edges

    def generate_dataset_one_subject(self, num_samples, T_obs, sample_freq=10, one_prob=0.5, fix_edges=None, seed=None):
        """
        see also https://github.com/ethanfetaya/NRI/blob/master/data/generate_dataset.py#L43
        NOTE: stacking list is slow; pre-allocate memory is much faster
        Returns:
        - locvel_all: [num_samples, T_save, self.n_nodes, 4] where T_save = int(T_obs/sample_freq-1)
        - edges: [num_samples, n_nodes, n_nodes]
        """

        ## need to re-seed: otherwise in multiprocessing we may end up with case where all subjects have the same graph
        if seed is not None:
            np.random.seed(seed)

        T_save = int(T_obs / sample_freq - 1)

        locvel_all = np.empty((num_samples, T_save, 4, self.n_nodes))
        edges_all = np.empty((num_samples, self.n_nodes, self.n_nodes))

        for i in range(num_samples):

            t0 = time.time()
            loc, vel, edges = self.sample_one_trajectory(T_obs=T_obs, sample_freq=sample_freq, one_prob=one_prob, fix_edges=fix_edges)

            locvel_all[i,:,:2,:], locvel_all[i,:,2:,:] = loc, vel
            edges_all[i,:,:] = edges

            if (i+1) % 100 == 0:
                t1 = time.time()
                print(f">> samples generated: {i+1}/{num_samples}, simulation time: {t1-t0:.2f}")
                t0 = t1

        locvel_all = np.transpose(locvel_all, (0,1,3,2)) ## transpose the last two axes
        return locvel_all, edges_all
