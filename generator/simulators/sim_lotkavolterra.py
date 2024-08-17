## modified based on https://github.com/i6092467/GVAR/blob/master/datasets/lotkaVolterra/multiple_lotka_volterra.py

import time
import numpy as np

class simMultiLotkaVolterra():
    def __init__(self, params):
        """
        Dynamical multi-species Lotka--Volterra system. The original two-species Lotka--Volterra is a special case
        with p = 1 , d = 1.

        @param p: number of predator/prey species. Total number of variables is 2*p.
        @param d: number of GC parents per variable.
        @param d_extra: additional GC parents per variable.
        @param alpha: strength of interaction of a prey species with itself.
        @param beta: strength of predator -> prey interaction.
        @param gamma: strength of interaction of a predator species with itself.
        @param delta: strength of prey -> predator interaction.
        @param sigma: scale parameter for the noise.
        """

        self.p = params['num_nodes']//2
        self.d = params['d']
        self.d_extra = params.get('d_extra',0)

        assert self.p >= self.d and self.p % self.d == 0
        
        # coupling strengths
        self.alpha = params.get('alpha',1.1)
        self.beta = params.get('beta',0.2)
        self.gamma = params.get('gamma',1.1)
        self.delta = params.get('delta',0.05)
        self.sigma = params.get('sigma',0.1)

        # number of subjects
        self.m_subjects = params.get('num_subjects',1)

    def _start_end_idx(self,j):
        """ helper function for getting the base case start and the end index """
        start = int(np.floor((j + self.d) / self.d) * self.d - 1 - self.d + 1)
        end = int(np.floor((j + self.d) / self.d) * self.d)
        return start, end

    def _init_GC(self):
        
        GC = np.zeros((self.p*2, self.p*2))
        for j in range(self.p):
            GC[j, j] = +1
            GC[j+self.p, j+self.p] = -1
            # predator-prey relationships
            start, end = self._start_end_idx(j)
            GC[j, int(start+self.p):int(end+self.p)] = -1
            GC[j + self.p, start:end] = +1
        
        return GC

    def induce_GC_perturbation(self, GC_bar):

        GC = GC_bar.copy()
        eligible_slots = []
        for j in range(self.p):
            start, end = self._start_end_idx(j)
            eligible_tuple = [(j,idx) for idx in (list(range(0, start)) + list(range(end, self.p)))]
            eligible_slots.extend(eligible_tuple)

        extra_slots = np.random.choice(len(eligible_slots), size=self.d_extra, replace=False)
        for slot_idx in extra_slots:
            slot_tuple = eligible_slots[slot_idx]
            GC[slot_tuple[0],slot_tuple[1]+self.p] = -1
            GC[slot_tuple[1]+self.p,slot_tuple[0]] = +1
        
        return GC
            
    def _sim_one_trajectory(self, GC, t, dt=0.01, downsample_factor=10):
        
        xs_0 = np.random.uniform(10, 100, size=(self.p, ))
        ys_0 = np.random.uniform(10, 100, size=(self.p, ))

        ts = np.arange(t) * dt

        xs = np.zeros((t, self.p))
        ys = np.zeros((t, self.p))
        xs[0, :] = xs_0
        ys[0, :] = ys_0
        for k in range(t - 1):
            xs[k + 1, :], ys[k + 1, :] = self.next(xs[k, :], ys[k, :], GC, dt)

        return np.concatenate((xs[::downsample_factor, :], ys[::downsample_factor, :]), 1)

    def gen_GC_params(self):
        """
        main function for generating GC parameters
        """
        ## the original (common) GC
        GC_bar = self._init_GC()
        
        ## induce perturbation
        GC_by_subject = []
        for subject_id in range(self.m_subjects):
            GC = self.induce_GC_perturbation(GC_bar)
            GC_by_subject.append(GC)
        GC_by_subject = np.stack(GC_by_subject,axis=0)
        self.GC_params = {'GCbar': GC_bar, 'GCs': GC_by_subject}

    def sim_trajectory(self, T_obs, dt=0.01, downsample_factor=10):
        """
        main function for simulating the trajectories of all subjects based on their respective GCs
        """
        if not self.GC_params:
            print(f'GC_params is missing; generating GC matrices ...')
            self.gen_GC_params()

        GCs = self.GC_params['GCs']
        data = {}
        for i in range(self.m_subjects):
            t0 = time.time()
            data[i] = self._sim_one_trajectory(GCs[i], int(T_obs * downsample_factor), dt=dt, downsample_factor=downsample_factor)
            t1 = time.time()
            print(f">> trajectories for subject {i+1}/{self.m_subjects} generated, simulation time: {t1-t0:.2f}s")
        return data
    
    # Dynamics
    # State transitions using the Runge-Kutta method
    def next(self, x, y, GC, dt):
        
        xdot1, ydot1 = self.f(x, y, GC)
        xdot2, ydot2 = self.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2, GC)
        xdot3, ydot3 = self.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2, GC)
        xdot4, ydot4 = self.f(x + xdot3 * dt, y + ydot3 * dt, GC)
        # Add noise to simulations
        xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + np.random.normal(scale=self.sigma, size=(self.p, ))
        ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + np.random.normal(scale=self.sigma, size=(self.p, ))
        # Clip from below to prevent populations from becoming negative
        return np.maximum(xnew, 0), np.maximum(ynew, 0)

    def f(self, x, y, GC):
        
        """ simulate based on GC """
        np.fill_diagonal(GC, 0)

        xdot = np.zeros((self.p, ))
        ydot = np.zeros((self.p, ))

        for j in range(self.p):
            
            predators = [temp-self.p for temp in np.nonzero(GC[j,:])[0]]
            y_Nxj = y[predators]
            xdot[j] = self.alpha * x[j] - self.beta * x[j] * np.sum(y_Nxj) - self.alpha * (x[j] / 200) ** 2
            
            preys = np.nonzero(GC[self.p+j,:])[0]
            x_Nyj = x[preys]
            ydot[j] = self.delta * np.sum(x_Nyj) * y[j] - self.gamma * y[j]
            
        return xdot, ydot
