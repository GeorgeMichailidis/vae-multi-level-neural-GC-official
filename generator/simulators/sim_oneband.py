
import time
import math
import numpy as np

class _simOneBandedVARBase():
    """
    simulate the trajectories of a 1-band VAR system with perturbation
    """
    def __init__(self, params):

        super().__init__()
        self.n_nodes = params['num_nodes']
        self.m_subjects = params['num_subjects']
        self.sigma_obs = params['sigma_obs']
        
    def _init_GC(self):
        GC_init = np.eye(self.n_nodes)
        for i in range(self.n_nodes):
            if i == 0:
                GC_init[i, (i+1)] = 1
            elif i == self.n_nodes - 1:
                GC_init[i, (i-1)] = 1
            else:
                GC_init[i, (i+1)] = 1
                GC_init[i, (i-1)] = 1
        return GC_init
        
    def _collect_perturbed_GCs(self, GC_init):
        ## induce perturbation
        GC_by_subject, changes_by_subject = [], []
        for subject_id in range(self.m_subjects):
            GC, changes = self._apply_perturbation(GC_init)
            GC_by_subject.append(GC)
            changes_by_subject.append(changes)
        
        return GC_by_subject, changes_by_subject
        
    def _apply_perturbation(self, mtx):
        raise NotImplementedError
        
    def gen_VAR_params(self, update_attr=True):
        """
        set up the VAR parameters in accordance with the perturbation logic
        the resulting ones correspond to the ground truth GC
        """
        raise NotImplementedError
    
    def _sim_one_trajectory(self):
        raise NotImplementedError
    
    def sim_trajectory(self, T_obs):
        """
        simulate the trajectories of all subjects based on their respective GCs
        """
        if not self.VAR_params:
            print(f'VAR_params is missing; generating GC matrices ...')
            self.gen_VAR_params(update_attr=True)

        GCs = self.VAR_params['GCs']
        data = {}
        for i in range(self.m_subjects):
            t0 = time.time()
            data[i] = self._sim_one_trajectory(GCs[i], T_obs = T_obs)
            t1 = time.time()
            print(f">> trajectories for subject {i+1}/{self.m_subjects} generated, simulation time: {t1-t0:.2f}s")
        return data

class simOneBanded0VAR(_simOneBandedVARBase):
    """
    - no perturbation, +/- 1 band
    """
    def __init__(self, params):
        super(simOneBanded0VAR,self).__init__(params)
        
    def gen_VAR_params(self, update_attr=True):
        
        ## the original (common) GC has a band of 2
        GC_bar = self._init_GC()
        GC_by_subject = np.stack([GC_bar for _ in range(self.m_subjects)], axis=0)
        
        if update_attr:
            self.VAR_params = {'GCbar': GC_bar, 'GCs': GC_by_subject}
    
    def _assert(self, j, idx0, idx1, idx2):
        ## the following should always hold according to the perturbation logic
        assert idx1 == j
        if j > 0 and j < self.n_nodes - 1:
            assert idx0 == j-1 and idx2 == j+1, f'j={j},idx0={idx0},idx2={idx2}; expecting j={j},idx0={j-1},idx2={j+1}'
        return
    
    def _sim_one_trajectory(self, GC, T_obs, burn_in=200, show_snr=False):
        
        T = int(T_obs + burn_in)
        noise = self.sigma_obs * np.random.normal(size=(T, self.n_nodes))
        signal = np.zeros((T,self.n_nodes))
        x = np.empty((T,self.n_nodes))
        
        for t in range(1,T):
            for j in range(self.n_nodes):
                if j == 0:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j+1]
                elif j == self.n_nodes - 1:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j-1]
                else: ## [1, self.n_nodes -2)
                    idx0, idx1, idx2 = np.nonzero(GC[j,:])[0] ## from left to right: idx of the support
                    self._assert(j, idx0, idx1, idx2)
                    signal[t,j] = 0.25*x[t-1,j] + np.sin(x[t-1,idx0]*x[t-1,idx2]) + np.cos(x[t-1,idx0]+x[t-1,idx2])
                x[t,j] = signal[t,j] + noise[t,j]
        
        if show_snr:
            signal, noise = signal[burn_in:,:], noise[burn_in:,:]
            signal_var = np.var(signal,axis=0)
            noise_var = np.var(noise,axis=0)
            snr = signal_var/noise_var
            snr_str=','.join([f'{snr[idx]:.2f}' for idx in range(self.n_nodes)])
            print(f'avgSNR={np.mean(snr):.2f}; snr=[{snr_str}]')
        
        return x[burn_in:,:]

class simOneBanded1VAR(_simOneBandedVARBase):
    """
    - when perturbation is present, original support is moved to other locations
    - only interaction effects are present; every other row is perturbed
    """
    def __init__(self, params):
        super(simOneBanded1VAR,self).__init__(params)
        
    def _apply_perturbation(self, mtx):
        """
        pertubation logic/steps:
        - perturbation is done along rows and is absent every other row (i.e., row_id that are multiples of 2 will be fixed)
        - diagonal elements are held fixed and will not be perturbed whatsoever
        - for elements within the +/- 1band and when they are perturbed: flip a bernoulli with certain success prob; if it comes as head, it will be "moved" to some other location
        """
        mtx = mtx.copy()
        changes = [] ## record the moved position (row_id, original_col_id, new_col_id)
        
        for row_id in range(1, self.n_nodes-2):
            if row_id % 2 == 0:
                continue
            else:
                pool = [row_id-1, row_id+1]
                for col_id in pool:
                    eligible_slots = list(range(0, row_id-1)) if col_id < row_id else list(range(row_id+2,self.n_nodes))
                    if len(eligible_slots):
                        new_col_id = np.random.choice(eligible_slots, size=1)[0]
                        mtx[row_id, col_id] = 0
                        mtx[row_id, new_col_id] = 1
                        changes.append([row_id, col_id, new_col_id])
        
        changes = np.array(changes)
        return mtx, changes
        
    def gen_VAR_params(self, update_attr=True):
        
        ## the original (common) GC has a band of 2
        GC_init = self._init_GC()
            
        ## induce perturbation
        GC_by_subject, changes_by_subject = self._collect_perturbed_GCs(GC_init)
        
        ## create GC bar based on the perturbation logic -- this gives the "probabilistic" info
        GC_bar_prob = GC_init.copy()
        for row_id in range(1, self.n_nodes-2):
            if row_id % 2 == 0:
                continue
            else:
                pool = [row_id-1, row_id+1] ## elements that are getting moved
                for col_id in pool:
                    eligible_slots = list(range(0, row_id-1)) if col_id < row_id else list(range(row_id+2,self.n_nodes))
                    if len(eligible_slots) > 0:
                        ## the original position is flipped to zero
                        GC_bar_prob[row_id, col_id] = 0
                        ## eligible positions have 1/(len(eligible_slots)) of getting chosen
                        for pos in eligible_slots:
                            GC_bar_prob[row_id, pos] = 1/(len(eligible_slots))
            
        GC_bar = 1 * (GC_bar_prob==1)
        if update_attr:
            self.VAR_params = {'GCbar': GC_bar, 'GCs': GC_by_subject, 'GCbar_prob': GC_bar_prob, 'GC_init': GC_init, 'changes_by_subject': changes_by_subject}
        return
        
    def _assert(self, j, idx0, idx1, idx2):
        ## the following should always hold according to the perturbation logic
        assert idx1 == j
        if j % 2 == 0: ## odd, no move
            assert idx0 == j-1 and idx2 == j+1, f'j={j},idx0={idx0},idx2={idx2}; expecting j={j},idx0={j-1},idx2={j+1}'
        return
    
    def _sim_one_trajectory(self, GC, T_obs, burn_in=200, show_snr=False):
        
        T = int(T_obs + burn_in)
        noise = self.sigma_obs * np.random.normal(size=(T, self.n_nodes))
        signal = np.zeros((T,self.n_nodes))
        x = np.empty((T,self.n_nodes))
        
        for t in range(1,T):
            for j in range(self.n_nodes):
                if j == 0:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j+1]
                elif j == self.n_nodes - 1:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j-1]
                else: ## [1, self.n_nodes -2)
                    idx0, idx1, idx2 = np.nonzero(GC[j,:])[0] ## from left to right: idx of the support
                    self._assert(j, idx0, idx1, idx2)
                    signal[t,j] = 0.25*x[t-1,j] + np.sin(x[t-1,idx0]*x[t-1,idx2]) + np.cos(x[t-1,idx0]+x[t-1,idx2])
                    #if j % 3 == 0: ## unperturbed row, common component: strengthen it by the main effects
                    #    signal[t,j] += 0.5*x[t-1,idx0] - 0.5*x[t-1,idx2]
                x[t,j] = signal[t,j] + noise[t,j]
        
        ## calculate signal-to-noise
        if show_snr:
            signal, noise = signal[burn_in:,:], noise[burn_in:,:]
            signal_var = np.var(signal,axis=0)
            noise_var = np.var(noise,axis=0)
            snr = signal_var/noise_var
            snr_str=','.join([f'{snr[idx]:.2f}' for idx in range(self.n_nodes)])
            print(f'avgSNR={np.mean(snr):.2f}; snr=[{snr_str}]')
        
        return x[burn_in:,:]

class simOneBanded2VAR(_simOneBandedVARBase):
    """
    - when perturbation is present, original support is moved to other locations
    - only interaction effects are present; every 3rd row is perturbed
    """
    def __init__(self, params):
        super(simOneBanded2VAR,self).__init__(params)
        
    def _apply_perturbation(self, mtx):
        """
        pertubation logic/steps:
        - perturbation is done along rows and is absent every 3rd row (i.e., row_id that are multiples of 3 will be fixed)
        - diagonal elements are held fixed and will not be perturbed whatsoever
        - for elements within the +/- 1band and when they are perturbed: flip a bernoulli with certain success prob; if it comes as head, it will be "moved" to some other location
        """
        mtx = mtx.copy()
        changes = [] ## record the moved position (row_id, original_col_id, new_col_id)
        
        for row_id in range(2, self.n_nodes-2):
            if row_id % 3 == 0:
                continue
            else:
                pool = [row_id-1, row_id+1]
                for col_id in pool:
                    eligible_slots = list(range(0, row_id-1)) if col_id < row_id else list(range(row_id+2,self.n_nodes))
                    if len(eligible_slots):
                        new_col_id = np.random.choice(eligible_slots, size=1)[0]
                        mtx[row_id, col_id] = 0
                        mtx[row_id, new_col_id] = 1
                        changes.append([row_id, col_id, new_col_id])
        
        changes = np.array(changes)
        return mtx, changes
        
    def gen_VAR_params(self, update_attr=True):
        
        ## the original (common) GC has a band of 2
        GC_init = self._init_GC()
            
        ## induce perturbation
        GC_by_subject, changes_by_subject = self._collect_perturbed_GCs(GC_init)
        
        ## create GC bar based on the perturbation logic -- this gives the "probabilistic" info
        GC_bar_prob = GC_init.copy()
        for row_id in range(2, self.n_nodes-2):
            if row_id % 3 == 0:
                continue
            else:
                pool = [row_id-1, row_id+1] ## elements that are getting moved
                for col_id in pool:
                    eligible_slots = list(range(0, row_id-1)) if col_id < row_id else list(range(row_id+2,self.n_nodes))
                    if len(eligible_slots) > 0:
                        ## the original position is flipped to zero
                        GC_bar_prob[row_id, col_id] = 0
                        ## eligible positions have 1/(len(eligible_slots)) of getting chosen
                        for pos in eligible_slots:
                            GC_bar_prob[row_id, pos] = 1/(len(eligible_slots))
            
        GC_bar = 1 * (GC_bar_prob==1)
        if update_attr:
            self.VAR_params = {'GCbar': GC_bar, 'GCs': GC_by_subject, 'GCbar_prob': GC_bar_prob, 'GC_init': GC_init, 'changes_by_subject': changes_by_subject}
        return
        
    def _assert(self, j, idx0, idx1, idx2):
        ## the following should always hold according to the perturbation logic
        assert idx1 == j
        if j % 3 == 0: ## odd, no move
            assert idx0 == j-1 and idx2 == j+1, f'j={j},idx0={idx0},idx2={idx2}; expecting j={j},idx0={j-1},idx2={j+1}'
        return
    
    def _sim_one_trajectory(self, GC, T_obs, burn_in=200, show_snr=False):
        
        T = int(T_obs + burn_in)
        noise = self.sigma_obs * np.random.normal(size=(T, self.n_nodes))
        signal = np.zeros((T,self.n_nodes))
        x = np.empty((T,self.n_nodes))
        
        for t in range(1,T):
            for j in range(self.n_nodes):
                if j == 0:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j+1]
                elif j == self.n_nodes - 1:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j-1]
                #elif j == 1 or j == self.n_nodes - 2:
                #    signal[t,j] = 0.25*x[t-1,j] + 0.5*x[t-1,j-1] - 0.5*x[t-1,j+1]
                elif j == 1 or j == self.n_nodes - 2:
                    signal[t,j] = np.sin(x[t-1,j]*x[t-1,j+1]) + 0.5*np.cos(x[t-1,j]+x[t-1,j-1])
                else: ## [2, self.n_nodes -2)
                    idx0, idx1, idx2 = np.nonzero(GC[j,:])[0] ## from left to right: idx of the support
                    self._assert(j, idx0, idx1, idx2)
                    signal[t,j] = 0.25*x[t-1,j] + np.sin(x[t-1,idx0]*x[t-1,idx2]) + np.cos(x[t-1,idx0]+x[t-1,idx2])
                    #if j % 3 == 0: ## unperturbed row, common component: strengthen it by the main effects
                    #    signal[t,j] += 0.5*x[t-1,idx0] - 0.5*x[t-1,idx2]
                
                x[t,j] = signal[t,j] + noise[t,j]
        
        ## calculate signal-to-noise
        if show_snr:
            signal, noise = signal[burn_in:,:], noise[burn_in:,:]
            signal_var = np.var(signal,axis=0)
            noise_var = np.var(noise,axis=0)
            snr = signal_var/noise_var
            snr_str=','.join([f'{snr[idx]:.2f}' for idx in range(self.n_nodes)])
            print(f'avgSNR={np.mean(snr):.2f}; snr=[{snr_str}]')
        
        return x[burn_in:,:]

class simOneBanded3VAR(_simOneBandedVARBase):
    """
    - only the diagonals are held intact
    """
    def __init__(self, params):
        super(simOneBanded3VAR,self).__init__(params)
        
    def _apply_perturbation(self, mtx):
        """
        pertubation logic/steps:
        - perturbation is done along rows and is absent every other row (i.e., row_id that are multiples of 2 will be fixed)
        - diagonal elements are held fixed and will not be perturbed whatsoever
        - for elements within the +/- 1band and when they are perturbed: flip a bernoulli with certain success prob; if it comes as head, it will be "moved" to some other location
        """
        mtx = mtx.copy()
        changes = [] ## record the moved position (row_id, original_col_id, new_col_id)
        
        for row_id in range(1, self.n_nodes-2):
            pool = [row_id-1, row_id+1]
            for col_id in pool:
                eligible_slots = list(range(0, row_id-1)) if col_id < row_id else list(range(row_id+2,self.n_nodes))
                if len(eligible_slots):
                    new_col_id = np.random.choice(eligible_slots, size=1)[0]
                    mtx[row_id, col_id] = 0
                    mtx[row_id, new_col_id] = 1
                    changes.append([row_id, col_id, new_col_id])
        
        changes = np.array(changes)
        return mtx, changes
        
    def gen_VAR_params(self, update_attr=True):
        
        ## the original (common) GC has a band of 2
        GC_init = self._init_GC()
            
        ## induce perturbation
        GC_by_subject, changes_by_subject = self._collect_perturbed_GCs(GC_init)
        
        ## create GC bar based on the perturbation logic -- this gives the "probabilistic" info
        GC_bar_prob = GC_init.copy()
        for row_id in range(1, self.n_nodes-1):
            pool = [row_id-1, row_id+1] ## elements that are getting moved
            for col_id in pool:
                eligible_slots = list(range(0, row_id-1)) if col_id < row_id else list(range(row_id+2,self.n_nodes))
                if len(eligible_slots) > 0:
                    ## the original position is flipped to zero
                    GC_bar_prob[row_id, col_id] = 0
                    ## eligible positions have 1/(len(eligible_slots)) of getting chosen
                    for pos in eligible_slots:
                        GC_bar_prob[row_id, pos] = 1/(len(eligible_slots))
            
        GC_bar = 1 * (GC_bar_prob==1)
        if update_attr:
            self.VAR_params = {'GCbar': GC_bar, 'GCs': GC_by_subject, 'GCbar_prob': GC_bar_prob, 'GC_init': GC_init, 'changes_by_subject': changes_by_subject}
        return
        
    def _assert(self, j, idx0, idx1, idx2):
        ## the following should always hold according to the perturbation logic
        assert idx1 == j
        return
    
    def _sim_one_trajectory(self, GC, T_obs, burn_in=200, show_snr=False):
        
        T = int(T_obs + burn_in)
        noise = self.sigma_obs * np.random.normal(size=(T, self.n_nodes))
        signal = np.zeros((T,self.n_nodes))
        x = np.empty((T,self.n_nodes))
        
        for t in range(1,T):
            for j in range(self.n_nodes):
                if j == 0:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j+1]
                elif j == self.n_nodes - 1:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j-1]
                else: ## [1, self.n_nodes -2]
                    idx0, idx1, idx2 = np.nonzero(GC[j,:])[0] ## from left to right: idx of the support
                    self._assert(j, idx0, idx1, idx2)
                    signal[t,j] = 0.25*x[t-1,j] + np.sin(x[t-1,idx0]*x[t-1,idx2]) + np.cos(x[t-1,idx0]+x[t-1,idx2])
                x[t,j] = signal[t,j] + noise[t,j]
        
        ## calculate signal-to-noise
        if show_snr:
            signal, noise = signal[burn_in:,:], noise[burn_in:,:]
            signal_var = np.var(signal,axis=0)
            noise_var = np.var(noise,axis=0)
            snr = signal_var/noise_var
            snr_str=','.join([f'{snr[idx]:.2f}' for idx in range(self.n_nodes)])
            print(f'avgSNR={np.mean(snr):.2f}; snr=[{snr_str}]')
        
        return x[burn_in:,:]

class simOneBanded4VAR(_simOneBandedVARBase):
    """
    - only the first and the last rows are shared
    """
    def __init__(self, params):
        super(simOneBanded4VAR,self).__init__(params)
        
    def _apply_perturbation(self, mtx):
        """
        pertubation logic/steps:
        - perturbation is done along rows and is absent every other row (i.e., row_id that are multiples of 2 will be fixed)
        - diagonal elements are held fixed and will not be perturbed whatsoever
        - for elements within the +/- 1band and when they are perturbed: flip a bernoulli with certain success prob; if it comes as head, it will be "moved" to some other location
        """
        mtx = mtx.copy()
        changes = [] ## record the moved position (row_id, original_col_id, new_col_id)
        
        for row_id in range(1, self.n_nodes-2):
            old_pool = [row_id-1, row_id, row_id+1]
            new_pool = np.random.choice(list(range(self.n_nodes)), size=3, replace=False)
            ## flip the old to zero
            for col_id in old_pool:
                mtx[row_id, col_id] = 0
            ## set the news to 1
            for new_col_id in new_pool:
                mtx[row_id, new_col_id] = 1
            ## record the changes
            for idx in range(len(old_pool)):
                changes.append([row_id, old_pool[idx], new_pool[idx]])
            
        changes = np.array(changes)
        return mtx, changes
        
    def gen_VAR_params(self, update_attr=True):
        
        ## the original (common) GC has a band of 2
        GC_init = self._init_GC()
            
        ## induce perturbation
        GC_by_subject, changes_by_subject = self._collect_perturbed_GCs(GC_init)
        
        ## create GC bar based on the perturbation logic -- this gives the "probabilistic" info
        GC_bar_prob = GC_init.copy()
        for row_id in range(1, self.n_nodes-1):
            GC_bar_prob[row_id, :] = 1/self.n_nodes
            
        GC_bar = 1 * (GC_bar_prob==1)
        if update_attr:
            self.VAR_params = {'GCbar': GC_bar, 'GCs': GC_by_subject, 'GCbar_prob': GC_bar_prob, 'GC_init': GC_init, 'changes_by_subject': changes_by_subject}
        return
        
    def _assert(self, j, idx0, idx1, idx2):
        ## nothing can be aseerted here
        return
    
    def _sim_one_trajectory(self, GC, T_obs, burn_in=200, show_snr=False):
        
        T = int(T_obs + burn_in)
        noise = self.sigma_obs * np.random.normal(size=(T, self.n_nodes))
        signal = np.zeros((T,self.n_nodes))
        x = np.empty((T,self.n_nodes))
        
        for t in range(1,T):
            for j in range(self.n_nodes):
                if j == 0:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j+1]
                elif j == self.n_nodes - 1:
                    signal[t,j] = 0.4*x[t-1,j] - 0.5*x[t-1,j-1]
                else: ## [1, self.n_nodes -2]
                    idx0, idx1, idx2 = np.nonzero(GC[j,:])[0] ## from left to right: idx of the support
                    self._assert(j, idx0, idx1, idx2)
                    signal[t,j] = 0.25*x[t-1,j] + np.sin(x[t-1,idx0]*x[t-1,idx2]) + np.cos(x[t-1,idx0]+x[t-1,idx2])
                x[t,j] = signal[t,j] + noise[t,j]
        
        ## calculate signal-to-noise
        if show_snr:
            signal, noise = signal[burn_in:,:], noise[burn_in:,:]
            signal_var = np.var(signal,axis=0)
            noise_var = np.var(noise,axis=0)
            snr = signal_var/noise_var
            snr_str=','.join([f'{snr[idx]:.2f}' for idx in range(self.n_nodes)])
            print(f'avgSNR={np.mean(snr):.2f}; snr=[{snr_str}]')
        
        return x[burn_in:,:]
