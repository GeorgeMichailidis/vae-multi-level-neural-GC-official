"""
utility functions for simulating a VAR system
"""
import numpy as np
import torch


def gen_VAR_trs(dim,nlags,sigLow,sigHigh,sigDecay=1,targetSR=0.5,sparsity=None,is_diagonal=False):
    """
    main function for generating the transition matrices for a stationary VAR(q) system
    Argvs:
    - dim: dimension of the system (p)
    - nlags: numer of lags (q)
    - sigLow, sigHigh: initial values for the signal lower and upper bound (subject to scaling)
    - sigDecay: the decay over lags for the signals
    - targetSR: the target spectral radius of after stacking the transition matrices into the companion form
    - sparsity: sparsity level (i.e., support-size/p*p); either a float or a list or None
    - is_diagonal: whether transition matrices are diagonal
    Returns:
    - an np.array of shape (dim, dim, nlags), with each slice being the transition matrix of the corresponding lag
    """
    if nlags == 1:
        A = _gen_VAR1_trs(dim, sigLow=sigLow, sigHigh=sigHigh, targetSR=targetSR, sparsity=sparsity, is_diagonal=is_diagonal)
    else:
        sigLows, sigHighs = [sigLow*(sigDecay)**i for i in range(nlags)], [sigHigh*(sigDecay)**i for i in range(nlags)]
        A = _gen_VARq_trs(dim, nlags, sigLows, sigHighs, targetSR=targetSR, sparsities=sparsity, is_diagonal=is_diagonal)
    return A

def get_Gamma(mtx_stacked, Sigma_u):
    """
    calculate Gamma(0) according to Yule-Walker for a VAR(q) process
    see Lütkepohl (2005) chapter 2.1.4
    Argvs:
    - mtx_stacked: stacked transition matrix, (p,p,q) where q is the number of lags
    - Sigma_u: np.array, covariance matrix for the noise term, p-by-p
    """
    p, _, q = mtx_stacked.shape
    effective_size = p*q

    A = _to_companion(mtx_stacked) ## companion matrix
    if q > 1:
        Sigma_u_new = np.zeros((effective_size, effective_size))
        Sigma_u_new[:p,:p] = Sigma_u
        Sigma_u = Sigma_u_new

    vec_Gamma = np.linalg.inv(np.eye(effective_size**2) - np.kron(A, A)) @ Sigma_u.reshape(-1,1)
    Gamma = vec_Gamma.reshape(effective_size,effective_size)
    return Gamma[:p,:p]

def gen_LinearVAR_data(A, T_obs, sigma_matrix, burn_in=200):
    """
    generate observations for a VAR system, given transition matrices, assuming Gaussian noise
    Argvs:
    - A: transition matrices for each lag, stacked into an np.array of shape (p,p,q)
    - T_obs: total number of observations
    - sigma_matrix: coveriance matrix of noise term
    - burn_in: initial burn_in period
    Returns:
    - an np.array of shape (T_obs, p) where p is the dimension of the system
    """
    p, _, q = A.shape
    T_obs_effective = int(T_obs + burn_in)
    x = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma_matrix, size=T_obs_effective)
    for t in range(q, T_obs_effective):
        for d in range(1,q+1):
            x[t,:] += np.dot(A[:,:,d-1], x[t-d,:])
    return x[burn_in:, ]

def gen_NonLinearVAR_data(A, T_obs, sigma, g=None, fs=None, burn_in=200):
    """
    generate observations for a VAR system, given transition matrices, assuming Gaussian noise
    x_t = g(A_1f_1(x_{t-1}) + A_2f_2(x_t-2), ... + A_qf_q(x_{t-q})) + noise
    Argvs:
    - A: transition matrices for each lag, stacked into an np.array of shape (p,p,q)
    - T_obs: total number of observations
    - sigma: standard deviation for the observation noise
    - g, fs: non-linear functional class, None (identity), rbfNetwork or sinNetwork
    - burn_in: initial burn_in period
    Returns:
    - an np.array of shape (T_obs, p) where p is the dimension of the system
    """
    p, _, q = A.shape
    T_obs_effective = int(T_obs + burn_in)
    x = sigma * np.random.normal(size=(T_obs_effective, p))
    for t in range(q, T_obs_effective):
        from_lags = np.zeros(p)
        for d in range(1,q+1):
            x_lag = x[t-d,:] if fs[d-1] is None else fs[d-1].forward(x[t-d,:].reshape(1,p)).squeeze().numpy()
            curr_term = np.dot(A[:,:,d-1], x_lag)
            from_lags += curr_term
        x[t,:] += from_lags if g is None else g.forward(from_lags.reshape(1,p)).squeeze().numpy()
    return x[burn_in:, ]

def get_specradius(mtx):
    """ obtain the spectral radius of a square matrix"""
    return max(np.abs(np.linalg.eigvals(mtx)))

def scale_trsmtx(mtx_stacked, target, verbose=True):
    """
    scale VAR transition matrices to ensure stationarity
    Argvs:
    - mtx_stacked: np.array of shape (p,p,q) where q is the number of lags, p is the dim of the system;
    - target: spectral radius target;
    Returns:
    - scaled mtx_stacked; np.array of shape (p,p,q)
    Note: the scaling mechanism can ensure that for q == 1, the target is attained exactly; for q>1, this is only an approximate
    """
    if mtx_stacked.ndim < 3:
        mtx_stacked = mtx_stacked[:,:,np.newaxis]

    mtx_companion = _to_companion(mtx_stacked)
    old_specr = get_specradius(mtx_companion)
    mtx_companion_scaled = (target/old_specr) * mtx_companion
    mtx_stacked_scaled = _from_companion(mtx_companion_scaled, mtx_stacked.shape[-1])

    ## since the scaling won't be exact, do some sanity checks
    mtx_companion_new = _to_companion(mtx_stacked_scaled)
    new_specr = get_specradius(mtx_companion_new)
    if verbose:
        print(f'spectral radius before scaling = {old_specr:.3f}, after scaling = {new_specr:.3f}')
    if new_specr >= 1:
        print(f'![WARNING]: spectral radius after scaling = {new_specr:.3f}')

    return mtx_stacked_scaled

def _gen_randmtx(m, n, sigLow=1, sigHigh=3, sparsity = None, is_diagonal=False):
    """ generate a m by n random matrix with sparsity """
    if sparsity is None:
        mtx = np.random.uniform(sigLow,sigHigh,size=(m,n))
    else:
        mtx = np.random.uniform(sigLow,sigHigh,size=(m,n))*np.random.binomial(1,sparsity,size=(m,n))
    mtx = mtx * np.random.choice([-1,1],size=(m,n))
    if is_diagonal:
        mtx_diag = np.diag(np.diag(mtx))
        mtx = np.zeros((m, n))
        mtx[:min(m,n), :min(m,n)] = mtx_diag
    return mtx

def _to_companion(mtx_stacked):
    """
    transform stacked transition matrices into the companion form
    see Lütkepohl (2005), New Introduction to Multiple Time Series Analysis, chapter 2.1
    Argv:
    - mtx_stacked: np.array of shape (p, p, q)
    Return:
    - transition matrix in the companion form, of shape (p*q, p*q)
    """
    p, _, q = mtx_stacked.shape

    if q == 1:
        #print('[WARNING] _to_companion: mtx_stacked.shape[-1] = 1, squeezing the last dimension')
        return mtx_stacked.squeeze(-1)

    identity = np.identity((q-1)*p) ## identity matrix of shape ((q-1)*p, (q-1)*p)
    zeros = np.zeros(((q-1)*p,p))
    bottom = np.concatenate([identity, zeros],axis=1)
    top = np.concatenate([mtx_stacked[:,:,i] for i in range(q)],axis=1)

    return np.concatenate([top,bottom],axis=0)

def _from_companion(mtx, q, verbose=False):
    """ extract the lag coefficients from a companion form of shape (p*q, p*q) and stack them into a 3d np.array of shape p*p*q """
    if q == 1:
        #print('[WARNING] _from_companion: q=1, adding a new dimension')
        return mtx[:,:,np.newaxis]

    assert mtx.shape[1] % q == 0, 'number of columns in the companion matrix is not a multiple of the number of lags; something went wrong'
    p = int(mtx.shape[1]//q)
    mtx_stacked = []
    for i in range(q):
        mtx_stacked.append(mtx[:p,(i*p):((i+1)*p)])
    return np.stack(mtx_stacked,axis=-1)

def _gen_VAR1_trs(p, sigLow=1, sigHigh=3, targetSR=0.5, sparsity=None, is_diagonal=False):
    """
    generate the transition matrix for a stationary VAR(1) process
    Returns: an np.array of shape (p,p,1)
    """
    mtx_raw = _gen_randmtx(p, p, sigLow, sigHigh, sparsity, is_diagonal)
    mtx_scaled = scale_trsmtx(mtx_raw,target=targetSR)
    return mtx_scaled

def _gen_VARq_trs(p, q, sigLows, sigHighs, targetSR=0.6, sparsities=None, is_diagonal=False):
    """
    generate transition matrices for a stationary VAR(q) process for q > 1
    (note that the targetSR will not be attained exactly)

    Argvs:
    - p: dimension of the system
    - q: number of lags
    - sigLows, sigHighs: lists of length q, storing the lower and upper bound
    - targetSR: target spectral radius of the companion matrix
    - sparsities: list of length q or None, storing the sparsity level for each lag
    -- is_diagonal: whether transition matrices are diagonal

    Returns:
    - an np.array of shape (p,p,q)
    """
    assert len(sigLows) == len(sigHighs)

    mtx_stacked_raw = []
    for i in range(q):
        mtx_raw = _gen_randmtx(p,p,sigLow=sigLows[i],sigHigh=sigHighs[i],sparsity=None if sparsities is None else sparsities[i],is_diagonal=is_diagonal)
        mtx_stacked_raw.append(mtx_raw)

    mtx_stacked_raw = np.stack(mtx_stacked_raw,axis=-1)
    mtx_stacked_scaled = scale_trsmtx(mtx_stacked_raw,target=targetSR)

    return mtx_stacked_scaled
