import numpy as np
from numba import njit, prange, config, set_num_threads
from psutil import cpu_count

from .vecchia_utils import find_nn, backward_solve, forward_solve, forward_solve_sp, get_pred_nn
from .gpbase import GPtide
from .cov import transform_coordinates


core_num = cpu_count(logical = False)
max_threads = config.NUMBA_NUM_THREADS
core_num = min(core_num, max_threads)
config.THREADING_LAYER = 'workqueue'
set_num_threads(core_num)


class GPtideVecchia(GPtide):
    """
        Gaussian Process regression class

        Uses the Vecchia approximation in all computations (likelihood, prediction, etc.)

        Parameters
        ----------
        xd: numpy.ndarray [N, D]
            Input data locations
        xm: numpy.ndarray [M, D]
            Output/target point locations
        noise_var: float
            Data noise parameter (as std dev)
        covfunc: callable function 
            Function used to compute the covariance matrices
        covparams: tuple
            Parameters passed to `covfunc` - assumes this is a tuple of (gp_stddev, lengthscale_1, ..., lengthscale_D)

        Other Parameters
        ----------------
        nnum: int (default=30)
            Number of nearest neighbours to use in the Vecchia approximation
        scale_coords: bool (default=False)
            Whether to scale the input coordinates prior to nearest neighbour search - important for anisotropic kernels
            This will assume the last D elements of `covparams` are the scaling elements (unless rotate_coords is True)
        rotate_coords: bool (default=False)
            Whether to rotate the input coordinates prior to nearest neighbour search
            This will assume the last D elements of `covparams` are the rotation angles
        cov_kwargs: dictionary, optional
            keyword arguments passed to `covfunc`, including the indexes of the scaling params inside covparams.
            E.g. {'diag_idx: [1, 2, 3], 'offdiag_idx': [4, 5, 6]} where covparams[cov_kwargs['diag_idx']] are the diagonal elements
            of the scaling matrix A and covparams[cov_kwargs['offdiag_idx']] are the off-diagonal elements.
        mean_func: callable function
            Returns the mean function
        mean_params: tuple
            parameters passed to the mean function
        mean_kwargs: dict
            kwargs passed to the mean function
        order_func: callable function
            Function that returns the re-ordering indexes of the points
        order_params: tuple
            Parameters passed to the order function
        order_kwargs: dict
            kwargs passed to the order function
        nn_kwargs: dict
            kwargs passed to the nearest neighbours function
        nn_array: numpy.ndarray
            the nearest neighbours array (if pre-computed)
            default is None
        P: int
            number of output dimensions
        jitter: float
            small value added to the diagonal of the covariance matrix
            default is 1e-7
    """

            
    def __init__(self, xd, xm, noise_var, covfunc, covparams, **kwargs):
        """
        Initialise GP object and evaluate mean and covariance functions. 
        
        We do not want to call _calc_cov or _calc_weights during init so we overwrite those functions.
        """       
        # Init calls `_calc_cov` and `_calc_weights` which we re-defined below
        GPtide.__init__(self, xd, xm, noise_var, covfunc, covparams, **kwargs)
        self.covparams = np.array(self.covparams)
        
        # Check mean and change to length N array if float supplied
        self._check_mean()
                    
        # Run the re-ordering function 
        self._reorder_inputs()
            
        # Build the scaling matrix
        if self.scale_coords:
            
            if self.rotate_coords:
                self.rotate_params = self.covparams[-self.D:]
            else:
                self.rotate_params = None

            # Re-scale everyything now so we don't have to repeat it in functions
            self.xd = transform_coordinates(self.xd, self.rotate_params, self.covparams[1:self.D+1])
            self.xm = transform_coordinates(self.xm, self.rotate_params, self.covparams[1:self.D+1])  
            
        # Find the nearest neighbours if not supplied
        if self.nn_array is None:
            self.find_neighbours(**self.nn_kwargs)
        else:
            assert self.order_idx is not None, 'Order index must be supplied if nn_array is pre-computed'
            self.nn_array = self.nn_array
            
        
    def _calc_cov(self, covfunc, covparams):
        """Compute the covariance functions"""
        return None, None
    
    def _calc_weights(self, Kdd, sd, Kmd):
        """Calculate the cholesky factorization"""
        return None, None 
    
    def _check_mean(self):
        # Check if type is float and convert to array - this is a bit shit
        if isinstance(self.mu_d, float):
            self.mu_d = np.repeat(self.mu_d, len(self.xd))[:,None]
        if (self.mu_d.ndim == 1):
            if len(self.mu_d) == 1:
                self.mu_d = np.repeat(self.mu_d, len(self.xd))
            else:
                assert len(self.mu_d) == len(self.xd), 'Length of mean function must be 1 or equal to input coords' 
            self.mu_d = self.mu_d#[:,None]            
    
    def _reorder_inputs(self):
        """Re-order the input data"""
        if self.order_idx is not None:
            if self.verbose:
                print('Using supplied re-ordering indexes and ignoring any re-ordering function')
            self.ord = self.order_idx
        elif self.order_func is not None:
            self.ord = self.order_func(self.order_params, **self.order_kwargs)
        else:
            self.ord = np.arange(self.N)
            if self.verbose:
                print('Warning: no re-ordering specified for Vecchia GP')

        # Re-order the input coords and mean
        self.orig_ord = np.argsort(self.ord)
        self.xd = self.xd[self.ord]
        self.mu_d = self.mu_d[self.ord]
        
    def find_neighbours(self, method='sklearn', rand=0, **faiss_kwargs):
        """
        Find the nearest neighbours for each input point in the data set. We should be re-scaling coords each time we call this. A different function is used for finding output nearest neighbours.
        
        This needs to be called inside __init__ or __call__ for gptide mle, makes more sense to call in __init__.
        """
        self.nn_array = find_nn(self.xd, self.nnum, method=method, rand=rand, verbose=self.verbose, **faiss_kwargs)

    def __call__(self, yd, method='sklearn', **faiss_kwargs):
        """
        Predict the GP posterior mean given data
        Parameters
        ----------
        yd: numpy.ndarray [N,1]
            Observed data
        Returns
        --------
        numpy.ndarray
            Prediction
        """
        if yd.ndim == 1:
            yd = yd[:,None]
        assert yd.shape[0] == self.N, ' first dimension in input data must equal '
        
        # Re-order the data to match coords
        self.yd = (yd[self.ord].T - self.mu_d.T).T
        
        # Compute prediction nearest neighbours
        if np.array_equiv(self.xd[self.orig_ord], self.xm):
            print('Using training points for prediction')
            nn_pred = self.nn_array  
            self.xm = self.xm[self.ord]
            return_ord = True
        else:
            # Prediction points don't need re-ordering
            nn_pred = get_pred_nn(self.xm, self.xd, m=self.nnum, method=method, **faiss_kwargs)
            return_ord = False
            
        self.mean, self.err = gp_vecch(self.xd, self.xm, nn_pred, self.yd, self.covfunc, self.covparams, self.sd**2)
        
        if return_ord:
            return (self.mu_m + self.mean)[self.orig_ord], (self.err**0.5)[self.orig_ord]
        else:
            return self.mu_m + np.squeeze(self.mean), self.err**0.5
        
    def sample_prior(self, ptype='input', samples=1, add_noise=False):
        """
        Sample from the prior distribution.
        
        Default is to sample once at the input locations without GP specified noise.
        """
        if ptype == 'input':
            Xt = self.xd
            nn_array = self.nn_array
            mu = np.squeeze(self.mu_d)
            
        # elif ptype == 'output':
        #     Xt = self.xm
        #     nn_array = find_nn(Xt, self.nnum)
        #     mu = np.squeeze(self.mu_m)
            
        if add_noise:
            noise = self.sd**2
        else:
            noise = self.jitter

        prior_samples = fmvn_mu_sp(Xt, nn_array, self.covfunc, self.covparams, noise, mu=mu, n_samples=samples)
        # Re-order samples before returning
        return prior_samples[self.orig_ord]  
        
    def log_marg_likelihood(self, yd):
        """Compute the log-likelihood function of the GP under Vecchia approximation.

        Args:
            x (ndarray): a numpy 1d-array that contains the values of log-transformed model parameters: 
                log-transformed lengthscales followed by the log-transformed nugget. 

        Returns:
            llik: the (positive) log-likelihood.
        """      
        if yd.ndim == 1:
            yd = yd[:,None]
        # Re-order the data to match coords
        self.yd = yd[self.ord]
        
        llik = vecchia_llik(self.xd, self.yd, self.mu_d, self.nn_array, self.covfunc, self.covparams, self.sd**2)
        return float(llik)
    
    
    def conditional(self, yd, samples=1):
        """
        Draw samples from the conditional distribution given observed data.

        Parameters
        ----------
        yd: numpy.ndarray [N, 1]
            Observed data
        samples: int (default=1)
            Number of samples to draw

        Returns
        -------
        numpy.ndarray
            Samples from the conditional distribution
        """
        if yd.ndim == 1:
            yd = yd[:, None]
        assert yd.shape[0] == self.N, 'First dimension in input data must equal the number of training points'

        # Re-order the data to match coords
        self.yd = yd[self.ord] - self.mu_d

        # Compute prediction nearest neighbours
        nn_pred = get_pred_nn(self.xm, self.xd, m=self.nnum)
        
        # Use gp_vecch to get the mean and variance of the predictions
        pred_mean, pred_var = gp_vecch(self.xd, self.xm, nn_pred, self.yd, self.covfunc, self.covparams, self.sd**2)

        # Draw samples from the conditional distribution
        samples_cond = np.random.randn(samples, len(pred_mean)) * np.sqrt(pred_var) + pred_mean

        return samples_cond.T
  

# @njit(cache=True)
def fmvn_mu_sp(X, NNarray, covfunc, covparams, noise, mu=0.0, n_samples=1):
    """
    Generate multivariate Gaussian random samples with means.
    I think Ming commented out numba because it didn't make any difference (in my testing)
    """
    d = X.shape[0]
    samples = np.zeros((X.shape[0], n_samples))
    
    # Compute the Vecchia approx. of the lower triangular matrix
    L = L_matrix(X, NNarray, covfunc, covparams, noise)
    
    for i in prange(n_samples):
        sn = np.random.randn(d)
        samples[:,i] = forward_solve_sp(L, NNarray, sn) + mu
    return np.squeeze(samples)
    
    
@njit(cache=True, parallel=True, fastmath=True)
def vecchia_llik(X, y, mu, NNarray, covfunc, covparams, noise):
    n = X.shape[0]
    quad, logdet = np.array([0.]), np.array([0.])
    for i in prange(n):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        xi, yi, mi = X[idx,:], y[idx,:], mu[idx]
        Ki = K_matrix(xi, xi, covfunc, covparams, noise)
        Li = np.linalg.cholesky(Ki)  
        Liyi = forward_solve(Li, yi - mi)
        
        # Log det
        logdet += 2*np.log(np.abs(Li[-1,-1]))
        
        # Quadratic form    
        quad += Liyi[-1]**2
        
    llik = -0.5*(logdet + quad)
    return llik


@njit(cache=True)
def K_matrix(X, Xpr, covfunc, covparams, noise):
    """
    Compute the covariance matrix using the specified kernel.
        
    X: Array of scaled points
    Xpr: Array of scaled points
    noise: Nugget (noise) term
    
    Returns:
    Covariance matrix.
    """    
    nx, _ = X.shape
    nxpr, _ = Xpr.shape
    K = np.zeros((nx, nxpr))

    for i in range(nx):
        for j in range(i + 1):
            if i == j:
                K[i, j] = covfunc(X[i,:], Xpr[j,:], covparams) + noise
                # K[i, j] = covparams[0]**2. + noise   ### this would be faster but won't work for mixed kernels
            else:
                K[i, j] = covfunc(X[i,:], Xpr[j,:], covparams)
                K[j, i] = K[i, j]
    return K


@njit(cache=True, parallel=True)
def L_matrix(X, NNarray, covfunc, covparams, noise):
    """
    Compute the lower triangular matrix L for each row of the input matrix X, using only the nearest neighbours in NNarray.
    """
    n, m = NNarray.shape
    L_matrix = np.zeros((n, m))
    for i in prange(n):
        idx = NNarray[i]
        idx = idx[idx>=0][::-1]
        bsize = len(idx)
        Ii = np.zeros(bsize)
        Ii[-1] = 1.0
        xi = X[idx,:]
        Ki = K_matrix(xi, xi, covfunc, covparams, noise)
        Li = np.linalg.cholesky(Ki)
        LiIi = backward_solve(Li.T, Ii)
        L_matrix[i, :bsize] = LiIi.T[0][::-1]
    return L_matrix


@njit(cache=True, parallel=True)
def gp_vecch(xd, xm, NNarray, yd, covfunc, covparams, noise):
    """
        Make GP predictions using the Vecchia approximation.

        Parameters
        ----------
        xd: Array of training points.
        xm: Array of prediction points.
        NNarray: Array of nearest neighbor indices.
        yd: Array of training observations.
        scale: Scale (variance) parameter of the kernel.
        noise: Noise (variance) parameter of the kernel.

        Returns
        --------

        pred_mean: numpy.ndarray
            Mean of the predictions.
        pred_var: numpy.ndarray
            Variance of the predictions.
    """
    # Initiate output variables
    n_pred = xm.shape[0]
    pred_mean, pred_var = np.zeros(n_pred), np.zeros(n_pred)
    
    # Loop through the prediction points
    for i in prange(n_pred):
        # Get the indices of the neighbours
        idx = NNarray[i]
        idx = idx[idx >= 0]
        # Stack the training and prediction points
        Xi = np.vstack((xd[idx, :], xm[i:i+1, :]))
        # Compute the covariance matrix
        Ki = K_matrix(Xi, Xi, covfunc, covparams, noise)
        Li = np.linalg.cholesky(Ki)
        yi = yd[idx]
        pred_mean[i] = np.dot(Li[-1, :-1], forward_solve(Li[:-1, :-1], yi).flatten())
        pred_var[i] = Li[-1, -1]**2
    return pred_mean, pred_var


def build_scaling_matrix(length_scales, covariances):
    """
    Build an n x n covariance matrix where the diagonal contains the length scales
    and the off-diagonals contain the covariances between dimensions.

    Parameters
    ----------
    length_scales: list or array of length scales (diagonal elements)
    covariances: list or array of covariances (off-diagonal elements)

    Returns
    -------
    covariance_matrix: n x n numpy array
        The resulting covariance matrix.
    """
    n = len(length_scales)

    # Create an empty matrix
    covariance_matrix = np.zeros((n, n))

    # Fill the diagonal with length scales
    for i in range(n):
        covariance_matrix[i, i] = length_scales[i]

    # Fill the off-diagonal elements with covariances
    if covariances is None:
        return covariance_matrix
    
    else:
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                covariance_matrix[i, j] = covariances[k]
                covariance_matrix[j, i] = covariances[k]
                k += 1

        return covariance_matrix


# def rotation_matrix_3d(theta_x, theta_y, theta_z):
#     # Rotation matrix around the x-axis
#     R_x = np.array([[1, 0, 0],
#                     [0, np.cos(theta_x), -np.sin(theta_x)],
#                     [0, np.sin(theta_x), np.cos(theta_x)]])
    
#     # Rotation matrix around the y-axis
#     R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
#                     [0, 1, 0],
#                     [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
#     # Rotation matrix around the z-axis
#     R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                     [np.sin(theta_z), np.cos(theta_z), 0],
#                     [0, 0, 1]])
    
#     # Combined rotation matrix
#     R = R_z @ R_y @ R_x
#     return R