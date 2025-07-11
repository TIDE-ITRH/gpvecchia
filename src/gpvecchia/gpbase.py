"""
Classes for Gaussian Process regression
"""


class GPtide(object):
    """
    Gaussian Process base class 

    Intended as a placeholder for classes built with other libraries (scipy, jax)
    
    """
    
    scale_coords = False
    rotate_coords = False
    
    mean_func = None
    mean_params = ()
    mean_kwargs = {}
    
    cov_kwargs = {}
    cov_args = ()
    jitter = 1e-7
    
    order_func = None
    order_params = ()
    order_kwargs = {}
    order_idx = None
    
    nn_kwargs = {}
    nnum = 30
    nn_array = None
    nn_pred_array = None
    nn_append = None
    P = 1
    
    verbose = False
        
    
    def __init__(self, xd, xm, sd, covfunc, covparams, **kwargs):
        """
        Initialise GP object and evaluate mean and covatiance functions. 
        """
        
        self.__dict__.update(kwargs)
        
        assert xd.ndim==2
        
        self.N, self.D = xd.shape
        self.M, D = xm.shape
        
        self.N = self.N*self.P
        self.M = self.M*self.P

        self.xd = xd
        self.xm = xm
        
        self.sd = sd
        self.covfunc = covfunc
        self.covparams = covparams
        
        # Evaluate the covariance functions
        self.Kmd, self.Kdd = self._calc_cov(covfunc, covparams)
        
        # Evaluate the mean function
        if self.mean_func is None:
            self.mu_d = 0.
            self.mu_m = 0.
        else:
            self.mu_d = self.mean_func(self.xd, self.mean_params, **self.mean_kwargs)
            self.mu_m = self.mean_func(self.xm, self.mean_params, **self.mean_kwargs)
        
        # Calculate the cholesky of Kdd for later use
        self.L, self.w_md = self._calc_weights(self.Kdd, self.sd, self.Kmd)
            
        
    def prior(self, samples=1):
        """
        Placeholder
        """
        raise NotImplementedError
    
    def conditional(self, yd, samples=1):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def log_marg_likelihood(self, yd):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def update_xm(self, xm):
        """Update the output locations and the covariance kernel"""

        # print('Updating xm')

        self.M, _ = xm.shape

        self.M = self.M*self.P
        
        self.xm = xm
        self.Kmd = self.covfunc(self.xm, self.xd.T, self.covparams, **self.mean_kwargs) 
        
        # print('Updated xm')

    def __call__(self, yd):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def _calc_cov(self, covfunc, covparams):
        """
        Placeholder
        """
        raise NotImplementedError
    
    def _calc_weights(self, Kdd, sd, Kmd):
        """
        Placeholder
        """
        raise NotImplementedError
        
    def _calc_err(self, diag=True):
        """
        Placeholder
        """
        raise NotImplementedError
