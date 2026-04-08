import numpy as np
from scipy.sparse.linalg import LinearOperator
from .vecchia_utils import find_nn, forward_solve_sp, backward_solve
from .vecchia import L_matrix, vecchia_llik

class VecchiaOperator:
    """
    General Vecchia covariance operator:
        C ≈ L^{-1} D L^{-T}

    Provides:
        matvec(v)   -> C v
        solve(v)    -> C^{-1} v
        logdet()
        loglik(y)

    Can be used for:
        B operator (latent state GP)
        R operator (noise GP)
    """

    def __init__(self, X, covfunc, covparams, noise=0.0,
                 nnum=30, nn_array=None,
                 order_func=None, order_params=None, order_kwargs=None,
                 scale_coords=False, rotate_coords=False, cov_kwargs=None,
                 jitter=1e-8, verbose=False):
        """
        X : (N,D) coordinates
        covfunc : kernel
        covparams : kernel hyperparameters
        noise : nugget (for R or diagonal regularization)
        nnum : number of neighbours
        """
        self.X = X
        self.covfunc = covfunc
        self.covparams = np.array(covparams)
        self.noise = noise
        self.nnum = nnum
        self.verbose = verbose
        self.jitter = jitter

        # ordering
        if order_func is not None:
            self.ord = order_func(order_params, **(order_kwargs or {}))
        else:
            self.ord = np.arange(X.shape[0])

        self.orig_ord = np.argsort(self.ord)
        self.Xo = self.X[self.ord]

        # neighbors
        if nn_array is None:
            self.nn_array = find_nn(self.Xo, nnum)
        else:
            self.nn_array = nn_array

        # build Vecchia factor
        self._build_factor()

    # -------------------------
    # rebuild when hyperparams change
    # -------------------------
    def update_hyperparams(self, covparams=None, noise=None):
        if covparams is not None:
            self.covparams = np.array(covparams)
        if noise is not None:
            self.noise = noise
        self._build_factor()

    # -------------------------
    # build L factor
    # -------------------------
    def _build_factor(self):
        if self.verbose:
            print("Building Vecchia factor...")
        self.L = L_matrix(self.Xo,
                          self.nn_array,
                          self.covfunc,
                          self.covparams,
                          self.noise + self.jitter)

        # precompute logdet contribution
        self._logdet = self._compute_logdet()

    def _compute_logdet(self):
        # In Vecchia, determinant = product of conditional variances
        # which are encoded in diagonal entries of local Cholesky
        # You already compute this in vecchia_llik; reuse logic:
        logdet = 0.0
        n = self.Xo.shape[0]
        for i in range(n):
            idx = self.nn_array[i]
            idx = idx[idx >= 0][::-1]
            xi = self.Xo[idx]
            Ki = K_matrix(xi, xi, self.covfunc, self.covparams, self.noise + self.jitter)
            Li = np.linalg.cholesky(Ki)
            logdet += 2*np.log(np.abs(Li[-1, -1]))
        return logdet

    # -------------------------
    # linear algebra
    # -------------------------
    def solve(self, v):
        """
        Approximate C^{-1} v
        """
        v = v[self.ord]
        y = forward_solve_sp(self.L, self.nn_array, v)
        x = backward_solve(self.L.T, y)
        return x[self.orig_ord]

    def matvec(self, v):
        """
        Approximate C v using:
            C ≈ L^{-1} D L^{-T}
        Implemented as two triangular solves.
        """
        v = v[self.ord]
        z = backward_solve(self.L.T, v)
        x = forward_solve_sp(self.L, self.nn_array, z)
        return x[self.orig_ord]

    # -------------------------
    # likelihood
    # -------------------------
    def loglik(self, y, mu=None):
        """
        log N(y | mu, C)
        """
        if mu is None:
            mu = np.zeros_like(y)
        y = y[self.ord]
        mu = mu[self.ord]
        return vecchia_llik(self.Xo, y, mu,
                            self.nn_array,
                            self.covfunc,
                            self.covparams,
                            self.noise)

    def logdet(self):
        return self._logdet

    # -------------------------
    # LinearOperator wrapper
    # -------------------------
    def as_linear_operator(self):
        n = self.X.shape[0]
        return LinearOperator((n, n), matvec=self.matvec)
