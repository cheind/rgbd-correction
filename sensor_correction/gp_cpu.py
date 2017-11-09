__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.externals import joblib

class GPRegressor:
    '''Gaussian Process regressor on CPU.

    Takes input feature vectors X and target regression values Y and fits a Gaussian process,
    that can be used to predict for query points X*. This implementation uses a squared
    exponential kernel for determining the similarity between feature vectors. Tuning of
    hyper-parameters is supported by optimizing the negative log-marginal-likelihood through
    sklearn provided code.
    '''

    def fit(self, X, Y, length_scale=1.0, signal_std=1.0, noise_std=1e-10, normalize=False, optimize=False, repeat=0):
        '''Fit a Gaussian Process regressor.
        
        Params
        ------
        X : mx4 array
            Training feature vectors
        Y : mx1 array
            Target values
        
        Kwargs
        ------
        length_scale : scalar or 4x1 array, optional
            Kernel length scaling input feature dimensions
        signal_std : scalar, optional
            Signal sigma
        noise_std : scalar, optional
            Observation noise sigma
        normalize : bool, optional
            Whether or not to normalize Y by mean adjustment
        optimize : bool or list
            Turn on/off optimization. If list, only the parameters in list will be tuned.
        '''
        
        optimizer = 'fmin_l_bfgs_b'
        bounds_ls = bounds_ss = bounds_ns = (1e-3, 1e3)

        signal_var = signal_std**2
        noise_var = noise_std**2

        if isinstance(optimize, list):
            bounds_ls = (1e-3, 1e3) if 'length_scale' in optimize else (length_scale, length_scale)
            bounds_ss = (1e-3, 1e3) if 'signal_std' in optimize else (signal_var, signal_var)
            bounds_ns = (1e-3, 1e3) if 'noise_std' in optimize else (noise_var, noise_var)
        elif not optimize:
            optimizer = None

        kernel = ConstantKernel(signal_var, bounds_ss) * RBF(length_scale, bounds_ls) + WhiteKernel(noise_var, bounds_ns)

        self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=normalize, optimizer=optimizer, n_restarts_optimizer=repeat)
        self.gpr.fit(X, Y)

    def predict(self, X, return_std=False):
        '''Predict values.

        Params
        ------
        X : nx4 array
            Input feature vectors.
        
        Kwargs
        ------
        return_std : bool, optional
            If true, returns the uncertainty variances for query points. Useful for
            computing confidence values.

        Returns
        -------
        Y : nx1 array
            Predictions
        K : nx1 array
            Variances for query points. Only if return_std = true
        '''
        return self.gpr.predict(X, return_std=return_std)

    @property
    def length_scale(self):
        return self.gpr.kernel_.k1.k2.length_scale

    @property
    def signal_std(self):
        return np.sqrt(self.gpr.kernel_.k1.k1.constant_value)

    @property
    def noise_std(self):
        return np.sqrt(self.gpr.kernel_.k2.noise_level)

    def save(self, fname):
        joblib.dump(self.gpr, fname) 

    def load(self, fname):
        self.gpr = joblib.load(fname) 



class GPRegressorStandalone:
    '''Standalone Gaussian Process regressor.'''

    def fit(self, X, Y, W, signal_std=1.0, noise_std=1e-10, normalize=False):

        self.noise_std = noise_std
        self.signal_std = signal_std        
        self.W = W
        self.X = X

        if normalize:
            self.ymean = np.mean(Y)
            Y = Y - self.ymean
        else:
            self.ymean = np.zeros(1)

        self.K = GPRegressorStandalone.kernel(X, X, self.W, self.signal_std) + np.eye(X.shape[0]) * self.noise_std
        self.L = np.linalg.cholesky(self.K)

        self.Li = stri(self.L.T, np.eye(self.L.shape[0]))
        self.Ki = self.Li.dot(self.Li.T)
        self.alpha = stri(self.L.T, stri(self.L, Y, check_finite=False, lower=True))
    
    def predict(self, X, return_std=False):
        Ks = GPRegressorStandalone.kernel(self.X, X, self.W, self.signal_std)        
        pred = Ks.T.dot(self.alpha) # Zero mean
        pred += self.ymean

        if return_std:
            Kss = GPRegressorStandalone.kernel(X, X, self.W, self.signal_std)
            sigma = np.copy(np.diag(Kss))
            sigma -= np.einsum("ij,ij->i", np.dot(Ks.T, self.Ki), Ks.T)
            sigma[sigma < 0.] = 0.
            sigma = np.sqrt(sigma)
            return pred, sigma
        else:
            return pred

    @staticmethod
    def dist(A, B, W):
        '''Pairwise squared weighted distance.'''
        diff = A[np.newaxis, :, :] - B[:, np.newaxis, :]
        d = np.einsum('jil,jil->ij', np.tensordot(diff, W, axes=(2,0)), diff)
        return d

    @staticmethod
    def kernel(A, B, W, signal_std=1.):
        '''Squared exponential covariance function.'''
        d = GPRegressorStandalone.dist(A, B, W)
        return signal_std**2 * np.exp(-0.5 * d)
