import warnings
from copy import copy
from operator import itemgetter

import numpy as np
from numpy.random.mtrand import RandomState
from numpy import ndarray
from scipy.optimize import minimize


class GaussianProcessRegressor:
    """
    Gaussian process regression (GPR)
    A sklearn style GPR instance with RBF kernel
    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    Parameters
    ----------
    distance: str, ['euclidean', 'precomputed'] default='precomputed'
        The distance of RBF kernel specifying the covariance function of the GP.
    distance_matrix: ndarray
        The precomputed distance matrix
    length_scale: float or ndarray
        The length scale of the kernel.
    length_scale_bounds: tuple, a pair of float
        The lower and upper bound on 'length_scale'.
    n_restarts_optimiser: int, default=100
        The number of restarts of the optimiser for finding the kernel's
        parameters which maximize the log-marginal likelihood.
    random_seed: int or RandomState, default=None
        Determines random number generation used to initialize the centers.

    Attributes
    ----------
    X_train_: ndarray
        array-like of shape (n_samples, n_features)
        Feature vectors or other representations of training data
    y_train_: ndarray
        array-like of shape (n_samples,)
        Target values in training data

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <https://gaussianprocess.org/gpml/chapters/RW2.pdf>`
    """

    def __init__(self, distance='precomputed', distance_matrix=None,
                 length_scale=1, length_scale_bounds=(1e-4, 1e4),
                 constant=1, constant_bounds=(1e-4, 1e4),
                 noise=1e-3, random_seed=0, n_restarts_optimiser=100,
                 optimize='lbfgs'):

        self.k_matrix = distance_matrix
        self.distance = distance
        self._train_X, self._train_y = None, None
        self.length_scale = np.asarray(length_scale)
        self.length_scale_bounds = length_scale_bounds
        self.theta = np.asarray([constant])
        self.theta_bounds = constant_bounds
        self.noise = np.asarray([noise])
        self._random = RandomState(seed=random_seed)
        self.n_restarts_optimiser = n_restarts_optimiser
        self._is_fit = False
        self.optimize = optimize
        if distance is not 'euclidean':
            assert self.length_scale.shape[0] == distance_matrix.shape[0], \
                'The length scale dim ({}) must equal to the kernel ' \
                'matrix dim ({}).'.format(self.length_scale.shape[0],
                                          distance_matrix.shape[0])

    @property
    def X_train_(self):
        return self._train_X

    @property
    def y_train_(self):
        return self._train_y

    def fit(self, X, y):
        """
        Fit Gaussian process regression model

        Parameters
        ----------
        X: ndarray, shape (n_samples, n_features) or list of id
            Feature vectors or other representations of training data.

        y: ndarray, shape (n_samples,)
            Target values

        Returns
        -------
        self : object
            GaussianProcessRegressor class instance
        """
        # store train data
        self._train_X = np.asarray(X)
        self._train_y = np.asarray(y)
        # Optimise length scale and theta
        # The bounds includes length scale, theta
        init_params = np.hstack((self.length_scale, self.theta))
        bounds = [self.length_scale_bounds] * self.length_scale.shape[0] + \
                 [self.theta_bounds]
        # First opt starting from the specified point in kernel
        opt = [(self._optimisation(
            self._negative_log_marginal_likelihood,
            init_params,
            bounds
        ))]
        # Additional runs are performed from log-uniform chosen initial
        if self.n_restarts_optimiser > 0:
            for iteration in range(self.n_restarts_optimiser):
                # Random init the optimised parameters for each restart loop
                theta_init = self._random.uniform(
                    self.theta_bounds[0], self.theta_bounds[1]
                )
                length_scale_init = self._random.uniform(
                    self.length_scale_bounds[0], self.length_scale_bounds[1],
                    size=self.length_scale.size
                )
                init_params = np.hstack((length_scale_init, theta_init))
                opt.append(self._optimisation(
                    self._negative_log_marginal_likelihood, init_params, bounds,
                ))
        # Select result from run with minimal negative log-marginal likelihood
        opt_lml_values = list(map(itemgetter(1), opt))
        # Load theta and length scale from the minimum function
        # Shape of params: (length_scale*n, theta*1)
        self.theta = opt[np.argmin(opt_lml_values)][0][-1]
        self.length_scale = opt[np.argmin(opt_lml_values)][0][:-1]
        self._is_fit = True
        return self

    def predict(self, X, return_std=False, return_cov=False):
        """
        Predict using the Gaussian process regression model

        Parameters
        ----------
        X: ndarray
            Query points where the GP is evaluated.

        return_std: bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean: ndarray of shape (n_samples, )
            Mean of predictive distribution a query points

        y_std: ndarray of shape (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        y_cov: ndarray of shape (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """
        if not self._is_fit:
            print("GPR Model not fit yet.")
            return
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")
        """
        Args
        ----
        X: New input locations (n, d)
        self.train_X: Training locations (k, d)
        self.train_y: Training targets (k, 1)
        """
        X = np.asarray(X)
        # Equation 2.20 of GPML
        K = (self._kernels(self._train_X, self._train_X) +
             self.noise * np.eye(len(self._train_X)))  # (n, n)
        Ks = self._kernels(self._train_X, X)  # (n, k)
        Kss = self._kernels(X, X)  # (k, k)
        K_inv = np.linalg.inv(K)  # (n, n)

        # Equation 2.23 of GPML
        y_mean = Ks.T.dot(K_inv).dot(self._train_y).reshape(-1,)
        # Equation 2.24 of GPML
        y_cov = Kss - Ks.T.dot(K_inv).dot(Ks)
        if return_cov:
            return y_mean, y_cov
        elif return_std:
            y_var = copy(np.diag(y_cov))
            # Check if any of the variances is negative because of
            # numerical issues. If yes: set the variance to 0.
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                warnings.warn("Predicted variances smaller than 0. "
                              "Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean

    def _negative_log_marginal_likelihood(self, params):
        """
        Return log-marginal likelihood of theta, length scale and noise

        Parameters
        ----------
        params: list or ndarray
            Optimised kernel hyperparameters

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        """
        self.length_scale, self.theta = params[:-1], params[-1]
        # The covariance matrix for the noise targets, Equation 2.20
        Kyy = (self._kernels(self._train_X, self._train_X) +
               self.noise ** 2 * np.eye(len(self._train_X)))
        # Equation 5.8 from GPML
        loss = -0.5 * self._train_y.T.dot(np.linalg.inv(Kyy)).dot(self._train_y)
        loss -= 0.5 * np.linalg.slogdet(Kyy)[1]
        loss -= 0.5 * len(self._train_X) * np.log(2 * np.pi)
        return -loss.ravel()

    def _optimisation(self, obj_func, init_params, bounds):
        if self.optimize == 'lbfgs':
            opt_res = minimize(obj_func, x0=init_params,
                               bounds=bounds, method='L-BFGS-B')
            params_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimize):
            params_opt, func_min = self.optimize(obj_func, init_params, bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimize}.")

        return params_opt, func_min

    def _kernels(self, x1, x2):
        if self.distance == 'euclidean':
            # RBF kernel
            x1 = x1 / self.length_scale
            x2 = x2 / self.length_scale
            dists = (np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) -
                     2 * np.dot(x1, x2.T)) ** 2
        elif self.distance == 'precomputed':
            # Precomputed kernel
            l = self.length_scale.reshape((-1, 1, 1))
            dists = sum(self.k_matrix[:, x1.reshape(-1, ), :][:, :,
                        x2.reshape(-1, )] / l
                        ) ** 2
        else:
            raise RuntimeError('Only Euclidean and precomputed distance '
                               'are supported.')
        return self.theta ** 2 * np.exp(-0.5 * dists)
