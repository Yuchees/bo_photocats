#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main class for Bayes optimisation. Coding structure is inspired by skopt.
"""
__author__ = 'Yu Che'

import numpy as np
import copy
from copy import deepcopy
from numpy import ndarray

from .util import remove_same_points, is_2d, is_same_dim
from .acquisition import acquisition_function

from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class BayesOptimizer(object):
    """
    Run Bayes optimization loop

    Parameters
        ----------
        bounds: ndarray
            Searching space boundaries
        base_estimator: GaussianProcessRegressor
            Scikit learn GPs, need return_std=True
        random_seed: int
            Random seed for picking start points when using L-BFGS-B
            optimizer method
        acq_func: str
            The name of acquisition function.
            Supporting: 'EI', 'PI', 'UCB' and 'LCB'
        optimizer: str
            Function to minimize over the posterior distribution
            Supporting: 'gradient' and 'sampling'
        n_restarts: int
            Number of evaluations of aca_func with initialization points
        sampling: ndarray, optional
            List of sampled points if using sampling optimizer
        acq_func_kwargs: dict, optional
            Additional arguments to be passed to the acq_func
    """
    def __init__(self, bounds, base_estimator, random_seed=0, acq_func='EI',
                 optimizer='gradient', n_restarts=10, sampling=np.array([]),
                 acq_func_kwargs=None):
        self.__bounds = bounds
        self.base_estimator = base_estimator
        self.random = np.random.RandomState(random_seed)
        self.acq_func = acq_func
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        if optimizer not in ['sampling', 'gradient']:
            raise ValueError('Excepted optimizer to be "sampling" '
                             'or "gradient", got {}.'.format(optimizer))
        allowed_acq_func = ['EI', 'PI', 'UCB', 'LCB']
        if acq_func not in allowed_acq_func:
            raise ValueError('Excepted acquisition function to be in {}, '
                             'got {}.'.format(', '.join(allowed_acq_func),
                                              acq_func))
        if n_restarts < 0 or not isinstance(n_restarts, int):
            raise ValueError('Expected n_restarts is an int and >=0, '
                             'got {}.'.format(n_restarts))
        self.sampling = sampling
        self._acq_func_kwargs = acq_func_kwargs
        self._xi = None
        self._yi = None
        self._steps_index = []
        self._y_opt = None
        self._gpr = []

    def __str__(self):
        str_sum = '{}, measured {} points.'.format(self._gpr, len(self.xi))
        return str_sum

    __repr__ = __str__

    @property
    def xi(self):
        """
        Get the xi used by Bayes Optimizer

        Returns
        -------
        xi: ndarray
            List of points
        """
        return self._xi

    @property
    def yi(self):
        """
        Get the yi used by Bayes Optimizer

        Returns
        -------
        yi: ndarray
            List of values at xi
        """
        return self._yi

    @property
    def y_opt(self):
        return self._y_opt

    @property
    def bounds(self):
        return self.__bounds

    @property
    def acq_func_param(self):
        return self._acq_func_kwargs

    @property
    def gpr(self):
        return self._gpr

    @property
    def max_step(self):
        return max(self._steps_index)

    def ask(self, num_samples=1):
        """
        Query one point at which objective should be evaluated.

        Returns
        -------
        min_x: ndarray or list
            The suggested point
        """
        assert isinstance(self._xi, ndarray), \
            'None estimated points are given, run tell() first.'
        dim = self._xi.shape[1]
        min_val = min_x = None
        # Get the best mean from known samples
        self._y_opt = np.max(self.base_estimator.predict(self.xi))
        if self.optimizer == 'sampling':
            # Removing the measured points in sampling
            checked_samples = remove_same_points(self.sampling, self.xi)
            # Acquisition function minimization
            min_x = self._find_next_samples(
                checked_samples=checked_samples,
                num_samples=num_samples
            )
        elif self.optimizer == 'gradient':

            def min_obj(X):
                # Minimization objective is the negative acquisition function
                return -acquisition_function(
                    X=X.reshape(-1, dim),
                    y_opt=self.y_opt,
                    gpr=self.base_estimator,
                    acq_func=self.acq_func,
                    acq_func_kwargs=self._acq_func_kwargs
                )

            # Find the best optimum by starting from n_restart different
            # random points.
            for x0 in self.random.uniform(self.bounds[:, 0], self.bounds[:, 1],
                                          size=(self.n_restarts, dim)):
                res = minimize(fun=min_obj, x0=x0,
                               bounds=self.bounds, method='L-BFGS-B')
                if res.fun < min_val:
                    min_val = res.fun[0]
                    min_x = res.x
        else:
            raise ValueError('')
        return min_x  # .reshape(1, -1)

    def _acq_values(self, acq_func='EI', num_step=0,
                    acq_func_kwargs=None, sampling=None):
        """
        Calculate the acquisition function values at each sampling points.

        Parameters
        ----------
        acq_func: str
            The name of acquisition function
        num_step: int
            The step number of different BO steps
        acq_func_kwargs: dict
            The fixed acquisition function parameters
        sampling: ndarray
            The coordination of sampling points

        Returns
        -------
        values: ndarray
            Calculated acq_values
        """
        base_estimator = self._gpr[num_step]
        if sampling is None:
            x = self.sampling
        else:
            x = sampling
        values = acquisition_function(X=x, y_opt=self.y_opt,
                                      gpr=base_estimator, acq_func=acq_func,
                                      acq_func_kwargs=acq_func_kwargs)
        return values

    def _find_next_samples(self, checked_samples, num_samples):
        """
        Function to minimise over the posterior distribution.

        Parameters
        ----------
        checked_samples: ndarray
            The coordination of sampling points
        num_samples: int
            The number of picked samples

        Returns
        -------
        list: The list of suggested points
        """
        acq_values = -acquisition_function(
            X=checked_samples,
            y_opt=self.y_opt,
            gpr=self.base_estimator,
            acq_func=self.acq_func,
            acq_func_kwargs=self._acq_func_kwargs
        )
        # Find the minimizing point
        min_x = []
        sorted_values = copy.copy(acq_values)
        sorted_values.sort()
        # TODO: Using np.argwhere() to find the index
        for value in sorted_values[:num_samples]:
            picked_x = checked_samples[
                np.argwhere(acq_values == value).reshape(-1,)
            ]
            min_x.append(picked_x)
        min_x.reverse()
        return min_x

    def parallel_ask(self, acq_func_args, num_samples=1):
        """
        Query several points at which objective should be evaluated. Points are
        suggested by different acq_func arguments. Arguments need to be a list
        in this dictionary.

        Parameters
        ----------
        acq_func_args: dict
            Additional arguments to be passed to the acq_func. To get
            parallel suggestion, the given acq_func_args must have
            different values in a list.

        num_samples: int
            The number of suggested samples in each parameter.

        Returns
        -------
        next_points: list
            List of suggested points
        """
        next_points = []
        # Get different parameters
        for value in list(acq_func_args.values())[0]:
            if 'xi' in acq_func_args.keys():
                self._acq_func_kwargs = {'xi': value}
            elif 'kappa' in acq_func_args.keys():
                self._acq_func_kwargs = {'kappa': value}
            suggested_points = self.ask(num_samples=num_samples)
            # Remove suggested points to avoid repeating
            self.sampling = remove_same_points(
                self.sampling, np.array(suggested_points).reshape(
                    -1, self.sampling.shape[1]
                )
            )
            next_points.append(suggested_points)
        return next_points

    def tell(self, x, y):
        """
        Record evaluated points of the objective function.

        Parameters
        ----------
        x: ndarray
            Points at which objective was evaluated
        y: ndarray
            Values of objective at x
        """
        # Data structure detect
        assert is_2d(x) and is_2d(y), \
            'Given x and y must be a 2D array, ' \
            'use reshape(1, -1) for one point.'
        assert x.shape[0] == y.shape[0], \
            'The given sample x {} and y {} is not compatible.'.format(
                x.shape,
                y.shape
            )
        # Step0
        if self._xi is None:
            self._xi = x
            self._yi = y
            self._steps_index += [0] * x.shape[0]
        else:
            # Add new evaluated points to previous points after step0
            assert is_same_dim(self._xi, x), \
                'The given point dimension {} is different with ' \
                'previous point {}'.format(x.shape[1], self._xi[1])
            self._xi = np.vstack((self._xi, x))
            self._yi = np.vstack((self._yi, y))
            self._steps_index += [max(self._steps_index) + 1] * x.shape[0]
        # Update the Gaussian process with existing points
        self.base_estimator.fit(self._xi, self._yi)
        self._gpr.append(deepcopy(self.base_estimator))

    def get_step_xy(self, num_of_step, return_y=False):
        """
        Query the measured x in a step of optimizer

        Parameters
        ----------
        num_of_step: int
            The number of queried step
        return_y: bool, optional
            If True, the yi array is returned along with xi

        Returns
        -------
        xi_step: ndarray
            The x in queried step
        yi_step: ndarray, optional
            The y in queried step.
            Only returned when return_y is True.
        """
        assert len(self._steps_index) > 0, \
            'The optimizer have not run step0.'
        assert num_of_step <= self.max_step, \
            'The optimizer have not run step {}, maximum step is {}'.format(
                num_of_step, self.max_step
            )
        xi_index = np.array(self._steps_index)
        xi_step = self._xi[np.argwhere(xi_index == num_of_step)].reshape(
            -1, self.xi.shape[1]
        )
        if return_y:
            yi_step = self._yi[np.argwhere(xi_index == num_of_step)].reshape(
                -1, self.yi.shape[1]
            )
            return xi_step, yi_step
        return xi_step


def gpr_matern_kernel(param):
    """
    Gaussian process regressor with constant Matern kernel.

    Parameters
    ----------
    param: dict
        Hyper parameters

    Returns
    -------
    gpr: GaussianProcessRegressor
        A GPs object
    """
    kernel = ConstantKernel(
        constant_value=param['constant'],
        constant_value_bounds=param['constant_bounds']
    ) * Matern(
        length_scale=param['length_scale'],
        length_scale_bounds=param['length_scale_bounds'],
        nu=param['nu']
    )
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=param['alpha'],
                                   normalize_y=True)
    return gpr
