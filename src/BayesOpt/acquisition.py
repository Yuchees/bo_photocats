#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The acquisition function, implementing from PMM
"""
__author__ = 'Yu Che'

import warnings
import numpy as np
from numpy import ndarray
from scipy.stats import norm


def acquisition_function(X, gpr, acq_func, y_opt=0, acq_func_kwargs=None):
    """
    Computes the acquisition function at points X based on existing gpr model.

    Parameters
    ----------
    X: ndarray
        Points at which acquisition shall be computed (m x d).
    gpr: GaussianProcessRegressor
        A GaussianProcessRegressor fitted to samples.
    acq_func: {'EI', 'PI', 'UCB', 'LCB'}, optional
        Acquisition function name.
    y_opt: float
        Optimized y vales, usually use np.max(y)
    acq_func_kwargs: dict
        Parameters for different acquisition function

    Returns
    -------
    acq_values: ndarray
        Acquisition values at points X
    """
    xi = kappa = 0
    if acq_func_kwargs is None:
        acq_func_kwargs = {}
        xi = acq_func_kwargs.get('xi', 0.01)
        kappa = acq_func_kwargs.get('kappa', 1.96)
    else:
        if acq_func in ['EI', 'PI']:
            xi = acq_func_kwargs.get('xi')
        elif acq_func in ['UCB', 'LCB']:
            kappa = acq_func_kwargs.get('kappa')
    mean, std = gpr.predict(X, return_std=True)
    mean = mean.reshape(-1,)
    if acq_func == 'EI':
        return __expected_improvement(y_opt, mean, std, xi)
    elif acq_func == 'PI':
        return __probability_improvement(y_opt, mean, std, xi)
    elif acq_func == 'UCB':
        return __upper_confidence_bound(mean, std, kappa)
    elif acq_func == 'LCB':
        return __lower_confidence_bound(mean, std, kappa)
    else:
        raise ValueError('Excepted acquisition function to be "EI", "PI", "UCB"'
                         'or "LCB", got {}'.format(acq_func))


def __expected_improvement(y_opt, mean, std, xi):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = mean - y_opt - xi
        z = imp / std
        ei = imp * norm.cdf(z) + std * norm.pdf(z)
        ei[std == 0.0] = 0.0
    return ei


def __probability_improvement(y_opt, mean, std, xi):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = (mean - y_opt - xi)/std
        pi = norm.cdf(z)
    return pi


def __upper_confidence_bound(mean, std, kappa):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ucb = mean + kappa * std
    return ucb


def __lower_confidence_bound(mean, std, kappa):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lcb = mean - kappa * std
    return lcb


if __name__ == '__main__':
    from sklearn.gaussian_process import GaussianProcessRegressor
    kernel = GaussianProcessRegressor()
    X_init = np.array([
        [1, 0],
        [2, 1.5],
        [3, 2.1]
    ])
    y_init = np.array([
        [1.3],
        [4],
        [1]
    ])
    x_sample = np.array([
        [1.5, 0.4],
        [1.9, 0.2]
    ])
    kernel.fit(X_init, y_init)
    ac = acquisition_function(X=x_sample, y_opt=np.max(y_init).reshape(-1, 1),
                              gpr=kernel,
                              acq_func='EI')
    print(ac)
