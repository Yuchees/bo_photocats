#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Yu Che'

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pandas.core.frame import DataFrame
from numpy import ndarray


def is_same_dim(array_0, array_1):
    """
    Check if points in array0 and array1 have same dimensions

    Parameters
    ----------
    array_0: ndarray
        The first array for testing
    array_1: ndarray
        The second array for comparing

    Returns
    -------
    bool
    """
    assert is_2d(array_0) and is_2d(array_1), ''
    return array_0.shape[1] == array_1.shape[1]


def is_2d(array0):
    """
    Check if the array0 is 2D

    Parameters
    ----------
    array0: ndarray
        The given array for testing

    Returns
    -------
    bool
    """
    return len(array0.shape) == 2


# noinspection PyArgumentList
def find_same_points(array0, array1):
    """
    Finding the points position of array0 if they exist in array1

    Parameters
    ----------
    array0: ndarray
        The first array for testing
    array1: ndarray
        The second array for comparing

    Returns
    -------
    sample_idx: list
        The position in array0
    """
    assert is_same_dim(array0, array1), 'Given array must have same dim.'
    sample_index = []
    for array in array1:
        # Calculate the distance matrix between selected array and array0
        distance_array = cdist(array0, array.reshape(1, -1))
        if distance_array.min() < 1e-10:
            # Find the closest point
            array_index = np.argwhere(distance_array == distance_array.min())
            sample_index.append(array_index[0, 0])
    return sample_index


def remove_same_points(array0, array1):
    """
    Removing points in array0 if they exist or close to any point in array1

    Parameters
    ----------
    array0: ndarray
        The first array for testing
    array1: ndarray
        The second array for comparing

    Returns
    -------
    array0: ndarray
        The modified array0 removed points
    """
    assert (is_2d(array0) and is_2d(array1)), \
        'The given array has different dimensions, got {} and {}.'.format(
            array0.shape, array1.shape
        )
    if array1 is None:
        return array0
    else:
        repeated_points = find_same_points(array0, array1)
        removed_samples = np.delete(array0, repeated_points, axis=0)
        return removed_samples


def kwargs_generator(mean, size, name='kappa'):
    """
    Acquisition function parameters generator
    Using exponential distribution to pick parameters randomly

    Parameters
    ----------
    mean: float
        The mean of the distribution
    size: int
        The number of generated parameters
    name: {'kappa', 'xi'}, optional
        The name of acquisition function parameters
    Returns
    -------
    dict
        A dictionary contains the parameter list
    """
    params = np.random.exponential(scale=mean, size=size)
    params.sort()
    params = params.tolist()
    return {name: params}


def load_exp_data(exp_df_path, sheet_name, samples, return_df=False):
    """
    Loading experiments data and convert to array for next step of optimisation

    Parameters
    ----------
    exp_df_path: str
        Excel file path
    sheet_name: str
        String used for requesting sheet
    samples: ndarray
        The matrix for picking data points
    return_df: bool
        If True, the full experiment DataFrame is returned along with the x, y

    Returns
    -------
    x : ndarray
        Experiment samples
    y : ndarray
        Measured yield
    exp_df : DataFrame, optional
        Full experiment DataFrame.
        Only returned when return_df is True.
    """
    exp_df = pd.read_excel(
        exp_df_path, index_col=0, sheet_name=sheet_name
    ).loc[:, ['name', 'smiles', 'yield']]
    exp_index = exp_df.index.tolist()
    x = samples[exp_index]
    y = (exp_df.loc[:, 'yield'].values/100).reshape(-1, 1)
    if return_df:
        return x, y, exp_df
    else:
        return x, y


def get_next_df(suggested_x, parallel_param, samples, df):
    """
    Get the DataFrame for the suggested samples

    Parameters
    ----------
    suggested_x: ndarray
        The suggested points from optimiser
    parallel_param: dict
        The parameters of parallel suggestion
    samples: ndarray
        The full samples matrix
    df: DataFrame
        The full samples DataFrame including smiles and names
    Returns
    -------
    DataFrame
    The DataFrame of suggested points
    """
    suggested_x = np.array(suggested_x).reshape(-1, 5)
    next_x_id = find_same_points(samples, suggested_x)
    next_df = df.loc[next_x_id, : 'SMILE']
    next_df = next_df.reindex(columns=['name', 'SMILE', 'kappa'])
    next_df.kappa = parallel_param['kappa']
    return next_df


def summary_df(exp_df_file, drop_columns):
    """
    Concatenate experimental tables along and addling steps labels

    Parameters
    ----------
    exp_df_file: str
        The path of Excel files
    drop_columns: list
        List of columns to be dropped
    Returns
    -------
    df: DataFrame
        Concatenated Dataframe
    """
    xls = pd.ExcelFile(exp_df_file)
    df = None
    for number, sheet in enumerate(xls.sheet_names):
        _df = pd.read_excel(xls, sheet, index_col=0).drop(columns=drop_columns)
        _df.loc[:, 'step'] = number
        if df is None:
            df = _df
        else:
            df = pd.concat([df, _df])
    return df


if __name__ == '__main__':
    x0 = np.load('../../data/test_file/samples_x.npy')
    x1 = np.load('../../data/test_file/picked_x.npy')
    result = find_same_points(x0, x1)
    print(result)
