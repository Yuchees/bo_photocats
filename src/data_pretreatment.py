#!/usr/bin/env python3
"""
Data pretreatment and calculation functions for SMP
"""
import numpy as np
import warnings
from numpy import ndarray
from pandas.core.frame import DataFrame
from datetime import datetime

from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols

from sklearn import manifold
from sklearn.metrics.pairwise import euclidean_distances


def cal_fingerprints(df, fingerprint='Morgan_fingerprints'):
    """
    Calculate fingerprints using RDKit function and add a new column in the
    given DataFrame.

    Parameters
    ----------
    df: DataFrame
        Origin DataFrame including SMILES
    fingerprint: str
        Name of the selected fingerprints
    Returns
    -------
    None
    """
    print('Start fingerprints calculation...')
    start = datetime.now()
    df[fingerprint] = None
    df['info'] = None
    info = {}
    df[fingerprint] = df[fingerprint].astype('object')
    for i in df.index:
        mol = Chem.MolFromSmiles(df.loc[i, 'smiles'])
        if fingerprint == 'Morgan_fingerprints':
            df[fingerprint] = df[fingerprint].astype('object')
            fps = AllChem.GetMorganFingerprint(mol, radius=5, bitInfo=info)
            df.loc[i, 'info'] = str(info)
        elif fingerprint == 'Morgan_fingerprints_bit':
            fps = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=1024, bitInfo=info
            )
            df.loc[i, 'info'] = str(info)
        elif fingerprint == 'Topological_fingerprints':
            fps = FingerprintMols.FingerprintMol(mol)
        else:
            raise TypeError('Please write a correct fingerprint name.')
        df.loc[i, fingerprint] = fps
    print('Finished!\n'
          'Total time:{}'.format(datetime.now() - start))


def cal_euclidean_matrix(feature):
    """
    Calculate the Euclidean distance matrix as the similarity matrix
    with minmax normalisation.

    Parameters
    ----------
    feature: ndarray
        Selected features in matrix

    Returns
    -------
    ndarray
        The similarity matrix
    """
    distance_matrix = euclidean_distances(feature)
    similarity_matrix = distance_matrix / distance_matrix.max()
    return similarity_matrix


def cal_fps_similarity_matrix(df, fingerprint_name='Morgan_fingerprints'):
    """
    Calculate the similarity matrix by their fingerprints distance.

    Parameters
    ----------
    df: DataFrame
        DataFrame with fingerprints
    fingerprint_name: str
        The name of fingerprint column in dataFrame
    Returns
    -------
    ndarray
        The similarity matrix
    """
    print('Start similarity calculation...')
    start = datetime.now()
    similarity_matrix = np.zeros(shape=(len(df), len(df)))
    for i in df.index:
        for j in df.index:
            fps_i = df.loc[i, fingerprint_name]
            fps_j = df.loc[j, fingerprint_name]
            similarity = DataStructs.DiceSimilarity(fps_i, fps_j)
            similarity_matrix[i][j] = 1 - similarity
    print('Finished!\n'
          'Total time:{}'.format(datetime.now() - start))
    return similarity_matrix


def dimensionality_reduction(similarity_matrix, method='mds'):
    """
    Using non-linear dimensionality reduction method with the given
    similarity matrix to calculate 2D coordinators.

    Parameters
    ----------
    similarity_matrix: ndarray
        The calculated similarity matrix
    method: str
        The dimensionality reduction method
    Returns
    -------
    ndarray
        2D position matrix
    """
    print('Start {} calculation...'.format(method))
    start = datetime.now()
    seed = np.random.RandomState(seed=0)
    if method == 'mds':
        mds = manifold.MDS(
            n_components=2, max_iter=30000, random_state=seed,
            eps=1e-12, dissimilarity="precomputed"
        )
        pos = mds.fit_transform(similarity_matrix)
    elif method == 'tsne':
        tsne = manifold.TSNE(
            n_components=2, n_iter=30000, random_state=seed, perplexity=50,
            min_grad_norm=1e-12, metric='precomputed'
        )
        pos = tsne.fit_transform(similarity_matrix)
    elif method == 'isomap':
        isomap = manifold.Isomap(
            n_components=2, n_neighbors=12, max_iter=30000,
        )
        pos = isomap.fit_transform(similarity_matrix)
    else:
        raise ValueError('Please type correct method name.')
    print('Finished!\n'
          'Total time:{}'.format(datetime.now() - start))
    return pos


def ks_selection(distance_matrix, n_examples, init=None):
    """
    Representative subset selection using the Kennard-Stone algorithm to
    pick number of most different points.

    Parameters
    ----------
    distance_matrix: ndarray
        Precomputed similarity matrix
    n_examples: int
        The number of selected points
    init: None or list of index
        The initial selected points as the starting set

    Returns
    -------
    list
        The index of picked points
    """
    if not isinstance(n_examples, int):
        raise TypeError('The number of examples must be an integer.')
    elif n_examples < 3:
        raise ValueError('The number of examples must be '
                         'equal or greater than 2.')

    def get_max(matrix, pair=False):
        idx_max = np.argwhere(matrix == matrix.max())
        num_max = len(idx_max)
        if num_max > 2:
            warnings.warn('only the first one is picked '
                          'from {} maximum samples.'.format(num_max/2),
                          DeprecationWarning)
        if pair:
            return list(idx_max[0])
        else:
            return idx_max[0][0]
    if init is None:
        # Step 1: Select the farthest pairwise points as starting points
        subset = []
        subset += get_max(distance_matrix, pair=True)
    elif isinstance(init, list):
        subset = init
    else:
        raise TypeError('The init parameter must be a list of index.')
    # Step 2: Pick the next point
    for i in range(n_examples-2):
        # The distance matrix
        selected_dis_matrix = distance_matrix[subset, :]
        shortest_dis_matrix = np.min(selected_dis_matrix, axis=0)
        subset.append(get_max(shortest_dis_matrix))
    return subset
