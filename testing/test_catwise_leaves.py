from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from stratx.featimp import *
from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import nan

from test_catmerge import stratify_cats, get_leaves

def test_single_leaf():
    np.random.seed(999)
    df = pd.DataFrame([
        # x1, x2, y         stratify x1, consider y ~ x2
         [1,  3,  5],
         [1,  4,  6],
         [1,  4,  5],
         [1,  2,  4]
    ], columns=['x1','x2','y'])
    X = df.drop('y', axis=1)
    y = df['y']

    leaves = get_leaves(X, y, 'x2') # get index of samples in each leaf
    expected_leaves = [np.array([0, 1, 2, 3])]  # leaf 0
    np.testing.assert_array_equal(leaves, expected_leaves)

    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="x2", min_samples_leaf=4)

    expected_leaf_deltas = np.array([nan, nan, -1, 0, .5]).reshape(-1,1)
    expected_leaf_counts = np.array([0, 0, 1, 1, 2]).reshape(-1,1)
    expected_refcats = np.array([3])
    np.testing.assert_array_almost_equal(leaf_deltas, expected_leaf_deltas, decimal=1)
    np.testing.assert_array_equal(leaf_counts, expected_leaf_counts)
    np.testing.assert_array_equal(refcats, expected_refcats)
    assert ignored==0


def test_two_leaves():
    np.random.seed(999)
    df = pd.DataFrame([
        # x1, x2, y         stratify x1, consider y ~ x2
         [1,  3,  5],
         [2,  4,  6],
         [1,  4,  5],
         [2,  2,  4]
    ], columns=['x1','x2','y'])
    X = df.drop('y', axis=1)
    y = df['y']

    leaves = get_leaves(X, y, 'x2') # get index of samples in each leaf
    expected_leaves = [np.array([0, 2]),  # leaf 0
                       np.array([1, 3])]  # leaf 1
    np.testing.assert_array_equal(leaves, expected_leaves)

    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="x2", min_samples_leaf=2)
    expected_leaf_deltas = np.array([[nan, nan],    # 0
                                     [nan, nan],    # 1
                                     [nan, 0],      # 2
                                     [0,   nan],    # 3
                                     [0,   2]])     # 4
    expected_leaf_counts = np.array([[0,   0],
                                     [0,   0],
                                     [0,   1],
                                     [1,   0],
                                     [1,   1]])
    expected_refcats = np.array([4, 2])
    np.testing.assert_array_almost_equal(leaf_deltas, expected_leaf_deltas, decimal=1)
    np.testing.assert_array_equal(leaf_counts, expected_leaf_counts)
    np.testing.assert_array_equal(refcats, expected_refcats)
    assert ignored==0

