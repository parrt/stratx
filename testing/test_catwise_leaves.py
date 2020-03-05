"""
MIT License

Copyright (c) 2019 Terence Parr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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


# def test_ignored_leaf():
#     assert 0==1


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


def test_two_leaves_with_2nd_ignored():
    np.random.seed(999)
    df = pd.DataFrame([
        # x1, x2, y         stratify x1, consider y ~ x2
         [1,  3,  5],
         [1,  4,  6],
         [2,  4,  7],
         [2,  4,  8]
    ], columns=['x1','x2','y'])
    X = df.drop('y', axis=1)
    y = df['y']

    leaves = get_leaves(X, y, 'x2') # get index of samples in each leaf
    expected_leaves = [np.array([0, 1]),  # leaf 0
                       np.array([2, 3])]  # leaf 1
    np.testing.assert_array_equal(leaves, expected_leaves)

    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="x2", min_samples_leaf=2)
    expected_leaf_deltas = np.array([[nan],    # cat 0
                                     [nan],    # 1
                                     [nan],    # 2
                                     [-1],     # 3
                                     [0]])     # 4
    expected_leaf_counts = np.array([[0],
                                     [0],
                                     [0],
                                     [1],
                                     [1]])
    expected_refcats = np.array([4])
    np.testing.assert_array_almost_equal(leaf_deltas, expected_leaf_deltas, decimal=1)
    np.testing.assert_array_equal(leaf_counts, expected_leaf_counts)
    np.testing.assert_array_equal(refcats, expected_refcats)
    assert ignored==2


def test_three_leaves_no_overlap():
    np.random.seed(999)
    df = pd.DataFrame([
        # x1, x2, y         stratify x1, consider y ~ x2
         [1,  2,  9],
         [1,  3,  7],
         [3,  4,  6],
         [3,  5,  5],
         [4,  6,  4],
         [4,  7,  3]
    ], columns=['x1','x2','y'])
    X = df.drop('y', axis=1)
    y = df['y']

    leaves = get_leaves(X, y, 'x2', min_samples_leaf=2) # get index of samples in each leaf
    expected_leaves = [np.array([0, 1]),  # leaf 0
                       np.array([2, 3]),  # leaf 1
                       np.array([4, 5])]  # leaf 2
    np.testing.assert_array_equal(leaves, expected_leaves)

    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="x2", min_samples_leaf=2)
    print(leaf_deltas, leaf_counts, refcats)
    expected_leaf_deltas = np.array([[nan,  nan,  nan],    # cat 0
                                     [nan,  nan,  nan],    # cat 2
                                     [2,    nan,  nan],    # cat 3
                                     [0,    nan,  nan],
                                     [nan,    0,  nan],
                                     [nan,   -1,  nan],
                                     [nan,  nan,  1],
                                     [nan,  nan,  0]])
    expected_leaf_counts = np.array([[0,   0,   0],
                                     [0,   0,   0],
                                     [1,   0,   0],
                                     [1,   0,   0],
                                     [0,   1,   0],
                                     [0,   1,   0],
                                     [0,   0,   1],
                                     [0,   0,   1]])
    expected_refcats = np.array([3, 4, 7])
    np.testing.assert_array_almost_equal(leaf_deltas, expected_leaf_deltas, decimal=1)
    np.testing.assert_array_equal(leaf_counts, expected_leaf_counts)
    np.testing.assert_array_equal(refcats, expected_refcats)
    assert ignored==0
