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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

import stratx.partdep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import nan


def get_leaves(X, y, colname, min_samples_leaf=1):
    X_not_col = X.drop(colname, axis=1).values
    rf = RandomForestRegressor(n_estimators=1,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=False,
                               max_features=1.0,
                               oob_score=False)
    rf.fit(X_not_col, y)
    leaves = stratx.partdep.leaf_samples(rf, X_not_col)
    return leaves


def stratify_cats(X, y,
                  colname,  # X[colname] expected to be numeric codes
                  max_catcode=None,
                  # if we're bootstrapping, might see diff max's so normalize to one max
                  n_trees=1,
                  min_samples_leaf=10,
                  max_features=1.0,
                  bootstrap=False,
                  supervised=True,
                  verbose=False):
    X_not_col = X.drop(colname, axis=1).values
    X_col = X[colname].values
    if max_catcode is None:
        max_catcode = np.max(X_col)
    if supervised:
        rf = RandomForestRegressor(n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
        rf.fit(X_not_col, y)
        if verbose:
            print(f"CatStrat Partition RF: dropping {colname} training R^2 {rf.score(X_not_col, y):.2f}")
    else:
        print("USING UNSUPERVISED MODE")
        X_synth, y_synth = stratx.partdep.conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    min_samples_leaf=min_samples_leaf,# * 2,  # there are 2x as many samples (X,X') so must double leaf size
                                    bootstrap=bootstrap,
                                    max_features=max_features,
                                    oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    # rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X_not_col, y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    # leaf_deltas, leaf_counts, leaf_avgs, leaf_sizes, leaf_catcounts, ignored = \
    #     catwise_leaves(rf, X, y, colname, verbose=verbose)

    leaf_deltas, leaf_counts, ignored = \
        stratx.partdep.catwise_leaves(rf, X_not_col, X_col, y.values, max_catcode)
    print("leaf_deltas\n",leaf_deltas)
    print("leaf_counts\n",leaf_counts)

    return leaf_deltas, leaf_counts, ignored


def synthetic_poly_data(n=1000,p=3,dtype=float):
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = (np.random.random_sample(size=n) * 1000).astype(dtype)
    yintercept = 100
    df['y'] = np.sum(df, axis=1) + yintercept
    terms = [f"x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " where x_i ~ U(0,10)"
    return df, eqn


def toy_weather_data(n = 1000, p=50):
    """
    For each state, create a (fictional) ramp of data from day 1 to 365 so mean is not
    0, as we'd get from a sinusoid.
    """
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df_avgs = pd.read_csv("state_avgtemp.csv")
    print("avg of states' avg temps:",np.mean(df_avgs['avgtemp']))

    df = pd.DataFrame()
    df['dayofyear'] = np.random.randint(1, 365 + 1, size=n)
    df['state'] = np.random.randint(0, p, size=n) # get only p states
    df['temp'] = .1 * df['dayofyear'] + df_avgs['avgtemp'].iloc[df['state']].values
    return df.drop('temp', axis=1), df['temp'], df_avgs['state'].values, df_avgs.iloc[0:p]


def check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas, leaf_counts=None, marginal_avg_y_per_cat=None):
    if leaf_counts is None:
        leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    avg_per_cat, count_per_cat = \
        stratx.partdep.avg_values_at_cat(leaf_deltas, leaf_counts, marginal_avg_y_per_cat=marginal_avg_y_per_cat)


    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)


def test_single_leaf():
    set_random_seed(999)
    leaf_deltas = np.array([
        [0],
        [1],
        [2],
        [nan],
        [0]
    ])
    leaf_counts = np.array([1,1,1,0,1]).reshape(-1,1)
    avg_per_cat, count_per_cat = stratx.partdep.avg_values_at_cat(leaf_deltas, leaf_counts)
    expected = np.array([0, 1, 2, nan, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)


def test_two_leaves_with_one_refcat():
    leaf_deltas = np.array([
        [0,0],
        [1,1],
        [2,3],
        [nan,2],
        [0,nan],
        [nan, nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    # expected_avg_per_cat = np.array([0, 1, 2.5, 2, 0, nan])
    expected_avg_per_cat = np.array([-0.17,  0.83,  2.33,  1.67,  0,  nan])
    expected_count_per_cat = np.array([2, 2, 2, 1, 1, 0])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)


def test_two_leaves_with_two_refcats():
    leaf_deltas = np.array([
        [0,nan],
        [1,0],
        [2,3],
        [nan,2],
        [0,nan],
        [nan, nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    expected_avg_per_cat = np.array([0, 0.5, 2.5, 2, 0, nan])
    expected_count_per_cat = np.array([1, 2, 2, 1, 1, 0])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)


def test_merging_gets_negative_values():
    # Within a leaf, all deltas are >=0 but merging can cause negative values
    leaf_deltas = np.array([
        [0,   nan],
        [1,   nan],
        [2,     5],
        [nan,   0],
        [0,   nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    expected_avg_per_cat = np.array([0, 1, 2, -3, 0])
    expected_count_per_cat = np.array([1, 1, 2, 1, 1])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)


def test_two_leaves_with_non_0_and_1_catcodes():
    leaf_deltas = np.array([
        [nan,    nan],
        [nan,    nan],
        [0,      nan],
        [5,      nan],
        [1,      0],
        [2,      3],
        [nan,    7],
        [0,      nan],
        [nan,    nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    expected_avg_per_cat = np.array([nan, nan, 0, 5, 0.5, 2.5, 7, 0, nan])
    expected_count_per_cat = np.array([0, 0, 1, 1, 2, 2, 1, 1, 0])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)


def test_two_leaves_with_disconnected_2nd_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    leaf_deltas = np.array([
        [0,      nan],
        [1,      nan],
        [2,      nan],
        [nan,      0],  # refcat is 3 which has no value in other leaf
        [nan,      3]
    ])
    leaf_counts = np.array([
        [1,      0],
        [1,      0],
        [5,      0],
        [0,      3],  # simulate more than single record per cat
        [0,      2]
    ], dtype=np.int)
    # invent some avg y's per category (marginal) with 2nd group lower than first
    marginal_avg_y_per_cat = np.array([4,8,12, 2,4])

    avg_y_group1 = np.mean([4,8,12])
    avg_y_group2 = np.mean([2,4])
    group_averages = [avg_y_group1,avg_y_group2]
    d = group_averages - group_averages[0]
    expected_avg_per_cat = np.array([d[0]+0, d[0]+1, d[0]+2,   d[1]+0, d[1]+3])
    expected_count_per_cat = np.array([1, 1, 5, 3, 2])

    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas, marginal_avg_y_per_cat=marginal_avg_y_per_cat, leaf_counts=leaf_counts)


def test_3_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_disconnected_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum.
    """
    leaf_deltas = np.array([
        [0,    nan,  nan],
        [1,    nan,  nan],
        [nan,    0,  nan],  # refcat is 2 which has no value in prev leaf
        [nan,    3,    0],  # leaf 3 is connected to leaf 2 but should be ignored
        [nan,    3,    0],
        [4,    nan,  nan],
        [5,    nan,  nan],
    ])
    # invent some avg y's per category (marginal) with 2nd group higher than first
    marginal_avg_y_per_cat = np.array([3,2,  4,8,12,  4,6])

    avg_y_group1 = np.mean([3,2,4,6])
    avg_y_group2 = np.mean([4,8,12])
    group_averages = [avg_y_group1,avg_y_group2]
    d = group_averages - np.min(group_averages)
    expected_avg_per_cat = np.array([d[0]+0, d[0]+1,  d[1]+0, d[1]+3, d[1]+3,  d[0]+4, d[0]+5])
    expected_count_per_cat = np.array([1, 1, 1, 2, 2, 1, 1])

    # print("leaf_counts\n", leaf_counts)
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas, marginal_avg_y_per_cat=marginal_avg_y_per_cat)


def test_3_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_first_leaf():
    leaf_deltas = np.array([
        [0,  nan,  nan],
        [1,  nan,  nan],
        [nan,  0,  nan],  # refcat is 2 which has no value in prev leaf
        [nan,  3,  nan],  # leave 3 is connected to leaf 1 don't ignored
        [nan,  3,  nan],
        [4,  nan,    0],
        [5,  nan,    1],
    ])
    # invent some avg y's per category (marginal) with 2nd group higher than first
    marginal_avg_y_per_cat = np.array([3,2,  4,8,12,  4,6])

    avg_y_group1 = np.mean([3,2,4,6])
    avg_y_group2 = np.mean([4,8,12])
    group_averages = [avg_y_group1,avg_y_group2]
    d = group_averages - np.min(group_averages)
    expected_avg_per_cat = np.array([d[0]+0, d[0]+1,  d[1]+0, d[1]+3, d[1]+3,  d[0]+4, d[0]+5])
    expected_count_per_cat = np.array([1, 1, 1, 1, 1, 2, 2])

    # print("leaf_deltas\n",leaf_deltas)
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas, marginal_avg_y_per_cat=marginal_avg_y_per_cat)


def test_3_leaves_with_2nd_incorporated_in_pass_2():
    leaf_deltas = np.array([
        [0,   nan, nan],
        [1,   nan, nan],
        [nan, 0,   nan],  # refcat is 2 which has no value in prev leaf
        [nan, 3,   nan],  # leave 3 is connected to leaf 1 don't ignored
        [nan, 3,     0],  # leaf 2 will appear in pass 2
        [4,   nan,   7],
        [5,   nan,   8],
        [6,   nan, nan],
        [nan, 4,   nan],
    ])
    # print("leaf_deltas\n",leaf_deltas)
    expected_avg_per_cat = np.array([0, 1, -6, -3, -3, 4, 5, 6, -2])
    expected_count_per_cat = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)


def test_4state_temperature():
    set_random_seed(999)
    X,y,states,df_avgs = toy_weather_data(n=9, p=4)

    """
    leaf_deltas:
    [[   nan    nan   0.  ]
     [  0.      nan -24.9 ]
     [ 45.97   0.      nan]
     [ 38.19  -3.78    nan]]
    """
    leaf_deltas, leaf_counts, ignored = stratify_cats(X,y,colname="state",min_samples_leaf=3)

    expected_avg_per_cat = np.array([24.900, 0.000, 44.970, 39.190])
    expected_count_per_cat = np.array([1, 2, 2, 2])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)

def test_temperature():
    """
    leaf_deltas

    [[  nan 26.85 27.9 ]
     [ 0.    0.    0.  ]
     [38.97   nan 30.37]
     [33.39 23.99 24.34]
     [28.67 21.47 19.37]]

    counts

    [[0 2 1]
     [1 1 1]
     [1 0 2]
     [1 2 2]
     [2 3 1]]
    """
    set_random_seed(999)
    X,y,states,df_avgs = toy_weather_data(n=20, p=5)

    leaf_deltas, leaf_counts, ignored = \
        stratify_cats(X,y,colname="state",min_samples_leaf=5)

    print(leaf_deltas)
    print(leaf_counts)

    expected_avg_per_cat = np.array([33.28,  3.94, 37.81, 31.18, 27.11])
    expected_count_per_cat = np.array([2, 3, 2, 3, 3])
    check(expected_avg_per_cat, expected_count_per_cat, leaf_deltas)


def set_random_seed(s):
    np.random.seed(s)