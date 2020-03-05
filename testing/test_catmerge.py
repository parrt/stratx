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


def get_leaves(X, y, colname, min_samples_leaf=1):
    X_not_col = X.drop(colname, axis=1).values
    rf = RandomForestRegressor(n_estimators=1,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=False,
                               max_features=1.0,
                               oob_score=False)
    rf.fit(X_not_col, y)
    leaves = leaf_samples(rf, X_not_col)
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
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    min_samples_leaf=min_samples_leaf * 2,
                                    # there are 2x as many samples (X,X') so must double leaf size
                                    bootstrap=bootstrap,
                                    max_features=max_features,
                                    oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    # rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X_not_col, y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    # leaf_deltas, leaf_counts, leaf_avgs, leaf_sizes, leaf_catcounts, ignored = \
    #     catwise_leaves(rf, X, y, colname, verbose=verbose)

    leaf_deltas, leaf_counts, refcats, ignored = \
        catwise_leaves(rf, X_not_col, X_col, y.values, max_catcode)

    return leaf_deltas, leaf_counts, refcats, ignored

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

    df_avgs = pd.read_csv("../articles/imp/genfigs/data/weather.csv")
    print("avg of states' avg temps:",np.mean(df_avgs['avgtemp']))

    df = pd.DataFrame()
    df['dayofyear'] = np.random.randint(1, 365 + 1, size=n)
    df['state'] = np.random.randint(0, p, size=n) # get only p states
    df['temp'] = .1 * df['dayofyear'] + df_avgs['avgtemp'].iloc[df['state']].values
    return df.drop('temp', axis=1), df['temp'], df_avgs['state'].values, df_avgs.iloc[0:p]

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
    refcats = np.array([0])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([0, 1, 2, nan, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert merge_ignored==0


def test_two_leaves_with_one_refcat():
    set_random_seed(999)
    leaf_deltas = np.array([
        [0,0],
        [1,5],
        [2,3],
        [nan,2],
        [0,nan],
        [nan, nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,0])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([0,  3,  2.5, 2,  0,  nan])
    expected_count_per_cat = np.array([2, 2, 2, 1, 1, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==0


def test_two_leaves_with_two_refcats():
    set_random_seed(999)
    leaf_deltas = np.array([
        [0,nan],
        [1,0],
        [2,3],
        [nan,2],
        [0,nan],
        [nan, nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,1])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([0, 1, 3, 3, 0, nan])
    expected_count_per_cat = np.array([1, 2, 2, 1, 1, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==0


def test_two_leaves_with_non_0_and_1_catcodes():
    set_random_seed(999)
    leaf_deltas = np.array([
        [nan, nan],
        [nan, nan],
        [0,      nan],
        [5,      nan],
        [1,      0],
        [2,      3],
        [nan, 7],
        [0,      nan],
        [nan, nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([2,4])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([nan, nan, 0, 5, 1, 3, 8, 0, nan])
    expected_count_per_cat = np.array([0, 0, 1, 1, 2, 2, 1, 1, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==0


def test_two_leaves_with_disconnected_2nd_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    set_random_seed(999)
    leaf_deltas = np.array([
        [0,      nan],
        [1,      nan],
        [2,      nan],
        [nan, 0],  # refcat is 3 which has no value in other leaf
        [nan, 3]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,3])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([0, 1, 2, nan, nan])
    expected_count_per_cat = np.array([1, 1, 1, 0, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==2


def test_3_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_disconnected_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    set_random_seed(999)
    leaf_deltas = np.array([
        [0,      nan, nan],
        [1,      nan, nan],
        [nan, 0,      nan],  # refcat is 2 which has no value in prev leaf
        [nan, 3,      0],    # leaf 3 is connected to leaf 2 but should be ignored
        [nan, 3,      2],
        [4, nan, nan],       # we sort by weight so add some to bottom
        [5, nan, nan],
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,2,3])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([0, 1, nan, nan, nan, 4, 5])
    expected_count_per_cat = np.array([1, 1, 0, 0, 0, 1, 1])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==5


def test_3_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_first_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    set_random_seed(999)
    leaf_deltas = np.array([
        [0,      nan, nan],
        [1,      nan, nan],
        [nan, 0,      nan],  # refcat is 2 which has no value in prev leaf
        [nan, 3,      nan],  # leave 3 is connected to leaf 1 don't ignored
        [nan, 3,      nan],
        [4, nan,      0],
        [5, nan,      1],
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,2,3])
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([0, 1, nan, nan, nan, 4, 5])
    expected_count_per_cat = np.array([1, 1, 0, 0, 0, 2, 2])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==3


def test_3_leaves_with_2nd_incorporated_in_pass_2():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    set_random_seed(999)
    leaf_deltas = np.array([
        [0,      nan, nan],
        [1,      nan, nan],
        [nan, 0,      nan],  # refcat is 2 which has no value in prev leaf
        [nan, 3,      nan],  # leave 3 is connected to leaf 1 don't ignored
        [nan, 3,      0],       # leaf 2 will appear in pass 2
        [4,      nan, 9],
        [5,      nan, 8],
        [6,      nan, nan],
        [nan, 4,      nan],
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,2,4]) # only used during merging same refcat so ignored during running sum
    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected_avg_per_cat = np.array([0, 1, -8, -5, -5, 4, 4, 6, -4])
    expected_count_per_cat = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1])
    np.testing.assert_array_almost_equal(avg_per_cat, expected_avg_per_cat, decimal=2)
    np.testing.assert_array_equal(count_per_cat, expected_count_per_cat)
    assert merge_ignored==0


def test_4state_temperature():
    set_random_seed(999)
    X,y,states,df_avgs = toy_weather_data(n=9, p=4)

    """
    leaf_deltas:
    [[   nan    nan   0.  ]
     [-38.19    nan -24.9 ]
     [  7.78   3.78    nan]
     [  0.     0.      nan]]
    
    First two columns merge right away due to common refcat:
    
    [[   nan   0.  ]
     [-38.19 -24.9 ]
     [  5.78    nan]
     [  0.      nan]]
    """
    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="state",min_samples_leaf=3)

    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([-13.29, -38.19, 5.78, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected)
    assert merge_ignored==0


def test_temperature():
    """
    avgs per refcat
    [[   nan   7.65    nan    nan]
     [-19.48 -19.      nan    nan]
     [ 10.59    nan    nan    nan]
     [  3.51    nan  15.38  -3.98]
     [   nan   4.27    nan   0.  ]
     [   nan    nan   0.   -12.31]
     [ -3.78   0.     4.84 -14.97]
     [  0.      nan   8.52    nan]]

    counts
    [[0 2 0 0]
     [1 1 0 0]
     [1 0 0 0]
     [1 0 2 1]
     [0 1 0 2]
     [0 0 1 1]
     [1 1 1 1]
     [1 0 1 0]]
    """
    set_random_seed(999)
    X,y,states,df_avgs = toy_weather_data(n=20, p=8)

    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="state",min_samples_leaf=5)

    avg_per_cat, count_per_cat, merge_ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([7.17, -19.48, 10.59, 4.26, 3.79, -8.52, -4.78, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert merge_ignored==0


def set_random_seed(s):
    np.random.seed(s)