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
    np.random.seed(999)
    leaf_deltas = np.array([
        [0],
        [1],
        [2],
        [np.nan],
        [0]
    ])
    leaf_counts = np.array([1,1,1,0,1]).reshape(-1,1)
    refcats = np.array([0])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([0, 1, 2, np.nan, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==0


def test_two_leaves_with_one_refcat():
    np.random.seed(999)
    leaf_deltas = np.array([
        [0,0],
        [1,5],
        [2,3],
        [np.nan,2],
        [0,np.nan],
        [np.nan, np.nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,0])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([0,  3,  2.5, 2,  0,  np.nan])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==0


def test_two_leaves_with_two_refcats():
    np.random.seed(999)
    leaf_deltas = np.array([
        [0,np.nan],
        [1,0],
        [2,3],
        [np.nan,2],
        [0,np.nan],
        [np.nan, np.nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,1])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([0, 1, 3, 3, 0, np.nan])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==0


def test_two_leaves_with_non_0_and_1_catcodes():
    np.random.seed(999)
    leaf_deltas = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [0,      np.nan],
        [5,      np.nan],
        [1,      0],
        [2,      3],
        [np.nan, 7],
        [0,      np.nan],
        [np.nan, np.nan]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([2,4])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([np.nan, np.nan, 0, 5, 1, 3, 8, 0, np.nan])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==0


def test_two_leaves_with_disconnected_2nd_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    np.random.seed(999)
    leaf_deltas = np.array([
        [0,      np.nan],
        [1,      np.nan],
        [2,      np.nan],
        [np.nan, 0],  # refcat is 3 which has no value in other leaf
        [np.nan, 3]
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,3])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([0, 1, 2, np.nan, np.nan])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==2


def test_3_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_disconnected_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    np.random.seed(999)
    leaf_deltas = np.array([
        [0,      np.nan, np.nan],
        [1,      np.nan, np.nan],
        [np.nan, 0,      np.nan],  # refcat is 2 which has no value in prev leaf
        [np.nan, 3,      0],       # leave 3 is connected to leaf 2 but should be ignored
        [np.nan, 3,      2],
        [4, np.nan, np.nan],       # we sort by weight so add some to bottom
        [5, np.nan, np.nan],
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,2,3])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([0, 1, np.nan, np.nan, np.nan, 4, 5])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==5


def test_3_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_first_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    np.random.seed(999)
    leaf_deltas = np.array([
        [0,      np.nan, np.nan],
        [1,      np.nan, np.nan],
        [np.nan, 0,      np.nan],  # refcat is 2 which has no value in prev leaf
        [np.nan, 3,      np.nan],  # leave 3 is connected to leaf 1 don't ignored
        [np.nan, 3,      np.nan],
        [4, np.nan,      0],
        [5, np.nan,      1],
    ])
    # print("leaf_deltas\n",leaf_deltas)
    leaf_counts = (~np.isnan(leaf_deltas)).astype(int)
    refcats = np.array([0,2,3])
    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats, verbose=True)
    expected = np.array([0, 1, np.nan, np.nan, np.nan, 4, 5])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==3


def test_4state_temperature():
    np.random.seed(999)
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

    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    print(avg_per_cat)
    expected = np.array([-13.29, -38.19, 5.78, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected)
    assert ignored==0


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
    np.random.seed(999)
    X,y,states,df_avgs = toy_weather_data(n=20, p=8)

    leaf_deltas, leaf_counts, refcats, ignored = stratify_cats(X,y,colname="state",min_samples_leaf=5)

    avg_per_cat, ignored = avg_values_at_cat(leaf_deltas, leaf_counts, refcats)
    expected = np.array([7.17, -19.48, 10.59, 4.26, 3.79, -8.52, -4.78, 0])
    np.testing.assert_array_almost_equal(avg_per_cat, expected, decimal=2)
    assert ignored==0