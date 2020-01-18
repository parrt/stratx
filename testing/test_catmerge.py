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

np.random.seed(999)

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
    # leaf_histos, leaf_avgs, leaf_sizes, leaf_catcounts, ignored = \
    #     catwise_leaves(rf, X, y, colname, verbose=verbose)

    leaf_histos, refcats, ignored = \
        catwise_leaves(rf, X_not_col, X_col, y.values, max_catcode)

    return leaf_histos, refcats, ignored

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
    df_avgs.head(3)

    df = pd.DataFrame()
    df['dayofyear'] = np.random.randint(1, 365 + 1, size=n)
    df['state'] = np.random.randint(0, p, size=n) # get only p states
    df['temp'] = .1 * df['dayofyear'] + df_avgs['avgtemp'].iloc[df['state']].values
    return df.drop('temp', axis=1), df['temp'], df_avgs['state'].values, df_avgs.iloc[0:p]


def test_single_leaf():
    leaf_histos = np.array([
        [0],
        [1],
        [2],
        [np.nan],
        [0]
    ])
    refcats = np.array([0])
    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    expected = np.array([0, 1, 2, np.nan, 0])
    np.testing.assert_array_equal(avg_per_cat, expected)
    assert ignored==0


def test_two_leaves_with_one_refcat():
    leaf_histos = np.array([
        [0,0],
        [1,5],
        [2,3],
        [np.nan,2],
        [0,np.nan],
        [np.nan, np.nan]
    ])
    # print("leaf_histos\n",leaf_histos)
    refcats = np.array([0,0])
    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    expected = np.array([0,  3,  2.5, 2,  0,  np.nan])
    np.testing.assert_array_equal(avg_per_cat, expected)
    assert ignored==0


def test_two_leaves_with_two_refcats():
    leaf_histos = np.array([
        [0,np.nan],
        [1,0],
        [2,3],
        [np.nan,2],
        [0,np.nan],
        [np.nan, np.nan]
    ])
    # print("leaf_histos\n",leaf_histos)
    refcats = np.array([0,1])
    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    expected = np.array([0, 1, 3, 3, 0, np.nan])
    np.testing.assert_array_equal(avg_per_cat, expected)
    assert ignored==0


def test_two_leaves_with_non_0_and_1_catcodes():
    leaf_histos = np.array([
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
    # print("leaf_histos\n",leaf_histos)
    refcats = np.array([2,4])
    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    expected = np.array([np.nan, np.nan, 0, 5, 1, 3, 8, 0, np.nan])
    np.testing.assert_array_equal(avg_per_cat, expected)
    assert ignored==0


def test_two_leaves_with_disconnected_2nd_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    leaf_histos = np.array([
        [0,      np.nan],
        [1,      np.nan],
        [np.nan, 0],  # refcat is 2 which has no value in other leaf
        [np.nan, 3]
    ])
    # print("leaf_histos\n",leaf_histos)
    refcats = np.array([0,2])
    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    expected = np.array([0, 1, np.nan, np.nan])
    np.testing.assert_array_equal(avg_per_cat, expected)
    assert ignored==1


def test_two_leaves_with_disconnected_2nd_leaf_followed_by_leaf_conn_to_disconnected_leaf():
    """
    It's possible for a leaf's refcat not to have a value in any earlier
    refcats, leaving nan in the running sum. No way to connect, must just drop
    """
    leaf_histos = np.array([
        [0,      np.nan, np.nan],
        [1,      np.nan, np.nan],
        [np.nan, 0,      np.nan],  # refcat is 2 which has no value in prev leaf
        [np.nan, 3,      0],       # leave 3 is connected to leaf 2 but should be ignored
        [np.nan, 3,      2]
    ])
    # print("leaf_histos\n",leaf_histos)
    refcats = np.array([0,2,3])
    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    expected = np.array([0, 1, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(avg_per_cat, expected)
    assert ignored==3


def test_temperature():
    X,y,states,df_avgs = toy_weather_data(n=9, p=4)

    leaf_histos, refcats, ignored = stratify_cats(X,y,colname="state",min_samples_leaf=3)

    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats)
    print(avg_per_cat)
    expected = np.array([0, -24.9, 21.07, 15.29])
    np.testing.assert_array_almost_equal(avg_per_cat, expected)
    assert ignored==0


# test_single_leaf()
# test_two_leaves_with_one_refcat()
# test_two_leaves_with_two_refcats()
# test_two_leaves_with_disconnected_2nd_leaf()