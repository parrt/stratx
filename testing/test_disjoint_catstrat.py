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

from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import nan

# TEST CATSTRATPD ON SYNTHETIC (SOME COLINEAR) DATASETS WITH KNOWN ANSWERS

def check(X, y, colname,
          expected_deltas, expected_avg_per_cat,
          expected_ignored=0,
          min_samples_leaf=15):
    leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored = \
        cat_partial_dependence(X, y, colname, min_samples_leaf=min_samples_leaf)

    print(leaf_deltas, avg_per_cat)

    # Normalize so all 0-based
    expected_avg_per_cat -= np.nanmin(expected_avg_per_cat)
    avg_per_cat -= np.nanmin(avg_per_cat)

    assert ignored==expected_ignored, f"Expected ignored {expected_ignored} got {ignored}"
    assert len(leaf_deltas)==len(expected_deltas), f"Expected ranges {expected_deltas}"
    np.testing.assert_array_equal(leaf_deltas, expected_deltas, f"Expected deltas {expected_deltas} got {leaf_deltas}")
    assert len(avg_per_cat)==len(expected_avg_per_cat), f"Expected slopes {expected_avg_per_cat}"
    np.testing.assert_array_equal(avg_per_cat, expected_avg_per_cat, f"Expected slopes {expected_avg_per_cat} got {avg_per_cat}")


def test_binary_one_region():
    df = pd.DataFrame()
    df['x1'] = [1, 1]
    df['x2'] = [3, 4]
    df['y'] =  [100, 130]
    X = df.drop('y', axis=1)
    y = df['y']

    expected_deltas = np.array([nan, nan, nan, 0, 30]).reshape(-1,1)
    expected_avg_per_cat = np.array([nan, nan, nan, 0, 30])

    check(X, y, "x2", expected_deltas, expected_avg_per_cat, min_samples_leaf=2)


def test_one_region():
    df = pd.DataFrame()
    df['x1'] = [1, 1, 1]
    df['x2'] = [3, 4, 5]
    df['y'] =  [10, 15, 20]
    X = df.drop('y', axis=1)
    y = df['y']

    expected_deltas = np.array([nan, nan, nan, 0, 5, 10]).reshape(-1,1)
    expected_avg_per_cat = np.array([nan, nan, nan, 0, 5, 10])

    check(X, y, "x2", expected_deltas, expected_avg_per_cat, min_samples_leaf=3)


def test_disjoint_regions():
    """
    What happens when we have two disjoint regions in x_j space?
    Does the 2nd start with 0 again with cumsum?
    """
    df = pd.DataFrame()
    df['x1'] = [1, 1, 1,    # stratify first three x2
                3, 3, 3]    # stratify 2nd three x2
    df['x2'] = [1, 2, 3,
                5, 6, 7]
    df['y'] =  [10, 13, 16, # first x2 region +2 slope
                50, 60, 70] # second x2 region +10 slope
    X = df.drop('y', axis=1)
    y = df['y']

    avg_y_group1 = np.mean([10, 13, 16])
    avg_y_group2 = np.mean([50, 60, 70])
    group_averages = [avg_y_group1,avg_y_group2]
    d = group_averages - group_averages[0]

    # plt.bar([1,2,3,5,6,7],np.array([10,13,15,50,60,70]))
    # plt.bar([1,2,3,5,6,7],np.array([10,13,15,50,60,70])-10)
    # plt.show()

    expected_deltas = np.array([[nan, nan],
                                [  0, nan],
                                [  3, nan],
                                [  6, nan],
                                [nan, nan],
                                [nan,   0],  # index cat 5
                                [nan,  10],
                                [nan,  20]])
    expected_avg_per_cat = np.array([nan, d[0]+0, d[0]+3, d[0]+6, nan, d[1]+0, d[1]+10, d[1]+20])

    check(X, y, "x2", expected_deltas, expected_avg_per_cat, min_samples_leaf=3)


def test_disjoint_regions_with_isolated_single_x_in_between():
    df = pd.DataFrame()
    df['x1'] = [1, 1, 1,    # stratify first three x2
                3, 3, 3,    # stratify middle group
                5, 5, 5]    # stratify 3rd group x2
    df['x2'] = [1, 2, 3,
                4, 4, 4,    # middle of other groups and same cat
                5, 6, 7]
    df['y'] =  [10, 11, 12, # first x2 region +1 slope
                9,  7,  8,
                20, 19, 18] # 2nd x2 region -1 slope but from higher y downwards
    X = df.drop('y', axis=1)
    y = df['y']

    avg_y_group1 = np.mean([10, 11, 12])
    avg_y_group2 = np.mean([9,   7, 8])
    avg_y_group3 = np.mean([20, 19, 18])
    group_averages = [avg_y_group1,avg_y_group2,avg_y_group3]
    d = group_averages - group_averages[0]

    expected_deltas = np.array([[nan, nan, nan],
                                [0,   nan, nan],
                                [1,   nan, nan],
                                [2,   nan, nan],
                                [nan,   0, nan],
                                [nan, nan,   2],
                                [nan, nan,   1],
                                [nan, nan,   0]])
    expected_avg_per_cat = np.array([nan, d[0]+0, d[0]+1, d[0]+2,   d[1]+0,   d[2]+2, d[2]+1, d[2]+0])

    check(X, y, "x2",
          expected_deltas, expected_avg_per_cat,
          min_samples_leaf=3)


def test_sawtooth_derivative_disjoint_regions_bulldozer():
    df = pd.DataFrame()
    df['YearMade'] = [1, 1, 1,  # stratify into 4 groups
                      3, 3, 3,
                      5, 5, 5,
                      7, 7, 7]
    df['ModelID'] = [1,   2,  3,
                     5,   6,  7,
                     9,  10, 11,
                     13, 14, 15]
    df['y'] =  [10,  9,  8,
                12, 13, 14,
                 6,  5,  4,
                16, 17, 18]
    print((df['y']+10).values)
    X = df.drop('y', axis=1)
    y = df['y']

    leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored = \
        cat_partial_dependence(X, y, "ModelID", min_samples_leaf=3)

    expected_deltas = np.array([[nan, nan, nan, nan],
                                [  2, nan, nan, nan],
                                [  1, nan, nan, nan],
                                [  0, nan, nan, nan],
                                [nan, nan, nan, nan],
                                [nan,   0, nan, nan],
                                [nan,   1, nan, nan],
                                [nan,   2, nan, nan],
                                [nan, nan, nan, nan],
                                [nan, nan,   2, nan],
                                [nan, nan,   1, nan],
                                [nan, nan,   0, nan],
                                [nan, nan, nan, nan],
                                [nan, nan, nan,   0],
                                [nan, nan, nan,   1],
                                [nan, nan, nan,   2]])
    expected_avg_per_cat = np.array([nan, 2, 1, 0, nan, 4, 5, 6, nan, -2, -3, -4, nan, 8, 9, 10])
    check(X, y, "ModelID", expected_deltas, expected_avg_per_cat, min_samples_leaf=3)

    plt.figure(figsize=(4,3))
    plt.scatter(df['ModelID'], y, s=8, label="Actual", c='#4A4A4A')
    plt.scatter(df['ModelID'], avg_per_cat[np.where(~np.isnan(avg_per_cat))],
                marker="s",
                s=10, label="PDP", c='#FEAE61')
    plt.xlabel("ModelID")
    plt.ylabel("SalePrice")
    plt.legend(loc="upper left")
    plt.show()


def test_sawtooth_derivative_disjoint_regions_bulldozer_some_negative():
    df = pd.DataFrame()
    df['YearMade'] = [1, 1, 1,  # stratify into 4 groups
                      3, 3, 3,
                      5, 5, 5,
                      7, 7, 7]
    df['ModelID'] = [1,   2,  3,
                     5,   6,  7,
                     9,  10, 11,
                     13, 14, 15]
    df['y'] =  [ 0, -1, -2, # first x2 region  -1 slope
                 2,  3,  4, # second x2 region +1 slope
                -4, -5, -6,
                 6,  7,  8]
    X = df.drop('y', axis=1)
    y = df['y']

    leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored = \
        cat_partial_dependence(X, y, "ModelID", min_samples_leaf=3)

    expected_deltas = np.array([[nan, nan, nan, nan],
                                [  2, nan, nan, nan],
                                [  1, nan, nan, nan],
                                [  0, nan, nan, nan],
                                [nan, nan, nan, nan],
                                [nan,   0, nan, nan],
                                [nan,   1, nan, nan],
                                [nan,   2, nan, nan],
                                [nan, nan, nan, nan],
                                [nan, nan,   2, nan],
                                [nan, nan,   1, nan],
                                [nan, nan,   0, nan],
                                [nan, nan, nan, nan],
                                [nan, nan, nan,   0],
                                [nan, nan, nan,   1],
                                [nan, nan, nan,   2]])
    expected_avg_per_cat = np.array([nan, 2, 1, 0, nan, 4, 5, 6, nan, -2, -3, -4, nan, 8, 9, 10])
    check(X, y, "ModelID", expected_deltas, expected_avg_per_cat, min_samples_leaf=3)

    plt.figure(figsize=(4,3))
    plt.scatter(df['ModelID'], y, s=8, label="Actual", c='#4A4A4A')
    plt.scatter(df['ModelID'], avg_per_cat[np.where(~np.isnan(avg_per_cat))],
                marker="s",
                s=10, label="PDP", c='#FEAE61')
    plt.xlabel("ModelID")
    plt.ylabel("SalePrice")
    plt.legend(loc="upper left")
    plt.show()


# ------ Stuff below is mostly for exploring disjoint regions ------------

def synthetic_bulldozer(n:int, gaps_in_ModelID=False, drop_modelIDs=None):
    """
    Bulldozer with ModelID, MachineHours, YearMade, EROP -> SalePrice

    where EROP (cage description) in {1,2,3,4},
          YearMade in {2000..2010},
          ModelID in {100,500},
          MachineHours in 0..1000.

    EROP tied to ModelID
    ModelID tied to YearMade; randomize ID within tranches
    MachineHours is independent
    """
    # First, set up random column values but tie them to other columns
    df = pd.DataFrame()
    df['YearMade'] = np.random.randint(2000,2010+1, size=(n,))
    df['MachineHours'] = np.random.random(size=n)*1000
    df['ModelID'] = 0

    n1 = np.sum(df['YearMade'].isin([2000,2001,2002]))
    n2 = np.sum(df['YearMade'].isin([2000,2003,2005]))
    n3 = np.sum(df['YearMade'].isin(range(2005,2007+1)))
    n4 = np.sum(df['YearMade'].isin(range(2008,2010+1)))
    df.loc[df['YearMade'].isin([2000,2001,2002]), 'ModelID'] = np.random.randint(100, 200+1, size=(n1,))
    df.loc[df['YearMade'].isin([2000,2003,2005]), 'ModelID'] = np.random.randint(201, 300+1, size=(n2,))
    df.loc[df['YearMade'].isin(range(2005,2007+1)), 'ModelID'] = np.random.randint(301, 400+1, size=(n3,))
    df.loc[df['YearMade'].isin(range(2008,2010+1)), 'ModelID'] = np.random.randint(401, 500+1, size=(n4,))
    df.loc[df['ModelID']==0, "ModelID"] = np.random.randint(100,500+1, np.sum(df['ModelID']==0))

    if gaps_in_ModelID:
        df = df[(df['ModelID']<300)|(df['ModelID']>=400)] # kill 300..399

    if drop_modelIDs is not None:
        df = df.iloc[np.where(~df['ModelID'].isin(np.random.randint(100,500+1,size=drop_modelIDs)))] # drop some

    print("n =", len(df))
    df['EROP'] = 1                              # None
    df.loc[df['ModelID'] % 2==0, 'EROP'] = 2    # Sides only
    df.loc[df['ModelID'] % 3==0, 'EROP'] = 3    # Full
    df.loc[df['ModelID'] % 4==0, 'EROP'] = 4    # With AC

    # Compute a sawtooth for known values of different models
    modelvalue = df['ModelID'].isin(range(100,199+1)) * -df['ModelID'] + \
                 df['ModelID'].isin(range(200,299+1)) * df['ModelID'] + \
                 df['ModelID'].isin(range(300,399+1)) * -df['ModelID'] + \
                 df['ModelID'].isin(range(400,500+1)) * df['ModelID']

    age = np.max(df['YearMade']) - df['YearMade']
    df['SalePrice'] = 15_000 \
                      - age*1000 \
                      - df['MachineHours']*1.5 \
                      + (df['EROP']-1) * 200 \
                      + modelvalue*10
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # plt.scatter(df['ModelID'], y, s=3)
    # plt.show()
    plt.scatter(df['ModelID'], modelvalue*10, s=1)
    plt.show()

    return df, X, y


def random_sawtooth_derivative_disjoint_regions_bulldozer():
    df, X, y = synthetic_bulldozer(n=100,
                                   gaps_in_ModelID=False,
                                   drop_modelIDs=None)

    # rf = RandomForestRegressor(n_estimators=50, oob_score=True)
    # rf.fit(X,y)
    # print("OOB", rf.oob_score_)

    leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored = \
        cat_partial_dependence(X, y, "ModelID", min_samples_leaf=10)

    # plot_catstratpd(X, y, colname='ModelID', targetname='SalePrice',
    #                 n_trials=1,
    #                 min_samples_leaf=15,
    #                 show_xticks=False,
    #                 show_impact=True,
    #                 # min_y_shifted_to_zero=True,
    #                 figsize=(10,5),
    #                 # yrange=(-150_000, 150_000),
    #                 verbose=False)
    # plt.show()
    #
    # plot_stratpd(X, y, colname='ModelID', targetname='SalePrice',
    #              show_impact=True,
    #              min_slopes_per_x=1,
    #              min_samples_leaf=15)
    # plt.show()
