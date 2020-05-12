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

import numpy as np
from numpy import nan, where
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from stratx.partdep import *

def slopes(X, y, colname, min_samples_leaf=10):
    rf = RandomForestRegressor(n_estimators=1,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=False,
                               max_features=1.0)
    rf.fit(X.drop(colname, axis=1), y)
    return collect_discrete_slopes(rf, X[colname], X.drop(colname, axis=1).values, y)


def check(X, y, colname, expected_xranges, expected_slopes, expected_ignored=0, min_samples_leaf=15):
    leaf_xranges, leaf_slopes, ignored = \
        slopes(X, y, colname=colname, min_samples_leaf=min_samples_leaf)

    # print(leaf_xranges, leaf_slopes, ignored)

    assert ignored==expected_ignored, f"Expected ignored {expected_ignored} got {ignored}"
    assert len(leaf_xranges)==len(expected_slopes), f"Expected ranges {expected_xranges}"
    assert np.isclose(leaf_xranges, np.array(expected_xranges)).all(), f"Expected ranges {expected_xranges} got {leaf_xranges}"
    assert len(leaf_slopes)==len(expected_slopes), f"Expected slopes {expected_slopes}"
    assert np.isclose(leaf_slopes, np.array(expected_slopes)).all(), f"Expected slopes {expected_slopes} got {leaf_slopes}"


def test_2_discrete_records_positive_slope():
    """
       x1  x2  y
    0   1   4  4
    1   2   5  7
    """
    data = {"x1":[1, 2], "x2":[4, 5], "y":[4, 7]}
    df = pd.DataFrame.from_dict(data)
    # print(df)

    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = [[1, 2]]
    expected_slopes = [3]
    check(X, y, 'x1', expected_xranges, expected_slopes)

def test_unsupported_ignored():
    """
       x1  x2  y
    0   1   5  4
    1   2   5  7
    """
    data = {"x1":[1, 2], "x2":[5, 5], "y":[4, 7]}
    df = pd.DataFrame.from_dict(data)
    # print(df)

    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = []
    expected_slopes = []
    check(X, y, 'x2', expected_xranges, expected_slopes, expected_ignored=2)


def test_2_discrete_records_negative_slope():
    """
       x1  x2  y
    0   1   4  7
    1   2   5  4
    """
    data = {"x1":[1, 2], "x2":[4, 5], "y":[7, 4]}
    df = pd.DataFrame.from_dict(data)
    # print(df)

    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = [[1, 2]]
    expected_slopes = [-3]
    check(X, y, 'x1', expected_xranges, expected_slopes)

def test_sim_2cat_problem():
    """
       x1  x2  y
    0   1   5  0
    1   2   5  30
    """
    data = {"x1":[1, 2], "x2":[5, 5], "y":[0, 30]}
    df = pd.DataFrame.from_dict(data)
    # print(df)

    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = np.array([[1, 2]])
    expected_slopes = np.array([30])
    check(X, y, 'x1', expected_xranges, expected_slopes)

    real_uniq_x = np.unique(X['x1'])
    slope_at_x, slope_counts_at_x = \
        avg_slopes_at_x_jit(real_uniq_x, expected_xranges, expected_slopes)

    print(slope_at_x, slope_counts_at_x)
