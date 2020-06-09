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

def check(X, y, colname,
          expected_xranges, expected_slopes,
          expected_pdpx, expected_pdpy,
          expected_ignored=0,
          min_samples_leaf=15):
    leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
        partial_dependence(X, y, colname,
                           min_samples_leaf=min_samples_leaf,
                           min_slopes_per_x=1)

    # print(leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy)

    assert ignored==expected_ignored, f"Expected ignored {expected_ignored} got {ignored}"
    assert len(leaf_xranges)==len(expected_slopes), f"Expected ranges {expected_xranges}"
    assert np.isclose(leaf_xranges, np.array(expected_xranges)).all(), f"Expected ranges {expected_xranges} got {leaf_xranges}"
    assert len(leaf_slopes)==len(expected_slopes), f"Expected slopes {expected_slopes}"
    assert np.isclose(leaf_slopes, np.array(expected_slopes)).all(), f"Expected slopes {expected_slopes} got {leaf_slopes}"
    assert len(pdpx)==len(expected_pdpx), f"Expected pdpx {expected_pdpx}"
    assert np.isclose(pdpx, np.array(expected_pdpx)).all(), f"Expected pdpx {expected_pdpx} got {pdpx}"
    assert len(pdpy)==len(expected_pdpy), f"Expected pdpy {expected_pdpy}"
    assert np.isclose(pdpy, np.array(expected_pdpy)).all(), f"Expected pdpy {expected_pdpy} got {pdpy}"


def test_binary_one_region():
    df = pd.DataFrame()
    df['x1'] = [1, 1]
    df['x2'] = [65, 60]
    df['y'] =  [100, 130]
    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = np.array([[60, 65]])
    expected_slopes = np.array([-6])
    expected_pdpx = np.array([60,65])
    expected_pdpy = np.array([0,-30])

    check(X, y, "x2",
          expected_xranges, expected_slopes,
          expected_pdpx, expected_pdpy,
          min_samples_leaf=2)


def test_one_region():
    df = pd.DataFrame()
    df['x1'] = [1, 1, 1]
    df['x2'] = [100,101,102]
    df['y'] =  [10, 11, 12]
    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = np.array([[100, 101],
                                 [101, 102]])
    expected_slopes = np.array([1, 1])
    expected_pdpx = np.array([100,101,102])
    expected_pdpy = np.array([0,    1,  2])

    check(X, y, "x2",
          expected_xranges, expected_slopes,
          expected_pdpx, expected_pdpy,
          min_samples_leaf=3)


def test_disjoint_regions():
    """
    What happens when we have two disjoint regions in x_j space?
    Does the 2nd start with 0 again with cumsum?
    """
    df = pd.DataFrame()
    df['x1'] = [1, 1, 1,    # stratify first three x2
                5, 5, 5]    # stratify 2nd three x2
    df['x2'] = [100,101,102,
                200,201,202]
    df['y'] =  [10, 11, 12, # first x2 region +1 slope
                20, 19, 18] # 2nd x2 region -1 slope but from higher y downwards
    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = np.array([[100, 101],
                                 [101, 102],
                                 [200, 201],
                                 [201, 202]])
    expected_slopes = np.array([1, 1,        -1, -1])
    expected_pdpx = np.array([100,101,102,   200,201,202])
    expected_pdpy = np.array([0,    1,  2,   2,    1,  0])

    check(X, y, "x2",
          expected_xranges, expected_slopes,
          expected_pdpx, expected_pdpy,
          min_samples_leaf=3)


def test_disjoint_regions_with_isolated_single_x_in_between():
    df = pd.DataFrame()
    df['x1'] = [1, 1, 1,    # stratify first three x2
                3, 3, 3,    # stratify middle group
                5, 5, 5]    # stratify 3rd group x2
    df['x2'] = [100,101,102,
                150,150,150,# middle of other groups and same x so no slope
                200,201,202]
    df['y'] =  [10, 11, 12, # first x2 region +1 slope
                0,   0,  0, # y value doesn't matter; no slope to compute
                20, 19, 18] # 2nd x2 region -1 slope but from higher y downwards
    X = df.drop('y', axis=1)
    y = df['y']

    expected_xranges = np.array([[100, 101],
                                 [101, 102],
                                 [200, 201],
                                 [201, 202]])
    expected_slopes = np.array([1, 1,        -1, -1])
    expected_pdpx = np.array([100,101,102,   200,201,202]) # NOTE: no x=150 position
    expected_pdpy = np.array([0,    1,  2,   2,    1,  0])

    check(X, y, "x2",
          expected_xranges, expected_slopes,
          expected_pdpx, expected_pdpy,
          min_samples_leaf=3,
          expected_ignored=3) # ignores position x=150

