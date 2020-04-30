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

import stratx.partdep as partdep
import stratx.featimp as featimp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import nan

def test_binary():
    a = np.array([4,2])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.abs(4-2)
    np.testing.assert_array_equal(b, expected)

def test_binary_zero():
    a = np.array([0,2])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.abs(0-2)
    np.testing.assert_array_equal(b, expected)

def test_binary_zero_reversed():
    a = np.array([2,0])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.abs(2-0)
    np.testing.assert_array_equal(b, expected)

def test_3():
    a = np.array([0,1,2])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.nanmean(np.abs([0-1,0-2,1-2]))
    np.testing.assert_array_equal(b, expected)

def test_3_reversed():
    a = np.array([2,1,0])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.nanmean(np.abs([2-1,2-0,1-0]))
    np.testing.assert_array_equal(b, expected)

def test_4_reversed():
    a = np.array([1,2,3,4])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.nanmean(np.abs([1-2,1-3,1-4,2-3,2-4,3-4]))
    np.testing.assert_array_equal(b, expected)

def test_2_nan_ignored():
    a = np.array([nan,1,nan,0])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.nanmean(np.abs([1-0]))
    np.testing.assert_array_equal(b, expected)

def test_4_nan_ignored():
    a = np.array([1,nan,nan,nan,3,nan,5,nan,7,nan])
    b = featimp.avg_all_pairs_abs_delta(a)
    expected = np.nanmean(np.abs([1-3,1-5,1-7,3-5,3-7,5-7]))
    np.testing.assert_array_equal(b, expected)
