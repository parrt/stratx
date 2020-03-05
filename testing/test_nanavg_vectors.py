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
import pandas as pd
import matplotlib.pyplot as plt

from stratx.partdep import *

def test_basic():
    a = np.array([0,1,2])
    b = np.array([0,1,2])
    c = nanavg_vectors(a, b)
    expected = np.array([0,1,2])
    np.testing.assert_array_equal(c, expected)

def test_avg():
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    c = nanavg_vectors(a, b)
    expected = np.array([2.5,3.5,4.5])
    np.testing.assert_array_equal(c, expected)

def test_weighted_avg():
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    c = nanavg_vectors(a, b, 2.0, 3.0)
    expected = np.array([2.8,3.8,4.8])
    np.testing.assert_array_equal(c, expected)

def test_a_missing_value():
    a = np.array([0,np.nan,2])
    b = np.array([0,1,2])
    c = nanavg_vectors(a, b)
    expected = np.array([0,1,2])
    np.testing.assert_array_equal(c, expected)

def test_b_missing_value():
    a = np.array([0,1,2])
    b = np.array([0,np.nan,2])
    c = nanavg_vectors(a, b)
    expected = np.array([0,1,2])
    np.testing.assert_array_equal(c, expected)

def test_both_missing_values():
    a = np.array([6,1,np.nan])
    b = np.array([6,np.nan,2])
    c = nanavg_vectors(a, b)
    expected = np.array([6,1,2])
    np.testing.assert_array_equal(c, expected)

