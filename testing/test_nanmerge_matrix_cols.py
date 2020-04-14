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

def test_same_refcat():
    A = np.array([[0,0],
                  [1,2],
                  [3,4]],
                 dtype=float)
    B = nanmerge_matrix_cols(A)
    expected = np.array([0,3,7])
    np.testing.assert_array_equal(B, expected)

def test_nan_does_not_kill_value():
    A = np.array([[0,0],
                  [1,np.nan],
                  [np.nan,4]],
                 dtype=float)
    B = nanmerge_matrix_cols(A)
    expected = np.array([0,1,4])
    np.testing.assert_array_equal(B, expected)

def test_nan_nan_is_nan():
    A = np.array([[0,0],
                  [np.nan,np.nan],
                  [5,4]],
                 dtype=float)
    B = nanmerge_matrix_cols(A)
    expected = np.array([0,np.nan,9])
    np.testing.assert_array_equal(B, expected)

def test_3col():
    A = np.array([[0,      1,      2],
                  [np.nan, np.nan, np.nan],
                  [4,      np.nan, 5],
                  [2,      np.nan, 3]],
                 dtype=float)
    B = nanmerge_matrix_cols(A)
    expected = np.array([3, np.nan, 9, 5])
    np.testing.assert_array_equal(B, expected)