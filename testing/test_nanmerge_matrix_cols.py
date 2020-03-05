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