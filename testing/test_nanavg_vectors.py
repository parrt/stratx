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

