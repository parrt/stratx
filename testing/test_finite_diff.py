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

import stratx.partdep


def check(x, y, expected_xranges, expected_slopes, expected_ignored=0):
    leaf_xranges, leaf_slopes, ignored = stratx.partdep.finite_differences(x, y)
    print(list(leaf_xranges), list(leaf_slopes), ignored)

    assert ignored==expected_ignored, f"Expected ignored {expected_ignored} got {ignored}"
    assert len(leaf_xranges)==len(expected_slopes), f"Expected ranges {expected_xranges}"
    assert stratx.partdep.np.isclose(leaf_xranges, stratx.partdep.np.array(expected_xranges)).all(), f"Expected ranges {expected_xranges} got {leaf_xranges}"
    assert len(leaf_slopes)==len(expected_slopes), f"Expected slopes {expected_slopes}"
    assert stratx.partdep.np.isclose(leaf_slopes, stratx.partdep.np.array(expected_slopes)).all(), f"Expected slopes {expected_slopes} got {leaf_slopes}"


def test_one_uniq_x():
    x = stratx.partdep.np.array([1, 1])
    y = stratx.partdep.np.array([5, 12])

    expected_xranges = stratx.partdep.np.array([[0]])
    expected_slopes = stratx.partdep.np.array([0])
    check(x, y, expected_xranges, expected_slopes, expected_ignored=2)


def test_just_forward_diff_at_left():
    x = stratx.partdep.np.array([1, 3])
    y = stratx.partdep.np.array([5, 12])

    expected_xranges = stratx.partdep.np.array([[1, 3]])
    expected_slopes = stratx.partdep.np.array([3.5])
    check(x, y, expected_xranges, expected_slopes)


def test_forward_diff_and_one_center():
    x = stratx.partdep.np.array([1, 3, 4])
    y = stratx.partdep.np.array([5, 12, 7])

    expected_xranges = stratx.partdep.np.array([[1, 3], [3, 4]])
    expected_slopes = stratx.partdep.np.array([3.5, -5])
    check(x, y, expected_xranges, expected_slopes)


def test_center_diff():
    x = stratx.partdep.np.array([1, 3, 4, 7, 13])
    y = stratx.partdep.np.array([5, 6, 8, 11, 15])

    expected_xranges = stratx.partdep.np.array([[1, 3], [3, 4], [4, 7], [7, 13]])
    expected_slopes = stratx.partdep.np.array([0.5, 2.0, 1, 0.6666666667])
    print(stratx.partdep.np.gradient(y, x))
    check(x, y, expected_xranges, expected_slopes)
