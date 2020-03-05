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
    return collect_discrete_slopes(rf, X, y, colname)


def check(data, expected, colname, min_samples_leaf=10):
    """
    @param data: the X,y data, with y as last column
    @param expected: the left/right xranges and last column is slope
    """
    columns = [f'x{i}' for i in range(1,np.array(data).shape[1])]+['y']
    df = pd.DataFrame(data=data, columns=columns)
    df['y'] = sum(df[x] for x in columns)
    X = df.drop('y', axis=1)
    y = df['y']
    leaf_xranges, leaf_slopes, ignored = \
        slopes(X, y, colname=colname, min_samples_leaf=min_samples_leaf)

    expected = np.array(expected)
    expected_xranges = expected[:,0:1+1]
    expected_slopes = expected[:,2]

    print(leaf_xranges)
    print(leaf_slopes)
    assert len(leaf_xranges)==len(expected_slopes), f"Expected ranges {expected_xranges}"
    assert np.isclose(leaf_xranges, np.array(expected_xranges)).all(), f"Expected ranges {expected_xranges} got {leaf_xranges}"
    assert len(leaf_slopes)==len(expected_slopes), f"Expected slopes {expected_slopes}"
    assert np.isclose(leaf_slopes, np.array(expected_slopes)).all(), f"Expected slopes {expected_slopes} got {leaf_slopes}"

    # print(X.sort_values('x1').iloc[:8, :])
    # s = pd.DataFrame()
    # s['left'] = leaf_xranges[:, 0]
    # s['right'] = leaf_xranges[:, 1]
    # s['slope'] = leaf_slopes
    # print(s.head(10))


def test_2_discrete_records():
    data = [[1, 2],
            [1, 3]]
    expected = [[2, 3, 1]]
    check(data, expected, colname='x2')


def test_1_record_edge_case():
    data = [[1, 2]]
    expected = []
    check(data, expected, colname='x2')


def test_random_floating_point_all_in_one_leaf():
    data = \
        [[0.008386, 3.724396, 0.123684],
         [0.012942, 8.592633, 0.973965],
         [0.048403, 4.945038, 0.373470],
         [0.072278, 6.022741, 0.767953],
         [0.080743, 6.678219, 0.375508],
         [0.109459, 8.365492, 0.801016],
         [0.110229, 1.457290, 0.541546],
         [0.115092, 0.806182, 0.972837]]
    expected = \
        [[0.806182, 1.45729,  0.99253119],
         [1.45729,  3.724396, 0.95507797],
         [3.724396, 4.945038, 1.03278357],
         [4.945038, 6.022741, 1.0221536],
         [6.022741, 6.678219, 1.01291424],
         [6.678219, 8.365492, 1.01701918],
         [8.365492, 8.592633, 0.57507892]]
    # Put all into one leaf
    check(data, expected, colname='x2', min_samples_leaf=10)


def test_random_floating_point_in_2_leaves():
    data = \
        [[0.008386, 3.724396, 0.123684],
         [0.012942, 8.592633, 0.973965],
         [0.048403, 4.945038, 0.373470],
         [0.072278, 6.022741, 0.767953]]
    expected = \
        [[3.724396, 8.592633, 1.17559478],
         [4.945038, 6.022741, 1.38819415]]
    # Put all into one leaf
    check(data, expected, colname='x2', min_samples_leaf=2)
