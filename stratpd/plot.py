import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from  matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import time
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from scipy.integrate import cumtrapz
from dtreeviz.trees import *


def leaf_samples(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest.
    """
    ntrees = len(rf.estimators_)
    leaf_ids = rf.apply(X) # which leaf does each X_i go to for each tree?
    d = pd.DataFrame(leaf_ids, columns=[f"tree{i}" for i in range(ntrees)])
    d = d.reset_index() # get 0..n-1 as column called index so we can do groupby
    """
    d looks like:
        index	tree0	tree1	tree2	tree3	tree4
    0	0	    8	    3	    4	    4	    3
    1	1	    8	    3	    4	    4	    3
    """
    leaf_samples = []
    for i in range(ntrees):
        """
        Each groupby gets a list of all X indexes associated with same leaf. 4 leaves would
        get 4 arrays of X indexes; e.g.,
        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
               array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
               array([21, 22, 23, 24, 25, 26, 27, 28, 29]), ... )
        """
        sample_idxs_in_leaf = d.groupby(f'tree{i}')['index'].apply(lambda x: x.values)
        if len(sample_idxs_in_leaf) >= 2:
            # can't detect changes with just one sample
            leaf_samples.extend(sample_idxs_in_leaf)
    return leaf_samples


def dtree_leaf_samples(dtree, X:np.ndarray):
    leaf_ids = dtree.apply(X)
    d = pd.DataFrame(leaf_ids, columns=['leafid'])
    d = d.reset_index() # get 0..n-1 as column called index so we can do groupby
    sample_idxs_in_leaf = d.groupby('leafid')['index'].apply(lambda x: x.values)
    return sample_idxs_in_leaf


def hires_slopes_from_one_leaf(x:np.ndarray, y:np.ndarray, hires_min_samples_leaf:int):
    start = time.time()
    X = x.reshape(-1,1)
    """
    Bootstrapping appears to be important, giving much better sine curve for weather().
    Tried many diff min_samples_leaf and min_samples_leaf=10 is largest value I feel
    comfortable with and gives good sine curve. Gotta keep cost down here as we might
    call this a lot.
    
    Tried single estimator w/o boostrap. Terrible.
    """
    rf = RandomForestRegressor(n_estimators=1,
                               min_samples_leaf=hires_min_samples_leaf,
                               max_features=1.0,
                               bootstrap=False,
                               )
    rf.fit(X, y)
    leaves = leaf_samples(rf, X)
    leaf_slopes = []
    leaf_xranges = []
    leaf_yranges = []
    for samples in leaves:
        leaf_x = X[samples]
        leaf_y = y[samples]
        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            continue
        lm = LinearRegression()
        lm.fit(leaf_x.reshape(-1, 1), leaf_y)
        leaf_slopes.append(lm.coef_[0])
        leaf_xranges.append(r)
        leaf_yranges.append((leaf_y[0], leaf_y[-1]))
    stop = time.time()
    # print(f"hires_slopes_from_one_leaf {stop - start:.3f}s")
    return leaf_xranges, leaf_yranges, leaf_slopes


def collect_leaf_slopes(rf, X, y, colname, hires_threshold, hires_min_samples_leaf):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the samples then isolate the column of interest X values
    and the target y values. Perform a regression to get the slope of X[colname] vs y.
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the range of X[colname] and associated slope for that range.
    """
    start = time.time()
    leaf_slopes = []
    leaf_xranges = []
    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    for samples in leaves:
        one_leaf_samples = X.iloc[samples]
        leaf_x = one_leaf_samples[colname].values
        leaf_y = y.iloc[samples].values
        if len(samples)>hires_threshold:
            #print(f"BIG {len(samples)}!!!")
            leaf_xranges_, leaf_yranges_, leaf_slopes_ = \
                hires_slopes_from_one_leaf(leaf_x, leaf_y, hires_min_samples_leaf)
            leaf_slopes.extend(leaf_slopes_)
            leaf_xranges.extend(leaf_xranges_)
            continue

        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            continue
        lm = LinearRegression()
        lm.fit(leaf_x.reshape(-1, 1), leaf_y)
        leaf_slopes.append(lm.coef_[0])
        leaf_xranges.append(r)
    leaf_slopes = np.array(leaf_slopes)
    leaf_xranges = np.array(leaf_xranges)
    stop = time.time()
    print(f"collect_leaf_slopes {stop - start:.3f}s")
    return leaf_xranges, leaf_slopes


def avg_slope_at_x(uniq_x, leaf_ranges, leaf_slopes):
    """
    Slope at max(x) is NaN since we have not data beyond that point.
    :param leaf_ranges:
    :param leaf_slopes:
    :return:
    """
    start = time.time()
    nx = len(uniq_x)
    nslopes = len(leaf_slopes)
    slopes = np.zeros(shape=(nx, nslopes))
    i = 0  # leaf index; we get a line for each leaf
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    for r, slope in zip(leaf_ranges, leaf_slopes):
        s = np.full(nx, slope) # s has value scope at all locations (flat line)
        # now trim line so it's only valid in range r
        s[np.where(uniq_x < r[0])] = np.nan
        s[np.where(uniq_x >= r[1])] = np.nan # don't set slope on right edge
        slopes[:, i] = s
        i += 1
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    avg_slope_at_x = np.nanmean(slopes, axis=1)

    stop = time.time()
    # print(f"avg_slope_at_x {stop - start:.3f}s")
    return avg_slope_at_x


def stratpd_plot(X, y, colname, targetname=None,
                 ax=None,
                 ntrees=1,
                 min_samples_leaf=2,
                 hires_min_samples_leaf=5,
                 alpha=.3,
                 hires_threshold=30,
                 xrange=None,
                 yrange=None,
                 pdp_dot_size=5,
                 title=None,
                 nlines=None,
                 show_dx_line=False,
                 supervised=True):
    """

    :param X:
    :param y:
    :param colname:
    :param targetname:
    :param ax:
    :param ntrees:
    :param min_samples_leaf:
    :param alpha:
    :param hires_threshold:
    :param xrange:
    :param yrange:
    :param show_derivative:
    :return:
    """

    # print(f"Unique {colname} = {len(np.unique(X[colname]))}/{len(X)}")
    if supervised:
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = False,
                                   max_features = 1.0,
                                   oob_score=False)
        rf.fit(X.drop(colname, axis=1), y)
    else:
        """
        Wow. Breiman's trick works in most cases. Falls apart on Boston housing MEDV target vs AGE
        """
        # print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = False,
                                   max_features = 1.0,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    # print(f"\nModel wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_xranges, leaf_slopes = \
        collect_leaf_slopes(rf, X, y, colname, hires_threshold=hires_threshold,
                            hires_min_samples_leaf=hires_min_samples_leaf)
    uniq_x = sorted(np.unique(X[colname]))
    slope_at_x = avg_slope_at_x(uniq_x, leaf_xranges, leaf_slopes)
    # print(f'uniq_x = [{", ".join([f"{x:4.1f}" for x in uniq_x])}]')
    # print(f'slopes = [{", ".join([f"{s:4.1f}" for s in slope_at_x])}]')

    if ax is None:
        fig, ax = plt.subplots(1,1)

    # print(f"diff: {np.diff(uniq_x)}")
    dx = slope_at_x[:-1] * np.diff(uniq_x)          # last slope is nan since no data after last x value
    # print(f"dx: {dx}")
    curve = np.cumsum(dx)                           # we lose one value here
    # curve = cumtrapz(slope_at_x, x=uniq_x)          # we lose one value here
    curve = np.concatenate([np.array([0]), curve])  # add back the 0 we lost
    # print(slope_at_x, len(slope_at_x))
    # print(dx)
    # print(uniq_x, len(uniq_x))
    # print(curve, len(curve))

    # if 0 is in x feature and not on left/right edge, get y at 0
    # and shift so that is x,y 0 point.
    # nx = len(uniq_x)
    # if uniq_x[int(nx*0.05)]<0 or uniq_x[-int(nx*0.05)]>0:
    #     closest_x_to_0 = np.abs(uniq_x - 0.0).argmin()
    #     y = curve[closest_x_to_0]
    #     curve -= y  # shift
    # Nah. starting with 0 is best

    ax.scatter(uniq_x, curve,
               s=pdp_dot_size, alpha=1,
               c='black', label="Avg piecewise linear")

    if show_dx_line:
        ax.plot(uniq_x, curve, ':',
                   alpha=1,
                   lw=1,
                   c='grey')

    segments = []
    for xr, slope in zip(leaf_xranges, leaf_slopes):
        y_delta = slope * (xr[1] - xr[0])
        closest_x_i = np.abs(uniq_x - xr[0]).argmin() # find curve point for xr[0]
        y = curve[closest_x_i]
        one_line = [(xr[0],y), (xr[1], y+y_delta)]
        segments.append( one_line )

    if nlines is not None:
        idxs = np.random.randint(low=0, high=len(segments), size=nlines)
        segments = np.array(segments)[idxs]

    lines = LineCollection(segments, alpha=alpha, color='#9CD1E3', linewidth=.5)
    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(min(uniq_x),max(uniq_x))
    if yrange is not None:
        ax.set_ylim(*yrange)
    ax.add_collection(lines)

    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)


def catwise_leaves(rf, X, y, colname):
    """
    Return a dataframe with the average y value for each category in each leaf
    normalized by subtracting min avg y value from all categories.
    The index has the complete category list. The columns are the y avg value changes
    found in a single leaf. Each row represents a category level. E.g.,

                       leaf0       leaf1
        category
        1         166.430176  186.796956
        2         219.590349  176.448626
    """
    start = time.time()
    catcol = X[colname].astype('category').cat.as_ordered()
    cats = catcol.cat.categories
    leaf_histos = pd.DataFrame(index=cats)
    leaf_histos.index.name = 'category'
    ci = 0
    Xy = pd.concat([X, y], axis=1)
    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    for samples in leaves:
        combined = Xy.iloc[samples]
        # print("\n", combined)
        min_y = np.min(combined.iloc[:, -1])
        avg_cat_y = combined.groupby(colname).mean()
        avg_cat_y = avg_cat_y.iloc[:,-1]
        if len(avg_cat_y) < 2:
            # print(f"ignoring len {len(avg_cat_y)} cat leaf")
            continue
        # record how much bump or drop we get per category above
        # minimum change seen by any category (works even when all are negative)
        # This assignment copies cat bumps to appropriate cat row using index
        # leaving cats w/o representation as nan
        relative_changes_per_cat = avg_cat_y - min_y
        leaf_histos['leaf' + str(ci)] = relative_changes_per_cat
        ci += 1

    # print(leaf_histos)
    stop = time.time()
    print(f"catwise_leaves {stop - start:.3f}s")
    return leaf_histos


def catstratpd_plot(X, y, colname, targetname,
                    cats=None,
                    ax=None,
                    sort='ascending',
                    ntrees=1,
                    min_samples_leaf=None,
                    alpha=.03,
                    yrange=None,
                    title=None):
    if min_samples_leaf is None:
        # rule of thumb: for binary, 2 samples / leaf seems good
        # but num cats + 3 seems better for non-binary
        min_samples_leaf = len(np.unique(X[colname]))
        if min_samples_leaf>2:
            min_samples_leaf += 3

    rf = RandomForestRegressor(n_estimators=ntrees,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap = False,
                               max_features = 1.0,
                               oob_score=False)

    # rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_histos = catwise_leaves(rf, X, y, colname)
    avg_per_cat = np.nanmean(leaf_histos, axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ncats = len(cats)
    nleaves = len(leaf_histos.columns)

    sort_indexes = range(ncats)
    if sort == 'ascending':
        sort_indexes = avg_per_cat.argsort()
        cats = cats[sort_indexes]
    elif sort == 'descending':
        sort_indexes = avg_per_cat.argsort()[::-1]  # reversed
        cats = cats[sort_indexes]

    min_value = np.min(avg_per_cat)

    xloc = 1
    sigma = .02
    mu = 0
    x_noise = np.random.normal(mu, sigma, size=nleaves) # to make strip plot
    for i in sort_indexes:
        ax.scatter(x_noise + xloc, leaf_histos.iloc[i]-min_value,
                   alpha=alpha, marker='o', s=10,
                   c='#9CD1E3')
        ax.plot([xloc - .1, xloc + .1], [avg_per_cat[i]-min_value] * 2,
                c='black', linewidth=2)
        xloc += 1
    ax.set_xticks(range(1, ncats + 1))
    ax.set_xticklabels(cats)

    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)

    if yrange is not None:
        ax.set_ylim(*yrange)

# -------------- S U P P O R T ---------------


def scramble(X : np.ndarray) -> np.ndarray:
    """
    From Breiman: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    "...the first coordinate is sampled from the N values {x(1,n)}. The second
    coordinate is sampled independently from the N values {x(2,n)}, and so forth."
    """
    X_rand = X.copy()
    ncols = X.shape[1]
    for col in range(ncols):
        X_rand[:,col] = np.random.choice(np.unique(X[:,col]), len(X), replace=True)
    return X_rand


def df_scramble(X : pd.DataFrame) -> pd.DataFrame:
    """
    From Breiman: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    "...the first coordinate is sampled from the N values {x(1,n)}. The second
    coordinate is sampled independently from the N values {x(2,n)}, and so forth."
    """
    X_rand = X.copy()
    for colname in X:
        X_rand[colname] = np.random.choice(X[colname].unique(), len(X), replace=True)
    return X_rand


def conjure_twoclass(X):
    """
    Make new data set 2x as big with X and scrambled version of it that
    destroys structure between features. Old is class 0, scrambled is class 1.
    """
    if isinstance(X, pd.DataFrame):
        X_rand = df_scramble(X)
        X_synth = pd.concat([X, X_rand], axis=0)
    else:
        X_rand = scramble(X)
        X_synth = np.concatenate([X, X_rand], axis=0)
    y_synth = np.concatenate([np.zeros(len(X)),
                              np.ones(len(X_rand))], axis=0)
    return X_synth, pd.Series(y_synth)
