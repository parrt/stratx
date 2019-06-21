import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from  matplotlib.collections import LineCollection
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import time
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from scipy.integrate import cumtrapz
import scipy.stats as stats
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
        leaf_samples.extend(sample_idxs_in_leaf) # add [...sample idxs...] for each leaf
    return leaf_samples


def bin_samples(rf, X:np.ndarray):
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
        leaf_samples.extend(sample_idxs_in_leaf) # add [...sample idxs...] for each leaf
    return leaf_samples


def dtree_leaf_samples(dtree, X:np.ndarray):
    leaf_ids = dtree.apply(X)
    d = pd.DataFrame(leaf_ids, columns=['leafid'])
    d = d.reset_index() # get 0..n-1 as column called index so we can do groupby
    sample_idxs_in_leaf = d.groupby('leafid')['index'].apply(lambda x: x.values)
    return sample_idxs_in_leaf


def discrete_xc_space(x: np.ndarray, y: np.ndarray, colname, verbose):
    start = time.time()

    ignored = 0
    xy = pd.concat([pd.Series(x), pd.Series(y)], axis=1)
    xy.columns = ['x', 'y']
    xy = xy.sort_values('x')
    df_avg = xy.groupby('x').mean().reset_index()
    x = df_avg['x'].values
    y = df_avg['y'].values
    uniq_x = np.unique(x)

    if len(uniq_x)==1:
        # print(f"ignore {len(x)} in discrete_xc_space")
        ignored += len(x)
        return np.array([]), np.array([]), np.array([]), np.array([]), ignored

    bin_deltas = np.diff(uniq_x)
    y_deltas = np.diff(y)
    leaf_slopes = y_deltas / bin_deltas  # "rise over run"
    leaf_xranges = np.array(list(zip(uniq_x, uniq_x[1:])))
    leaf_sizes = xy['x'].value_counts().sort_index().values
    leaf_r2 = np.full(shape=(len(uniq_x),), fill_value=np.nan) # unused

    # Now strip out elements for non-consecutive x
    if False:
        adj = np.where(bin_deltas == 1)[0]   # index of consecutive x values
        if len(adj)==0: # no consecutive?
            # print(f"ignore non-consecutive {len(x)} in discrete_xc_space")
            ignored += len(x)
            return np.array([]), np.array([]), np.array([]), np.array([]), ignored

        leaf_slopes = leaf_slopes[adj]
        leaf_xranges = leaf_xranges[adj]
        leaf_sizes = leaf_sizes[adj]
        leaf_r2 = leaf_r2[adj]

    stop = time.time()
    # print(f"discrete_xc_space {stop - start:.3f}s")
    return leaf_xranges, leaf_sizes, leaf_slopes, leaf_r2, ignored


def piecewise_xc_space(x: np.ndarray, y: np.ndarray, colname, nbins:int, verbose):
    start = time.time()

    ignored = 0
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []
    leaf_sizes = []

    # To get n bins, we need n+1 numbers in linear space
    domain = (np.min(x), np.max(x))
    bins = np.linspace(*domain, num=nbins+1, endpoint=True)
    binned_idx = np.digitize(x, bins)

    for i in range(1, len(bins)+1):
        bin_x = x[binned_idx == i]
        bin_y = y[binned_idx == i]
        if len(bin_x)<2: # either no or too little data
            # print(f"ignoring xleft=xright @ {r[0]}")
            ignored += len(bin_x)
            continue
        r = (np.min(bin_x), np.max(bin_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            ignored += len(bin_x)

        lm = LinearRegression()
        bin_x = bin_x.reshape(-1, 1)
        lm.fit(bin_x, bin_y)
        r2 = lm.score(bin_x, bin_y)

        leaf_sizes.append(len(bin_x))
        leaf_slopes.append(lm.coef_[0])
        leaf_xranges.append(r)
        leaf_r2.append(r2)

    if len(leaf_slopes)==0:
        # looks like binning was too fine and we didn't get any slopes
        # If y is evenly spread across integers, we might get single x value with lots of y,
        # which can't tell us about change in y over x as x isn't changing.
        # Fall back onto single line for whole leaf
        lm = LinearRegression()
        lm.fit(x.reshape(-1,1), y)
        leaf_slopes.append(lm.coef_[0])  # better to use univariate slope it seems
        r2 = lm.score(x.reshape(-1,1), y)
        leaf_r2.append(r2)
        r = (np.min(x), np.max(x))
        leaf_xranges.append(r)
        leaf_sizes.append(len(x))

    stop = time.time()
    # print(f"piecewise_xc_space {stop - start:.3f}s")
    return leaf_xranges, leaf_sizes, leaf_slopes, leaf_r2, ignored


def old_piecewise_xc_space(x: np.ndarray, y: np.ndarray, colname, hires_min_samples_leaf:int, verbose):
    start = time.time()
    X = x.reshape(-1,1)

    r2s = []

    # dbg = True
    dbg = False
    if dbg:
        print(f"\t{len(x)} samples")
        plt.scatter(x, y, c='black', s=.5)
        lm = LinearRegression()
        lm.fit(x.reshape(-1, 1), y)
        r2 = lm.score(x.reshape(-1, 1), y)
        px = np.linspace(min(x), max(x), 20)
        plt.plot(px, lm.predict(px.reshape(-1, 1)), lw=.5, c='red', label=f"R^2 {r2:.2f}")

    rf = RandomForestRegressor(n_estimators=1,
                               min_samples_leaf=hires_min_samples_leaf, # "percent" or number of samples allowed per leaf
                               max_features=1.0,
                               bootstrap=False)
    rf.fit(X, y)
    leaves = leaf_samples(rf, X)

    if verbose:
        print(f"Piecewise {colname}: {len(leaves)} leaves")

    ignored = 0
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []
    leaf_sizes = []
    for samples in leaves:
        leaf_x = X[samples]
        leaf_y = y[samples]
        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            ignored += len(samples)
            if verbose: print(f"\tIgnoring range {r} from {leaf_x.T[0:3]}... -> {leaf_y[0:3]}...")
            continue
        lm = LinearRegression()
        lm.fit(leaf_x.reshape(-1, 1), leaf_y)
        leaf_slopes.append(lm.coef_[0])
        r2 = lm.score(leaf_x.reshape(-1, 1), leaf_y)

        r2s.append(r2)
        if verbose:
            print(f"\tPiece {len(leaf_x)} obs, piecewise R^2 {r2:.2f}, R^2*n {r2*len(leaf_x):.2f}")
        if dbg:
            px = np.linspace(r[0], r[1], 20)
            plt.plot(px, lm.predict(px.reshape(-1,1)), lw=.5, c='blue', label=f"R^2 {r2:.2f}")

        leaf_r2.append(r2)
        leaf_xranges.append(r)
        leaf_sizes.append(len(samples))
        # leaf_sizes.append(1 / np.var(leaf_x))

    # print(f"\tAvg leaf R^2 {np.mean(r2s):.4f}, avg x len {np.mean(r2xNs)}")

    if verbose:
        print(f"\tIgnored {ignored} piecewise leaves")

    if dbg:
        plt.legend(loc='upper left', borderpad=0, labelspacing=0)
        plt.show()

    if len(leaf_slopes)==0:
        # looks like samples/leaf is too small and/or values are ints;
        # If y is evenly spread across integers, we will get single x value with lots of y,
        # which can't tell us about change in y over x as x isn't changing.
        # Fall back onto single line for whole leaf
        lm = LinearRegression()
        lm.fit(X, y)
        leaf_slopes.append(lm.coef_[0])  # better to use univariate slope it seems
        r2 = lm.score(X, y)
        leaf_r2.append(r2)
        r = (np.min(x), np.max(x))
        leaf_xranges.append(r)
        leaf_sizes.append(len(x))
        # leaf_sizes.append(1 / np.var(leaf_x))

    stop = time.time()
    # print(f"hires_slopes_from_one_leaf {stop - start:.3f}s")
    return leaf_xranges, leaf_sizes, leaf_slopes, leaf_r2, ignored


def collect_leaf_slopes(rf, X, y, colname, nbins, isdiscrete, verbose):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the samples then isolate the column of interest X values
    and the target y values. Perform another partition of X[colname] vs y and do
    piecewise linear regression to get the slopes in various regions of X[colname].
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the ranges of X[colname] partitions, num obs per leaf,
    associated slope for each range, r^2 of line through points.
    """
    start = time.time()
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []
    leaf_sizes = []

    ignored = 0

    leaves = leaf_samples(rf, X.drop(colname, axis=1))

    if verbose:
        nnodes = rf.estimators_[0].tree_.node_count
        print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
              f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    for samples in leaves:
        one_leaf_samples = X.iloc[samples]
        leaf_x = one_leaf_samples[colname].values
        leaf_y = y.iloc[samples].values

        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            ignored += len(leaf_x)
            continue

        if isdiscrete:
            leaf_xranges_, leaf_sizes_, leaf_slopes_, leaf_r2_, ignored_ = \
                discrete_xc_space(leaf_x, leaf_y, colname=colname, verbose=verbose)
        else:
            leaf_xranges_, leaf_sizes_, leaf_slopes_, leaf_r2_, ignored_ = \
                piecewise_xc_space(leaf_x, leaf_y, colname=colname, nbins=nbins, verbose=verbose)

        leaf_slopes.extend(leaf_slopes_)
        leaf_r2.extend(leaf_r2_)
        leaf_xranges.extend(leaf_xranges_)
        leaf_sizes.extend(leaf_sizes_)
        ignored += ignored_

    leaf_xranges = np.array(leaf_xranges)
    leaf_sizes = np.array(leaf_sizes)
    leaf_slopes = np.array(leaf_slopes)
    stop = time.time()
    if verbose: print(f"collect_leaf_slopes {stop - start:.3f}s")
    return leaf_xranges, leaf_sizes, leaf_slopes, leaf_r2, ignored


def avg_values_at_x(uniq_x, leaf_ranges, leaf_values, leaf_weights):
    """
    Compute the weighted average of leaf_values at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    start = time.time()
    nx = len(uniq_x)
    nslopes = len(leaf_values)
    slopes = np.zeros(shape=(nx, nslopes))
    weights = np.zeros(shape=(nx, nslopes))
    i = 0  # leaf index; we get a line for each leaf
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    for r, slope, w in zip(leaf_ranges, leaf_values, leaf_weights):
        s = np.full(nx, slope*w, dtype=float) # s has value*weight at all locations (flat line)
        # now trim line so it's only valid in range r;
        # don't set slope on right edge
        s[np.where( (uniq_x < r[0]) | (uniq_x >= r[1]) )] = np.nan
        slopes[:, i] = s
        # track weight (num obs in leaf) per range also so we can divide by total
        # obs per range to get weighted average below
        ws = np.full(nx, w, dtype=float)
        ws[np.where( (uniq_x < r[0]) | (uniq_x >= r[1]) )] = np.nan
        weights[:, i] = ws
        i += 1
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # avg_value_at_x = np.nanmean(slopes, axis=1)
    sum_values_at_x = np.nansum(slopes, axis=1)
    sum_weights_at_x = np.nansum(weights, axis=1)
    avg_value_at_x = sum_values_at_x / sum_weights_at_x
    stop = time.time()
    # print(f"avg_value_at_x {stop - start:.3f}s")
    return avg_value_at_x


def plot_stratpd(X, y, colname, targetname=None,
                 ax=None,
                 ntrees=1,
                 max_features = 1.0,
                 bootstrap=False,
                 min_samples_leaf=10,
                 nbins=3, # this is number of bins, so number of points in linear space is nbins+1
                          # ignored if isdiscrete; len(unique(X[colname])) used instead.
                          # must be >= 1
                 isdiscrete=False,
                 xrange=None,
                 yrange=None,
                 pdp_marker_size=2,
                 linecolor='#2c7fb8',
                 title=None,
                 nlines=None,
                 show_dx_line=False,
                 show_xlabel=True,
                 show_ylabel=True,
                 show_xticks=True,
                 connect_pdp_dots=False,
                 show_importance=False,
                 impcolor='#fdae61',
                 supervised=True,
                 alpha=.4,
                 verbose=False
                 ):
    # print(f"Unique {colname} = {len(np.unique(X[colname]))}/{len(X)}")
    if supervised:
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features)
        rf.fit(X.drop(colname, axis=1), y)
        if verbose:
            print(f"Strat Partition RF: missing {colname} training R^2 {rf.score(X.drop(colname, axis=1), y)}")

    else:
        """
        Wow. Breiman's trick works in most cases. Falls apart on Boston housing MEDV target vs AGE
        """
        if verbose: print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    real_uniq_x = np.array(sorted(np.unique(X[colname])))
    # print(f"\nModel wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_xranges, leaf_sizes, leaf_slopes, leaf_r2, ignored = \
        collect_leaf_slopes(rf, X, y, colname, nbins=nbins, isdiscrete=isdiscrete, verbose=verbose)
    if True:
        print(f"{'discrete ' if isdiscrete else ''}StratPD num samples ignored {ignored} for {colname}")


    slope_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_slopes, leaf_sizes)
    r2_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_r2, leaf_sizes)
    # Drop any nan slopes; implies we have no reliable data for that range
    # Make sure to drop uniq_x values too :)
    notnan_idx = ~np.isnan(slope_at_x) # should be same for slope_at_x and r2_at_x
    slope_at_x = slope_at_x[notnan_idx]
    uniq_x = real_uniq_x[notnan_idx]
    r2_at_x = r2_at_x[notnan_idx]
    # print(f'uniq_x = [{", ".join([f"{x:4.1f}" for x in uniq_x])}]')
    # print(f'slopes = [{", ".join([f"{s:4.1f}" for s in slope_at_x])}]')

    if len(uniq_x)==0:
        raise ValueError(f"Could not compute slopes for partial dependence curve; "
                             f"binning granularity is likely cause: nbins={nbins}, uniq x={len(real_uniq_x)}")

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

    if len(uniq_x) != len(curve):
        raise AssertionError(f"len(uniq_x) = {len(uniq_x)}, but len(curve) = {len(curve)}; nbins={nbins}")

    # plot partial dependence curve
    # cmap = cm.afmhot(Normalize(vmin=0, vmax=ignored))
    ax.scatter(uniq_x, curve, s=pdp_marker_size, c='k', alpha=1)

    if connect_pdp_dots:
        ax.plot(uniq_x, curve, ':',
                alpha=1,
                lw=1,
                c='grey')

    # widths = []
    segments = []
    for xr, slope in zip(leaf_xranges, leaf_slopes):
        w = np.abs(xr[1] - xr[0])
        # widths.append(w)
        y_delta = slope * (w)
        closest_x_i = np.abs(uniq_x - xr[0]).argmin() # find curve point for xr[0]
        y = curve[closest_x_i]
        one_line = [(xr[0],y), (xr[1], y+y_delta)]
        segments.append( one_line )

    # print(f"Avg width is {np.mean(widths):.2f} in {len(leaf_sizes)} leaves")

    if verbose:
        print(f"Found {len(segments)} lines")

    if nlines is not None:
        nlines = min(nlines, len(segments))
        idxs = np.random.randint(low=0, high=len(segments), size=nlines)
        segments = np.array(segments)[idxs]

    lines = LineCollection(segments, alpha=alpha, color=linecolor, linewidth=.5)
    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(min(uniq_x), max(uniq_x))
    if yrange is not None:
        ax.set_ylim(*yrange)
    ax.add_collection(lines)

    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)

    mx = np.max(uniq_x)
    if show_dx_line:
        r = LinearRegression()
        r.fit(uniq_x.reshape(-1,1), curve)
        x = np.linspace(np.min(uniq_x), mx, num=100)
        ax.plot(x, x * r.coef_[0] + r.intercept_, linewidth=1, c='orange')

    if show_importance:
        other = ax.twinx()
        other.set_ylim(0,1.0)
        other.tick_params(axis='y', colors=impcolor)
        other.set_ylabel("Feature importance", fontdict={"color":impcolor})
        other.plot(uniq_x, r2_at_x, lw=1, c=impcolor)
        a,b = ax.get_xlim()
        other.plot(b - (b-a) * .03, np.mean(r2_at_x), marker='>', c=impcolor)
        # other.plot(mx - (mx-mnx)*.02, np.mean(r2_at_x), marker='>', c=imp_color)

    return uniq_x, curve, r2_at_x, ignored


def plot_stratpd_gridsearch(X, y, colname, targetname,
                            min_samples_leaf_values=(2,5,10,20,30),
                            nbins_values=(1,2,3,4,5),
                            isdiscrete=False,
                            yrange=None,
                            show_regr_line=False):
    nrows = len(nbins_values)
    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(nrows, ncols + 1, figsize=((ncols + 1) * 2.5, nrows * 2.5))

    row = 0
    for i, nbins in enumerate(nbins_values):
        marginal_plot(X, y, colname, targetname, ax=axes[row, 0], show_regr_line=show_regr_line)
        col = 1
        for msl in min_samples_leaf_values:
            print(f"---------- min_samples_leaf={msl}, nbins={nbins:.2f} ----------- ")
            try:
                plot_stratpd(X, y, colname, targetname, ax=axes[row, col],
                             nbins=nbins,
                             min_samples_leaf=msl,
                             isdiscrete=isdiscrete,
                             yrange=yrange,
                             ntrees=1)
            except ValueError:
                axes[row, col].set_title(
                    f"Can't gen: leafsz={msl}, nbins={nbins}",
                    fontsize=8)
            else:
                axes[row, col].set_title(
                    f"leafsz={msl}, nbins={nbins}",
                    fontsize=9)
            col += 1
        row += 1


def marginal_plot(X, y, colname, targetname, ax, alpha=.1, show_regr_line=True):
    ax.scatter(X[colname], y, alpha=alpha, label=None)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    col = X[colname]

    if show_regr_line:
        r = LinearRegression()
        r.fit(X[[colname]], y)
        xcol = np.linspace(np.min(col), np.max(col), num=100)
        yhat = r.predict(xcol.reshape(-1, 1))
        ax.plot(xcol, yhat, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
        ax.text(min(xcol) * 1.02, max(y) * .95, f"$\\beta_{{{colname}}}$={r.coef_[0]:.3f}")


def catwise_leaves(rf, X, y, colname, verbose):
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
    ignored = 0
    catcol = X[colname].astype('category').cat.as_ordered()
    cats = catcol.cat.categories
    leaf_sizes = []
    leaf_histos = pd.DataFrame(index=cats)
    leaf_histos.index.name = 'category'
    ci = 0
    Xy = pd.concat([X, y], axis=1)
    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    for sample in leaves:
        combined = Xy.iloc[sample]
        # print("\n", combined)
        avg_cat_y = combined.groupby(colname).mean()
        avg_cat_y = avg_cat_y.iloc[:,-1]
        if len(avg_cat_y) < 2:
            # print(f"ignoring {len(sample)} obs for {len(avg_cat_y)} cat(s) in leaf")
            ignored += len(sample)
            continue
        min_avg = np.min(avg_cat_y)
        # record avg y value per cat in this leaf
        # This assignment copies cat y avgs to appropriate cat row using index
        # leaving cats w/o representation as nan
        leaf_histos['leaf' + str(ci)] = avg_cat_y - min_avg
        leaf_sizes.append(len(sample))
        ci += 1

    # print(leaf_histos)
    stop = time.time()
    if verbose: print(f"catwise_leaves {stop - start:.3f}s")
    return leaf_histos, leaf_sizes, ignored


# only works for ints, not floats
def plot_catstratpd(X, y, colname, targetname,
                    cats=None,
                    ax=None,
                    sort='ascending',
                    ntrees=1,
                    min_samples_leaf=10,
                    max_features=1.0,
                    bootstrap=False,
                    yrange=None,
                    title=None,
                    supervised=True,
                    alpha=.15,
                    color='#2c7fb8',
                    pdp_marker_size=.5,
                    marker_size=5,
                    pdp_color='black',
                    style:('strip','scatter')='strip',
                    show_xlabel=True,
                    show_ylabel=True,
                    show_xticks=True,
                    verbose=False):
    if ntrees==1:
        max_features = 1.0
        bootstrap = False

    if supervised:
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
        rf.fit(X.drop(colname, axis=1), y)
        if verbose:
            print(f"CatStrat Partition RF: missing {colname} training R^2 {rf.score(X.drop(colname, axis=1), y)}")
    else:
        print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    # rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_histos, leaf_sizes, ignored = catwise_leaves(rf, X, y, colname, verbose=verbose)

    if True:
        print(f"CatStratPD Num samples ignored {ignored} for {colname}")

    # weighted_histos = leaf_histos.mul(leaf_sizes)
    # weighted_sum_per_cat = np.nansum(weighted_histos, axis=1)
    # total_obs_in_leaves = np.sum(leaf_sizes)
    # avg_per_cat = weighted_sum_per_cat / total_obs_in_leaves

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

    # TODO: shouldn't have to do this using ref cat
    min_avg_value = np.min(avg_per_cat)

    # if too many categories, can't do strip plot
    if style=='strip':
        xloc = 1
        sigma = .02
        mu = 0
        x_noise = np.random.normal(mu, sigma, size=nleaves) # to make strip plot
        for i in sort_indexes:
            ax.scatter(x_noise + xloc, leaf_histos.iloc[i] - min_avg_value,
                       alpha=alpha, marker='o', s=marker_size,
                       c=color)
            ax.plot([xloc - .1, xloc + .1], [avg_per_cat[i]-min_avg_value] * 2,
                    c='black', linewidth=2)
            xloc += 1
    else: # do straight plot
        xlocs = np.arange(1,ncats+1)
        """
        
                       leaf0       leaf1
        category
        1         166.430176  186.796956
        2         219.590349  176.448626
        """
        sorted_histos = leaf_histos.iloc[sort_indexes,:]
        xloc = 1
        for i in range(nleaves):
            ax.scatter(xlocs, sorted_histos.iloc[:,i] - min_avg_value,
                       alpha=alpha, marker='o', s=marker_size,
                       c=color)
            xloc += 1
        ax.scatter(xlocs, avg_per_cat[sort_indexes] - min_avg_value, c=pdp_color, s=pdp_marker_size)

    ax.set_xticks(range(1, ncats + 1))
    if show_xticks: # sometimes too many
        ax.set_xticklabels(cats)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False)

    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)

    if yrange is not None:
        ax.set_ylim(*yrange)

    return cats, avg_per_cat[sort_indexes]-min_avg_value, ignored


# -------------- B I N N I N G ---------------

# TODO: can we delete all this section?

def hires_slopes_from_one_leaf_h(x:np.ndarray, y:np.ndarray, h:float):
    """
    Split x range into bins of width h and return
    """
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []

    uniq_x = np.array(sorted(np.unique(x)))
    # print(f"uniq x {uniq_x}")
    for ix in uniq_x:
        bin_x = x[(x >= ix) & (x < ix + h)]
        bin_y = y[(x >= ix) & (x < ix + h)]
        print()
        print(bin_x)
        print(bin_y)
        if len(bin_x)==0:
            continue
        r = (np.min(bin_x), np.max(bin_y))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            continue

        lm = LinearRegression()
        lm.fit(bin_x.reshape(-1, 1), bin_y)
        r2 = lm.score(bin_x.reshape(-1, 1), bin_y)

        leaf_slopes.append(lm.coef_[0])
        leaf_xranges.append(r)
        leaf_r2.append(r2)

    return leaf_xranges, leaf_slopes, leaf_r2

def hires_slopes_from_one_leaf_nbins(x:np.ndarray, y:np.ndarray, nbins:int):
    """
    Split x range into bins of width h and return
    """
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []

    bins = np.linspace(np.min(x), np.max(x), num=nbins, endpoint=True)
    binned_idx = np.digitize(x, bins)

    for i in range(1, nbins+1):
        bin_x = x[binned_idx == i]
        bin_y = y[binned_idx == i]
        if len(bin_x)==0:
            continue
        r = (np.min(bin_x), np.max(bin_y))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            continue

        lm = LinearRegression()
        lm.fit(bin_x.reshape(-1, 1), bin_y)
        r2 = lm.score(bin_x.reshape(-1, 1), bin_y)

        leaf_slopes.append(lm.coef_[0])
        leaf_xranges.append(r)
        leaf_r2.append(r2)

    return leaf_xranges, leaf_slopes, leaf_r2

def do_my_binning(x:np.ndarray, y:np.ndarray, h:float):
    """
    Split x range into bins of width h from X[colname] space
    """
    leaf_bin_avgs = []

    uniq_x = np.array(sorted(np.unique(x)))
    # print(f"uniq x {uniq_x}")
    for ix in uniq_x:
        bin_x = x[(x >= ix) & (x < ix + h)]
        bin_y = y[(x >= ix) & (x < ix + h)]
        print()
        print(bin_x)
        print(bin_y)
        if len(bin_x)==0:
            continue
        r = (np.min(bin_x), np.max(bin_y))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            continue

        leaf_bin_avgs.append(lm.coef_[0])

    return leaf_bin_avgs

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
