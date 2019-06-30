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
from scipy.stats import binned_statistic

import time
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype
from scipy.integrate import cumtrapz
import scipy.stats as stats
import inspect
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


def collect_point_betas(X, y, colname, leaves, nbins:int):
    ignored = 0
    leaf_xranges = []
    leaf_slopes = []
    point_betas = np.full(shape=(len(X),), fill_value=np.nan)

    for samples in leaves: # samples is set of obs indexes that live in a single leaf
        leaf_all_x = X.iloc[samples]
        leaf_x = leaf_all_x[colname].values
        leaf_y = y.iloc[samples].values
        # Right edge of last bin is max(leaf_x) but that means we ignore the last value
        # every time. Tweak domain right edge a bit so max(leaf_x) falls in last bin.
        last_bin_extension = 0.0000001
        domain = (np.min(leaf_x), np.max(leaf_x)+last_bin_extension)
        bins = np.linspace(*domain, num=nbins+1, endpoint=True)
        binned_idx = np.digitize(leaf_x, bins) # bin number for values in leaf_x
        for b in range(1, len(bins)+1):
            bin_x = leaf_x[binned_idx == b]
            bin_y = leaf_y[binned_idx == b]
            if len(bin_x) < 2: # could be none or 1 in bin
                ignored += len(bin_x)
                continue
            r = (np.min(bin_x), np.max(bin_x))
            if len(bin_x)<2 or np.isclose(r[0], r[1]):
    #             print(f'ignoring {bin_x} -> {bin_y} for same range')
                ignored += len(bin_x)
                continue
            lm = LinearRegression()
            leaf_obs_idx_for_bin = np.nonzero((leaf_x>=bins[b-1]) &(leaf_x<bins[b]))
            obs_idx = samples[leaf_obs_idx_for_bin]
            lm.fit(bin_x.reshape(-1, 1), bin_y)
            point_betas[obs_idx] = lm.coef_[0]
            leaf_slopes.append(lm.coef_[0])
            leaf_xranges.append(r)

    leaf_slopes = np.array(leaf_slopes)
    return leaf_xranges, leaf_slopes, point_betas, ignored


def plot_stratpd(X, y, colname, targetname,
                 ntrees=1, min_samples_leaf=10, bootstrap=False,
                 max_features=1.0,
                 nbins=3,  # piecewise binning
                 nbins_smoothing=None,  # binning of overall X[colname] space in plot
                 isdiscrete=False,
                 ax=None,
                 xrange=None,
                 yrange=None,
                 title=None,
                 show_xlabel=True,
                 show_ylabel=True,
                 show_pdp_line=True,
                 show_slope_lines=True,
                 pdp_marker_size=5,
                 pdp_line_width=.5,
                 slope_line_color='#2c7fb8',
                 slope_line_width=.5,
                 slope_line_alpha=.3,
                 pdp_line_color='black',
                 pdp_marker_color='black'
                 ):

    if isdiscrete:
        return plot_discrete_stratpd(**locals()) # delegate

    rf = RandomForestRegressor(n_estimators=ntrees,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               max_features=max_features)
    rf.fit(X.drop(colname, axis=1), y)
    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    nnodes = rf.estimators_[0].tree_.node_count
    print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
          f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    leaf_xranges, leaf_slopes, point_betas, ignored = \
        collect_point_betas(X, y, colname, leaves, nbins)
    Xbetas = np.vstack([X[colname].values, point_betas]).T # get x_c, beta matrix
    # Xbetas = Xbetas[Xbetas[:,0].argsort()] # sort by x coordinate (not needed)

    print(f"{'discrete ' if isdiscrete else ''}StratPD num samples ignored {ignored}/{len(X)} for {colname}")

    x = Xbetas[:, 0]
    domain = (np.min(x), np.max(x))  # ignores any max(x) points as no slope info after that
    if nbins_smoothing is None:
        # use all unique values as bin edges if no bin width
        bins_smoothing = np.array(sorted(np.unique(x)))
    else:
        bins_smoothing = np.linspace(*domain, num=nbins_smoothing + 1, endpoint=True)

    noinfo = np.isnan(Xbetas[:, 1])
    Xbetas = Xbetas[~noinfo]

    avg_slopes_per_bin, _, _ = binned_statistic(x=Xbetas[:, 0], values=Xbetas[:, 1],
                                                bins=bins_smoothing, statistic='mean')

    # beware: avg_slopes_per_bin might have nan for empty bins
    bin_deltas = np.diff(bins_smoothing)
    delta_ys = avg_slopes_per_bin * bin_deltas  # compute y delta across bin width to get up/down bump for this bin

    print('bins_smoothing', bins_smoothing, ', deltas', bin_deltas)
    print('avgslopes', delta_ys)

    # manual cumsum
    delta_ys = np.concatenate([np.array([0]), delta_ys])  # we start at 0 for min(x)
    plot_x = []
    plot_y = []
    cumslope = 0.0
    # delta_ys_ = np.concatenate([np.array([0]), delta_ys])  # we start at 0 for min(x)
    for x, slope in zip(bins_smoothing, delta_ys):
        if np.isnan(slope):
            print(f"{x:5.3f},{cumslope:5.1f},{slope:5.1f} SKIP")
            continue
        cumslope += slope
        plot_x.append(x)
        plot_y.append(cumslope)
        print(f"{x:5.3f},{cumslope:5.1f},{slope:5.1f}")
    plot_x = np.array(plot_x)
    plot_y = np.array(plot_y)

    # PLOT

    if ax is None:
        fig, ax = plt.subplots(1,1)

    # Draw bin left edge markers; ignore bins with no data (nan)
    ax.scatter(plot_x, plot_y,
               s=pdp_marker_size, c=pdp_marker_color)

    if show_pdp_line:
        ax.plot(plot_x, plot_y,
                lw=pdp_line_width, c=pdp_line_color)

    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(*domain)
    if yrange is not None:
        ax.set_ylim(*yrange)

    if show_slope_lines:
        segments = []
        for xr, slope in zip(leaf_xranges, leaf_slopes):
            w = np.abs(xr[1] - xr[0])
            delta_y = slope * w
            closest_x_i = np.abs(plot_x - xr[0]).argmin() # find curve point for xr[0]
            closest_x = plot_x[closest_x_i]
            closest_y = plot_y[closest_x_i]
            one_line = [(closest_x, closest_y), (closest_x+w, closest_y + delta_y)]
            segments.append( one_line )

        # if nlines is not None:
        #     nlines = min(nlines, len(segments))
        #     idxs = np.random.randint(low=0, high=len(segments), size=nlines)
        #     segments = np.array(segments)[idxs]

        lines = LineCollection(segments, alpha=slope_line_alpha, color=slope_line_color, linewidths=slope_line_width)
        ax.add_collection(lines)

    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)

    return leaf_xranges, leaf_slopes, Xbetas, plot_x, plot_y, ignored


def plot_discrete_stratpd(X, y, colname, targetname,
                          ntrees=1, min_samples_leaf=10, bootstrap=False,
                          max_features=1.0,
                          nbins=3,  # piecewise binning
                          nbins_smoothing=None,
                          # binning of overall X[colname] space in plot
                          isdiscrete=False,
                          ax=None,
                          xrange=None,
                          yrange=None,
                          title=None,
                          show_xlabel=True,
                          show_ylabel=True,
                          show_pdp_line=True,
                          show_slope_lines=True,
                          pdp_marker_size=5,
                          pdp_line_width=.5,
                          slope_line_color='#2c7fb8',
                          slope_line_width=.5,
                          slope_line_alpha=.3,
                          pdp_line_color='black',
                          pdp_marker_color='black'
                          ):
    rf = RandomForestRegressor(n_estimators=ntrees,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               max_features=max_features)
    rf.fit(X.drop(colname, axis=1), y)
    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    nnodes = rf.estimators_[0].tree_.node_count
    print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
          f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    if False: # fast for positive integers
        leaf_xranges, leaf_slopes, bin_betas, bin_counts = \
            collect_posint_betas(X, y, colname, leaves)
        # print(bin_counts)

        avg_slopes_per_bin = np.nanmean(bin_betas, axis=1)

        maxx = max(X[colname])

        avg_slopes_per_bin_ = np.concatenate(
            [np.array([0]), avg_slopes_per_bin[:-1]])  # drop last one and put 0 at front
        print(avg_slopes_per_bin_)
        pdpx = [] # track x,y together in case something y is nan
        pdpy = []
        cumslope = 0.0
        for x, slope in zip(np.arange(0, maxx + 1), avg_slopes_per_bin_):
            if np.isnan(slope):
                print(f"{x:5.3f},{cumslope:5.1f},{slope:5.1f} SKIP")
                continue
            cumslope += slope
            pdpx.append(x)
            pdpy.append(cumslope)
            print(f"{x:5.3f},{cumslope:5.1f},{slope:5.1f}")
        pdpx = np.array(pdpx)
        pdpy = np.array(pdpy)
    else:
        leaf_xranges, leaf_sizes, leaf_slopes, ignored = \
            collect_discrete_slopes(rf, X, y, colname)

        print('leaf_xranges', leaf_xranges)
        print('leaf_slopes', leaf_slopes)


        real_uniq_x = np.array(sorted(np.unique(X[colname])))
        if True:
            print(f"{'discrete ' if isdiscrete else ''}StratPD num samples ignored {ignored}/{len(X)} for {colname}")

        slope_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_slopes)
        # Drop any nan slopes; implies we have no reliable data for that range
        # Make sure to drop uniq_x values too :)
        notnan_idx = ~np.isnan(slope_at_x) # should be same for slope_at_x and r2_at_x
        slope_at_x = slope_at_x[notnan_idx]
        pdpx = real_uniq_x[notnan_idx]

        y_deltas = slope_at_x[:-1] * np.diff(pdpx)    # last slope is nan since no data after last x value
        # print(f"y_deltas: {y_deltas}")
        pdpy = np.cumsum(y_deltas)                    # we lose one value here
        pdpy = np.concatenate([np.array([0]), pdpy])  # add back the 0 we lost

    # PLOT

    if ax is None:
        fig, ax = plt.subplots(1,1)

    # Draw bin left edge markers; ignore bins with no data (nan)
    ax.scatter(pdpx, pdpy,
               s=pdp_marker_size, c=pdp_marker_color)

    if show_pdp_line:
        ax.plot(pdpx, pdpy,
                lw=pdp_line_width, c=pdp_line_color)

    domain = (np.min(X[colname]), np.max(X[colname]))  # ignores any max(x) points as no slope info after that
    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(*domain)
    if yrange is not None:
        ax.set_ylim(*yrange)

    if show_slope_lines:
        segments = []
        for xr, slope in zip(leaf_xranges, leaf_slopes):
            w = np.abs(xr[1] - xr[0])
            delta_y = slope * w
            closest_x_i = np.abs(pdpx - xr[0]).argmin() # find curve point for xr[0]
            closest_x = pdpx[closest_x_i]
            closest_y = pdpy[closest_x_i]
            one_line = [(closest_x, closest_y), (closest_x+w, closest_y + delta_y)]
            segments.append( one_line )

        # if nlines is not None:
        #     nlines = min(nlines, len(segments))
        #     idxs = np.random.randint(low=0, high=len(segments), size=nlines)
        #     segments = np.array(segments)[idxs]

        lines = LineCollection(segments, alpha=slope_line_alpha, color=slope_line_color, linewidths=slope_line_width)
        ax.add_collection(lines)

    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)


def collect_posint_betas(X, y, colname, leaves):
    """
    Only works for positive integers, not floats, not negatives. It's faster than generic
    version.
    :param X:
    :param y:
    :param colname:
    :param leaves:
    :return:
    """

    # TODO: actually am I assuming consecutive x values by storing in matrix
    # by x start location of slope? probably.

    maxx = max(X[colname])
    leaf_slopes = []
    leaf_xranges = []
    bin_betas = np.full(shape=(maxx + 1, len(leaves)), fill_value=np.nan)
    bin_counts = np.zeros(shape=(maxx + 1, len(leaves)))
    for li, samples in enumerate(
        leaves):  # samples is set of obs indexes that live in a single leaf
        leaf_all_x = X.iloc[samples]
        leaf_x = leaf_all_x[colname].values
        leaf_y = y.iloc[samples].values

        bcount = np.bincount(leaf_x)
        bsum = np.bincount(leaf_x, weights=leaf_y)  # sum ys for each values of x
        binavgs = bsum / bcount
        bins = np.nonzero(bsum)[0]
        binavgs = binavgs[bins]
        #     print()
        #     print('leaf_x',list(leaf_x))
        #     print('bcount',bcount)
        #     print('bsum',bsum)
        #     print('bins', bins)
        #     print('binavgs',binavgs)

        bin_deltas = np.diff(bins)
        y_deltas = np.diff(binavgs)
        leaf_bin_slopes = y_deltas / bin_deltas  # "rise over run"
        leaf_slopes.extend(leaf_bin_slopes)
        leaf_bin_xranges = np.array(list(zip(bins, bins[1:])))
        leaf_xranges.extend(leaf_bin_xranges)
        #     print('bin_deltas',bin_deltas)
        #     print('y_deltas', y_deltas)
        #     print('leaf_slopes',leaf_slopes)
        #     print(leaf_xranges)
        leaf_betas = np.full(shape=(maxx + 1,), fill_value=np.nan)
        leaf_betas[bins[:-1]] = leaf_bin_slopes
        bin_betas[:, li] = leaf_betas
        leaf_counts = np.zeros(shape=(maxx + 1,))
        leaf_counts[bins] = bcount[bins]
        bin_counts[:, li] = leaf_counts
    # print('bin_betas', bin_betas)
    # print('bin_counts', bin_counts)
    return leaf_xranges, leaf_slopes, bin_betas, bin_counts


def discrete_xc_space(x: np.ndarray, y: np.ndarray, colname, verbose):
    """
    Use the categories within a leaf as the bins to dynamically change the bins,
    rather then using a fixed nbins hyper parameter. Group the leaf x,y by x
    and collect the average y.  The unique x and y averages are the new x and y pairs.
    The slope for each x is:

        (y_{i+1} - y_i) / (x_{i+1} - x_i)

    If the ordinal/ints are exactly one unit part, then it's just y_{i+1} - y_i. If
    they are not consecutive, we do not ignore isolated x_i as it ignores too much data.
    E.g., if x is [1,3,4] and y is [9,8,10] then the second x coordinate is skipped.
    The two slopes are [(8-9)/2, (10-8)/1] and bin widths are [2,1].

    If there is exactly one category in the leaf, the leaf provides no information
    about how the categories contribute to changes in y. We have to ignore this leaf.
    """
    start = time.time()

    ignored = 0
    xy = pd.concat([pd.Series(x), pd.Series(y)], axis=1)
    xy.columns = ['x', 'y']
    xy = xy.sort_values('x')
    df_avg = xy.groupby('x').mean().reset_index()
    x = df_avg['x'].values
    y = df_avg['y'].values
    uniq_x = x

    if len(uniq_x)==1:
        # print(f"ignore {len(x)} in discrete_xc_space")
        ignored += len(x)
        return np.array([]), np.array([]), np.array([]), np.array([]), ignored

    bin_deltas = np.diff(uniq_x)
    y_deltas = np.diff(y)
    leaf_slopes = y_deltas / bin_deltas  # "rise over run"
    leaf_xranges = np.array(list(zip(uniq_x, uniq_x[1:])))
    leaf_sizes = xy['x'].value_counts().sort_index().values

    stop = time.time()
    # print(f"discrete_xc_space {stop - start:.3f}s")
    return leaf_xranges, leaf_sizes, leaf_slopes, None, ignored


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
        # TODO: adjust ignored variable
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


def collect_discrete_slopes(rf, X, y, colname, verbose=False):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the samples then isolate the column of interest X values
    and the target y values. Perform another partition of X[colname] vs y and do
    piecewise linear regression to get the slopes in various regions of X[colname].
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the ranges of X[colname] partitions, num obs per leaf,
    associated slope for each range

    Only does discrete now after doing pointwise continuous slopes differently.
    """
    start = time.time()
    leaf_deltas = []  # drop or rise between discrete x values
    leaf_xranges = [] # drop is from one discrete value to next
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

        leaf_xranges_, leaf_sizes_, leaf_slopes_, leaf_r2_, ignored_ = \
            discrete_xc_space(leaf_x, leaf_y, colname=colname, verbose=verbose)

        leaf_deltas.extend(leaf_slopes_)
        leaf_xranges.extend(leaf_xranges_)
        leaf_sizes.extend(leaf_sizes_)
        ignored += ignored_

    leaf_xranges = np.array(leaf_xranges)
    leaf_sizes = np.array(leaf_sizes)
    leaf_deltas = np.array(leaf_deltas)

    leaf_slopes = leaf_deltas / (leaf_xranges[:,1] - leaf_xranges[:,0]) # "rise over run"
    stop = time.time()
    if verbose: print(f"collect_leaf_slopes {stop - start:.3f}s")
    return leaf_xranges, leaf_sizes, leaf_slopes, ignored


def weighted_avg_values_at_x(uniq_x, leaf_ranges, leaf_values, leaf_weights, use_weighted_avg):
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
        if use_weighted_avg:
            s = np.full(nx, slope*w, dtype=float) # s has value*weight at all locations (flat line)
        else:
            s = np.full(nx, slope, dtype=float)
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
    if use_weighted_avg:
        sum_values_at_x = np.nansum(slopes, axis=1)
        sum_weights_at_x = np.nansum(weights, axis=1)
        avg_value_at_x = sum_values_at_x / sum_weights_at_x
    else:
        avg_value_at_x = np.nanmean(slopes, axis=1)

    stop = time.time()
    # print(f"avg_value_at_x {stop - start:.3f}s")
    return avg_value_at_x

def avg_values_at_x(uniq_x, leaf_ranges, leaf_values):
    """
    Compute the weighted average of leaf_values at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    start = time.time()
    nx = len(uniq_x)
    nslopes = len(leaf_values)
    slopes = np.zeros(shape=(nx, nslopes))
    i = 0  # leaf index; we get a line for each leaf
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    for r, slope in zip(leaf_ranges, leaf_values):
        s = np.full(nx, slope, dtype=float)
        # now trim line so it's only valid in range r;
        # don't set slope on right edge
        s[np.where( (uniq_x < r[0]) | (uniq_x >= r[1]) )] = np.nan
        slopes[:, i] = s
        i += 1
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    avg_value_at_x = np.nanmean(slopes, axis=1)

    stop = time.time()
    # print(f"avg_value_at_x {stop - start:.3f}s")
    return avg_value_at_x


def blort_plot_stratpd(X, y, colname, targetname=None,
                 ax=None,
                 ntrees=1,
                 max_features = 1.0,
                 bootstrap=False,
                 min_samples_leaf=10,
                 nbins=3, # this is number of bins, so number of points in linear space is nbins+1
                          # ignored if isdiscrete; len(unique(X[colname])) used instead.
                          # must be >= 1
                 isdiscrete=False,
                 use_weighted_avg=False,
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
        print(f"{'discrete ' if isdiscrete else ''}StratPD num samples ignored {ignored}/{len(X)} for {colname}")

    slope_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_slopes, leaf_sizes, use_weighted_avg)
    r2_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_r2, leaf_sizes, use_weighted_avg)
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
    dydx = slope_at_x[:-1] * np.diff(uniq_x)          # last slope is nan since no data after last x value
    # print(f"dydx: {dydx}")
    curve = np.cumsum(dydx)                           # we lose one value here
    # curve = cumtrapz(slope_at_x, x=uniq_x)          # we lose one value here
    curve = np.concatenate([np.array([0]), curve])  # add back the 0 we lost
    # print(slope_at_x, len(slope_at_x))
    # print(dydx)
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
                            nbins_smoothing=None,
                            isdiscrete=False,
                            yrange=None,
                            show_regr_line=False,
                            marginal_alpha=.05,
                            alpha=.1,
                            use_weighted_avg=True):
    ncols = len(min_samples_leaf_values)
    if isdiscrete:
        fig, axes = plt.subplots(1, ncols + 1,
                                 figsize=((ncols + 1) * 2.5, 2.5))
        marginal_plot_(X, y, colname, targetname, ax=axes[0],
                       show_regr_line=show_regr_line, alpha=alpha)
        axes[0].set_title("Marginal", fontsize=10)
        col = 1
        for msl in min_samples_leaf_values:
            print(
                f"---------- min_samples_leaf={msl} ----------- ")
            try:
                leaf_xranges, leaf_slopes, Xbetas, plot_x, plot_y, ignored = \
                    plot_stratpd(X, y, colname, targetname, ax=axes[col],
                                 min_samples_leaf=msl,
                                 isdiscrete=True,
                                 yrange=yrange,
                                 ntrees=1,
                                 slope_line_alpha=alpha)
            except ValueError:
                axes[col].set_title(
                    f"Can't gen: leafsz={msl}",
                    fontsize=8)
            else:
                axes[col].set_title(
                    f"leafsz={msl}, ignored={100*ignored / len(X):.2f}%",fontsize=9)
            col += 1

    else:
        nrows = len(nbins_values)
        fig, axes = plt.subplots(nrows, ncols + 1,
                                 figsize=((ncols + 1) * 2.5, nrows * 2.5))

        row = 0
        for i, nbins in enumerate(nbins_values):
            marginal_plot_(X, y, colname, targetname, ax=axes[row, 0], show_regr_line=show_regr_line)
            if row==0:
                axes[row,0].set_title("Marginal", fontsize=10)
            col = 1
            for msl in min_samples_leaf_values:
                print(f"---------- min_samples_leaf={msl}, nbins={nbins:.2f} ----------- ")
                try:
                    leaf_xranges, leaf_slopes, Xbetas, plot_x, plot_y, ignored = \
                        plot_stratpd(X, y, colname, targetname, ax=axes[row, col],
                                     nbins=nbins,
                                     min_samples_leaf=msl,
                                     nbins_smoothing=nbins_smoothing,
                                     yrange=yrange,
                                     ntrees=1)
                except ValueError:
                    axes[row, col].set_title(
                        f"Can't gen: leafsz={msl}, nbins={nbins}",
                        fontsize=8)
                else:
                    axes[row, col].set_title(
                        f"leafsz={msl}, nbins={nbins},\nignored={100*ignored/len(X):.2f}%",
                        fontsize=9)
                col += 1
            row += 1


def marginal_plot_(X, y, colname, targetname, ax, alpha=.1, show_regr_line=True):
    ax.scatter(X[colname], y, alpha=alpha, label=None, s=10)
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


def marginal_catplot_(X, y, colname, targetname, ax, catnames, alpha=.1):
    catcodes, catnames_, catcode2name = getcats(X, colname, catnames)

    ax.scatter(X[colname].values, y.values, alpha=alpha, label=None, s=10)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    # col = X[colname]
    # cats = np.unique(col)

    ax.set_xticks(catcodes)
    ax.set_xticklabels(catnames_)


def plot_catstratpd_gridsearch(X, y, colname, targetname,
                               min_samples_leaf_values=(2, 5, 10, 20, 30),
                               catnames=None,
                               yrange=None):
    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(1, ncols + 1,
                             figsize=((ncols + 1) * 2.5, 2.5))

    marginal_catplot_(X, y, colname, targetname, catnames=catnames, ax=axes[0], alpha=0.05)
    axes[0].set_title("Marginal", fontsize=10)

    col = 1
    for msl in min_samples_leaf_values:
        print(f"---------- min_samples_leaf={msl} ----------- ")
        if yrange is not None:
            axes[col].set_ylim(yrange)
        try:
            catcodes_, catnames_, curve, ignored = \
                plot_catstratpd(X, y, colname, targetname, ax=axes[col],
                                min_samples_leaf=msl,
                                catnames=catnames,
                                yrange=yrange,
                                ntrees=1,
                                show_xticks=True,
                                sort=False)
        except ValueError:
            axes[col].set_title(f"Can't gen: leafsz={msl}", fontsize=8)
        else:
            axes[col].set_title(f"leafsz={msl}, ignored={ignored / len(X):.2f}%", fontsize=9)
        col += 1


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
    # catcol = X[colname].astype('category').cat.as_ordered()
    cats = np.unique(X[colname])
    # catcounts = np.zeros(max(cats)+1, dtype=int)
    leaf_sizes = []
    leaf_avgs = []
    maxcat = max(cats)
    leaf_histos = pd.DataFrame(index=range(0,maxcat+1))
    leaf_histos.index.name = 'category'
    leaf_catcounts = pd.DataFrame(index=range(0,maxcat+1))
    leaf_catcounts.index.name = 'category'
    ci = 0
    Xy = pd.concat([X, y], axis=1)
    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    # print(f"{len(leaves)} leaves")
    for sample in leaves:
        combined = Xy.iloc[sample]
        # print("\n", combined)
        groupby = combined.groupby(colname)
        avg_y_per_cat = groupby.mean()
        avg_y_per_cat = avg_y_per_cat[y.name]#.iloc[:,-1]
        if len(avg_y_per_cat) < 2:
            # print(f"ignoring {len(sample)} obs for {len(avg_y_per_cat)} cat(s) in leaf")
            ignored += len(sample)
            continue

        # we'll weight by count per cat later so must track
        count_y_per_cat = combined[colname].value_counts()
        leaf_catcounts['leaf' + str(ci)] = count_y_per_cat

        # record avg y value per cat above avg y in this leaf
        # This assignment copies cat y avgs to appropriate cat row using index
        # leaving cats w/o representation as nan
        avg_y = np.mean(combined[y.name])
        leaf_avgs.append(avg_y)
        delta_y_per_cat = avg_y_per_cat - avg_y
        leaf_histos['leaf' + str(ci)] = delta_y_per_cat
        leaf_sizes.append(len(sample))
        # print(f"L avg {avg_y:.2f}:\n\t{delta_y_per_cat}")
        ci += 1

    # print(f"Avg of leaf avgs is {np.mean(leaf_avgs):.2f} vs y avg {np.mean(y)}")
    stop = time.time()
    if verbose: print(f"catwise_leaves {stop - start:.3f}s")
    return leaf_histos, np.array(leaf_avgs), leaf_sizes, leaf_catcounts, ignored


# only works for ints, not floats
def plot_catstratpd(X, y,
                    colname,  # X[colname] expected to be numeric codes
                    targetname,
                    catnames=None,  # map of catcodes to catnames; converted to map if sequence passed
                    # must pass dict or series if catcodes are not 1..n contiguous
                    # None implies use np.unique(X[colname]) values
                    # Must be 0-indexed list of names if list
                    ax=None,
                    sort='ascending',
                    ntrees=1,
                    min_samples_leaf=10,
                    max_features=1.0,
                    bootstrap=False,
                    yrange=None,
                    title=None,
                    supervised=True,
                    use_weighted_avg=True,
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

    catcodes, _, catcode2name = getcats(X, colname, catnames)

    # rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_histos, leaf_avgs, leaf_sizes, leaf_catcounts, ignored = \
        catwise_leaves(rf, X, y, colname, verbose=verbose)

    if True:
        print(f"CatStratPD Num samples ignored {ignored} for {colname}")

    if use_weighted_avg:
        weighted_histos = leaf_histos * leaf_catcounts
        weighted_sum_per_cat = np.nansum(weighted_histos, axis=1)  # sum across columns
        total_obs_per_cat = np.nansum(leaf_catcounts, axis=1)
        avg_per_cat = weighted_sum_per_cat / total_obs_per_cat
    else:
        avg_per_cat = np.nanmean(leaf_histos, axis=1)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ncats = len(catcodes)
    nleaves = len(leaf_histos.columns)

    sorted_catcodes = catcodes
    if sort == 'ascending':
        sorted_indexes = avg_per_cat[~np.isnan(avg_per_cat)].argsort()
        sorted_catcodes = catcodes[sorted_indexes]
    elif sort == 'descending':
        sorted_indexes = avg_per_cat.argsort()[::-1]  # reversed
        sorted_catcodes = catcodes[sorted_indexes]


    # The category y deltas straddle 0 but it's easier to understand if we normalize
    # so lowest y delta is 0
    min_avg_value = np.nanmin(avg_per_cat)

    # print(leaf_histos.iloc[np.nonzero(catcounts)])
    # # print(leaf_histos.notna().multiply(leaf_sizes, axis=1))
    # # print(np.sum(leaf_histos.notna().multiply(leaf_sizes, axis=1), axis=1))
    # print(f"leaf_sizes: {list(leaf_sizes)}")
    # print(f"weighted_sum_per_cat: {list(weighted_sum_per_cat[np.nonzero(weighted_sum_per_cat)])}")
    # # print(f"catcounts: {list(catcounts[np.nonzero(catcounts)])}")
    # # print(f"Avg per cat: {list(avg_per_cat[np.nonzero(catcounts)]-min_avg_value)}")
    # print(f"Avg per cat: {list(avg_per_cat[~np.isnan(avg_per_cat)]-min_avg_value)}")

    # if too many categories, can't do strip plot
    xloc = 0
    sigma = .02
    mu = 0
    if style == 'strip':
        x_noise = np.random.normal(mu, sigma, size=nleaves) # to make strip plot
    else:
        x_noise = np.zeros(shape=(nleaves,))
    for cat in sorted_catcodes:
        if catcode2name[cat] is None: continue
        ax.scatter(x_noise + xloc, leaf_histos.iloc[cat] - min_avg_value,
                   alpha=alpha, marker='o', s=marker_size,
                   c=color)
        if style == 'strip':
            ax.plot([xloc - .1, xloc + .1], [avg_per_cat[cat]-min_avg_value] * 2,
                    c='black', linewidth=2)
        else:
            ax.scatter(xloc, avg_per_cat[cat]-min_avg_value, c=pdp_color, s=pdp_marker_size)
        xloc += 1

    ax.set_xticks(range(0, ncats))
    if show_xticks: # sometimes too many
        ax.set_xticklabels(catcode2name[sorted_catcodes])
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

    ycats = avg_per_cat[sorted_catcodes] - min_avg_value
    return catcodes, catcode2name[sorted_catcodes], ycats, ignored


def getcats(X, colname, incoming_cats):
    if incoming_cats is None or isinstance(incoming_cats, pd.Series):
        catcodes = np.unique(X[colname])
        catcode2name = [None] * (max(catcodes) + 1)
        for c in catcodes:
            catcode2name[c] = c
        catcode2name = np.array(catcode2name)
        catnames = catcodes
    elif isinstance(incoming_cats, dict):
        catnames_ = [None] * (max(incoming_cats.keys()) + 1)
        catcodes = []
        catnames = []
        for code, name in incoming_cats.items():
            catcodes.append(code)
            catnames.append(name)
            catnames_[code] = name
        catcodes = np.array(catcodes)
        catnames = np.array(catnames)
        catcode2name = np.array(catnames_)
    elif not isinstance(incoming_cats, dict):
        # must be a list of names then
        catcodes = []
        catnames_ = [None] * len(incoming_cats)
        for cat, c in enumerate(incoming_cats):
            if c is not None:
                catcodes.append(cat)
            catnames_[cat] = c
        catcodes = np.array(catcodes)
        catcode2name = np.array(catnames_)
        catnames = np.array(incoming_cats)
    else:
        raise ValueError("catnames must be None, 0-indexed list, or pd.Series")
    return catcodes, catnames, catcode2name


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
