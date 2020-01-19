import numpy as np
from numpy import nan, where
import pandas as pd
from typing import Mapping, List, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.stats import binned_statistic
import warnings
import collections
from timeit import default_timer as timer
from sklearn.utils import resample

# from stratx.cy_partdep import cy_avg_values_at_x_double

from dtreeviz.trees import *
from snowballstemmer.dutch_stemmer import lab0
from numba import jit, prange
import numba


'''
def leaf_samples_general(rf, X:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest.
    """
    n_trees = len(rf.estimators_)
    leaf_ids = rf.apply(X) # which leaf does each X_i go to for each tree?
    d = pd.DataFrame(leaf_ids, columns=[f"tree{i}" for i in range(n_trees)])
    d = d.reset_index() # get 0..n-1 as column called index so we can do groupby
    """
    d looks like:
        index	tree0	tree1	tree2	tree3	tree4
    0	0	    8	    3	    4	    4	    3
    1	1	    8	    3	    4	    4	    3
    """
    leaf_samples = []
    for i in range(n_trees):
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
'''

def leaf_samples(rf, X_not_col:np.ndarray):
    """
    Return a list of arrays where each array is the set of X sample indexes
    residing in a single leaf of some tree in rf forest. For example, if there
    are 4 leaves (in one or multiple trees), we might return:

        array([array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
           array([10, 11, 12, 13, 14, 15]), array([16, 17, 18, 19, 20]),
           array([21, 22, 23, 24, 25, 26, 27, 28, 29]))
    """
    n_trees = len(rf.estimators_)
    leaf_samples = []
    leaf_ids = rf.apply(X_not_col)  # which leaf does each X_i go to for sole tree?
    for t in range(n_trees):
        # Group by id and return sample indexes
        uniq_ids = np.unique(leaf_ids[:,t])
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id) for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples


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


def partial_dependence(X:pd.DataFrame, y:pd.Series, colname:str,
                       min_slopes_per_x=5,
                       # ignore pdp y values derived from too few slopes (usually at edges)
                       # important for getting good starting point of PD so AUC isn't skewed.
                       parallel_jit=True,
                       n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                       supervised=True,
                       verbose=False):
    """
    Internal computation of partial dependence information about X[colname]'s effect on y.
    Also computes partial derivative of y with respect to X[colname].

    :param X: 
    :param y: 
    :param colname: 
    :param min_slopes_per_x:   ignore pdp y values derived from too few slopes (less than .3% of num records)
                            tried percentage of max slope count but was too variable; this is same count across all features
    :param n_trees:
    :param min_samples_leaf: 
    :param bootstrap: 
    :param max_features: 
    :param supervised: 
    :param verbose: 

    Returns:
        leaf_xranges    The ranges of X[colname] partitions


        leaf_slopes     Associated slope for each leaf xrange

        dx              The change in x from one non-NaN unique X[colname] to the next

        dydx            The slope at each non-NaN unique X[colname]

        pdpx            The non-NaN unique X[colname] values

        pdpy            The effect of each non-NaN unique X[colname] on y; effectively
                        the cumulative sum (integration from X[colname] x to z for all
                        z in X[colname]). The first value is always 0.

        ignored         How many samples from len(X) total records did we have to
                        ignore because of samples in leaves with identical X[colname]
                        values.
    """
    X_not_col = X.drop(colname, axis=1).values
    X_col = X[colname]
    if supervised:
        rf = RandomForestRegressor(n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features)
        rf.fit(X_not_col, y)
        if verbose:
            print(f"Strat Partition RF: dropping {colname} training R^2 {rf.score(X_not_col, y):.2f}")

    else:
        """
        Wow. Breiman's trick works in most cases. Falls apart on Boston housing MEDV target vs AGE
        """
        if verbose: print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                   min_samples_leaf=int(min_samples_leaf * 2),  # there are 2x as many samples (X,X') so must double leaf size
                                   bootstrap=bootstrap,
                                   max_features=max_features,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname, axis=1), y_synth)

    if verbose:
        leaves = leaf_samples(rf, X_not_col)
        nnodes = rf.estimators_[0].tree_.node_count
        print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
              f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    leaf_xranges, leaf_slopes, ignored = \
        collect_discrete_slopes(rf, X, y, colname)

    # print('leaf_xranges', leaf_xranges)
    # print('leaf_slopes', leaf_slopes)

    real_uniq_x = np.array(sorted(np.unique(X_col)))
    if verbose:
        print(f"discrete StratPD num samples ignored {ignored}/{len(X)} for {colname}")


    if parallel_jit:
        slope_at_x, slope_counts_at_x = \
            avg_values_at_x_jit(real_uniq_x, leaf_xranges, leaf_slopes)
    else:
        slope_at_x, slope_counts_at_x = \
            avg_values_at_x_nonparallel_jit(real_uniq_x, leaf_xranges, leaf_slopes)

    # Drop any nan slopes; implies we have no reliable data for that range
    # Last slope is nan since no data after last x value so that will get dropped too
    # Also cut out any pdp x for which we don't have enough support (num slopes avg'd together)
    # Make sure to drop slope_counts_at_x, uniq_x values too :)
    if min_slopes_per_x <= 0:
        min_slopes_per_x = 1 # must have at least one slope value
    notnan_idx = ~np.isnan(slope_at_x)
    relevant_slopes = slope_counts_at_x >= min_slopes_per_x
    idx = notnan_idx & relevant_slopes
    slope_at_x = slope_at_x[idx]
    slope_counts_at_x = slope_counts_at_x[idx]
    pdpx = real_uniq_x[idx]

    dx = np.diff(pdpx)
    dydx = slope_at_x[:-1] # ignore last point as dx is always one smaller
    y_deltas = dydx * dx
    # print(f"y_deltas: {y_deltas}")
    pdpy = np.cumsum(y_deltas)                    # we lose one value here
    pdpy = np.concatenate([np.array([0]), pdpy])  # add back the 0 we lost

    return leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored


def plot_stratpd_binned(X, y, colname, targetname,
                 n_trees=1, min_samples_leaf=10, bootstrap=False,
                 max_features=1.0,
                 nbins=3,  # piecewise binning
                 nbins_smoothing=None,  # binning of overall X[colname] space in plot
                 supervised=True,
                 ax=None,
                 xrange=None,
                 yrange=None,
                 title=None,
                 nlines=None,
                 show_xlabel=True,
                 show_ylabel=True,
                 show_pdp_line=False,
                 show_slope_lines=True,
                 pdp_marker_size=5,
                 pdp_line_width=.5,
                 slope_line_color='#2c7fb8',
                 slope_line_width=.5,
                 slope_line_alpha=.3,
                 pdp_line_color='black',
                 pdp_marker_color='black',
                 verbose=False
                 ):
    if supervised:
        rf = RandomForestRegressor(n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features)
        rf.fit(X.drop(colname, axis=1), y)
        if verbose:
            print(f"Strat Partition RF: dropping {colname} training R^2 {rf.score(X.drop(colname, axis=1), y):.2f}")

    else:
        """
        Wow. Breiman's trick works in most cases. Falls apart on Boston housing MEDV target vs AGE
        """
        if verbose: print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    min_samples_leaf=min_samples_leaf * 2,
                                    # there are 2x as many samples (X,X') so must double leaf size
                                    bootstrap=bootstrap,
                                    max_features=max_features,
                                    oob_score=False)
        rf.fit(X_synth.drop(colname, axis=1), y_synth)

    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    nnodes = rf.estimators_[0].tree_.node_count
    if verbose:
        print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
              f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    leaf_xranges, leaf_slopes, point_betas, ignored = \
        collect_point_betas(X, y, colname, leaves, nbins)
    Xbetas = np.vstack([X[colname].values, point_betas]).T # get x_c, beta matrix
    # Xbetas = Xbetas[Xbetas[:,0].argsort()] # sort by x coordinate (not needed)

    #print(f"StratPD num samples ignored {ignored}/{len(X)} for {colname}")

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

    # print('bins_smoothing', bins_smoothing, ', deltas', bin_deltas)
    # print('avgslopes', delta_ys)

    # manual cumsum
    delta_ys = np.concatenate([np.array([0]), delta_ys])  # we start at 0 for min(x)
    pdpx = []
    pdpy = []
    cumslope = 0.0
    # delta_ys_ = np.concatenate([np.array([0]), delta_ys])  # we start at 0 for min(x)
    for x, slope in zip(bins_smoothing, delta_ys):
        if np.isnan(slope):
            # print(f"{x:5.3f},{cumslope:5.1f},{slope:5.1f} SKIP")
            continue
        cumslope += slope
        pdpx.append(x)
        pdpy.append(cumslope)
        # print(f"{x:5.3f},{cumslope:5.1f},{slope:5.1f}")
    pdpx = np.array(pdpx)
    pdpy = np.array(pdpy)

    # PLOT

    if ax is None:
        fig, ax = plt.subplots(1,1)

    # Draw bin left edge markers; ignore bins with no data (nan)
    ax.scatter(pdpx, pdpy,
               s=pdp_marker_size, c=pdp_marker_color)

    if show_pdp_line:
        ax.plot(pdpx, pdpy,
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

    return leaf_xranges, leaf_slopes, Xbetas, pdpx, pdpy, ignored


def plot_stratpd(X:pd.DataFrame, y:pd.Series, colname:str, targetname:str,
                 min_slopes_per_x=5,  # ignore pdp y values derived from too few slopes (usually at edges)
                 # important for getting good starting point of PD so AUC isn't skewed.
                 n_trials=10, # how many pd curves to show (subsampling by 2/3 to get diff X sets)
                 n_trees=1, min_samples_leaf=10, bootstrap=False,
                 max_features=1.0,
                 supervised=True,
                 ax=None,
                 xrange=None,
                 yrange=None,
                 title=None,
                 show_xlabel=True,
                 show_ylabel=True,
                 show_pdp_line=False,
                 show_all_pdp=True,
                 show_slope_lines=True,
                 show_slope_counts=False,
                 show_x_counts=True,
                 show_impact=False,
                 show_impact_dots=True,
                 show_impact_line=True,
                 pdp_marker_size=4,
                 pdp_marker_alpha=.5,
                 pdp_line_width=.5,
                 slope_line_color='#2c7fb8',
                 slope_line_width=.5,
                 slope_line_alpha=.3,
                 pdp_line_color='black',
                 pdp_marker_color='black',
                 pdp_marker_cmap='coolwarm',
                 impact_fill_color='#FFE091',
                 impact_pdp_color='#D73028',
                 impact_marker_size=3,
                 fontname='Arial',
                 title_fontsize=11,
                 label_fontsize=10,
                 ticklabel_fontsize=10,
                 barchart_size=0.20,
                 # if show_slope_counts, what ratio of vertical space should barchart use at bottom?
                 barchar_alpha=0.7,
                 verbose=False,
                 figsize=None
                 ):
    """
    Plot the partial dependence of X[colname] on y.

    Returns:
        leaf_xranges    The ranges of X[colname] partitions


        leaf_slopes     Associated slope for each leaf xrange

        dx              The change in x from one non-NaN unique X[colname] to the next

        dydx            The slope at each non-NaN unique X[colname]

        pdpx            The non-NaN unique X[colname] values

        pdpy            The effect of each non-NaN unique X[colname] on y; effectively
                        the cumulative sum (integration from X[colname] x to z for all
                        z in X[colname]). The first value is always 0.

        ignored         How many samples from len(X) total records did we have to
                        ignore because of samples in leaves with identical X[colname]
                        values.
    """
    def avg_pd_curve(all_pdpx, all_pdpy):
        m = defaultdict(float)
        c = defaultdict(int)
        for i in range(n_trials):
            for px, py in zip(all_pdpx, all_pdpy):
                for x, y in zip(px, py):
                    m[x] += y
                    c[x] += 1
        for x in m.keys():
            m[x] /= c[x]

        # We now have dict with average pdpy for each pdpx found in any curve
        # but we need to ensure we get it back in pdpx order
        pdpx = np.array(sorted(m.keys()))
        pdpy = np.empty(shape=(len(m),))
        for i,x in enumerate(pdpx):
            pdpy[i] = m[x]
        return pdpx, pdpy

    # leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
    #     partial_dependence(X=X, y=y, colname=colname, min_slopes_per_x=min_slopes_per_x,
    #                        n_trees=n_trees, min_samples_leaf=min_samples_leaf,
    #                        bootstrap=bootstrap, max_features=max_features, supervised=supervised,
    #                        verbose=verbose)
    #
    all_pdpx = []
    all_pdpy = []
    n = len(X)
    ignored = 0
    for i in range(n_trials):
        # idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
        if n_trials>1:
            idxs = resample(range(n), n_samples=int(n * 2 / 3), replace=False)  # subset
            X_, y_ = X.iloc[idxs], y.iloc[idxs]
        else:
            X_, y_ = X, y

        leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored_ = \
            partial_dependence(X=X_, y=y_, colname=colname,
                               min_slopes_per_x=min_slopes_per_x,
                               n_trees=n_trees, min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap, max_features=max_features,
                               supervised=supervised,
                               verbose=verbose)
        ignored += ignored_
        all_pdpx.append(pdpx)
        all_pdpy.append(pdpy)

    ignored /= n_trials # average number of x values ignored across trials

    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

    if show_all_pdp:
        sorted_by_imp = np.argsort([np.mean(np.abs(v)) for v in all_pdpy])
        cmap = plt.get_cmap(pdp_marker_cmap)
        ax.set_prop_cycle(color=cmap(np.linspace(0,1,num=n_trials)))
        for i in range(n_trials):
            ax.scatter(all_pdpx[sorted_by_imp[i]], all_pdpy[sorted_by_imp[i]],
                       s=pdp_marker_size, label=colname, alpha=pdp_marker_alpha)

    # Get avg curve, reset pdpx and pdpy to the average
    pdpx, pdpy = avg_pd_curve(all_pdpx, all_pdpy)
    # pdpx, pdpy = np.array(list(m.keys())), np.array(list(m.values()))
    ax.scatter(pdpx, pdpy, c=pdp_marker_color, s=pdp_marker_size+1)

    if show_pdp_line:
        ax.plot(pdpx, pdpy, lw=pdp_line_width, c=pdp_line_color)

    domain = (np.min(X[colname]), np.max(X[colname]))  # ignores any max(x) points as no slope info after that

    min_y = min(pdpy)
    max_y = max(pdpy)
    if n_trials==1 and show_slope_lines:
        segments = []
        for xr, slope in zip(leaf_xranges, leaf_slopes):
            w = np.abs(xr[1] - xr[0])
            delta_y = slope * w
            closest_x_i = np.abs(pdpx - xr[0]).argmin() # find curve point for xr[0]
            closest_x = pdpx[closest_x_i]
            closest_y = pdpy[closest_x_i]
            slope_line_endpoint_y = closest_y + delta_y
            one_line = [(closest_x, closest_y), (closest_x + w, slope_line_endpoint_y)]
            segments.append( one_line )
            if slope_line_endpoint_y < min_y:
                min_y = slope_line_endpoint_y
            elif slope_line_endpoint_y > max_y:
                max_y = slope_line_endpoint_y

        lines = LineCollection(segments, alpha=slope_line_alpha, color=slope_line_color, linewidths=slope_line_width)
        ax.add_collection(lines)

    if xrange is not None:
        ax.set_xlim(*xrange)
    else:
        ax.set_xlim(*domain)
    if yrange is not None:
        ax.set_ylim(*yrange)
    else:
        ax.set_ylim(min_y, max_y)

    X_col = X[colname]
    _, pdpx_counts = np.unique(X_col[np.isin(X_col, pdpx)], return_counts=True)

    leave_room_scaler = 1.3
    x_width = max(pdpx) - min(pdpx) + 1
    count_bar_width = x_width / len(pdpx)
    if count_bar_width/x_width < 0.002:
        count_bar_width = x_width * 0.002 # don't make them so skinny they're invisible
    # print(f"x_width={x_width:.2f}, count_bar_width={count_bar_width}")
    if show_x_counts:
        ax2 = ax.twinx()
        # scale y axis so the max count height is 10% of overall chart
        ax2.set_ylim(0, max(pdpx_counts) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(pdpx_counts)]))
        ax2.bar(x=pdpx, height=pdpx_counts, width=count_bar_width,
                facecolor='#BABABA', align='center', alpha=barchar_alpha)
        ax2.set_ylabel(f"$x$ point count", labelpad=-12, fontsize=label_fontsize,
                       fontstretch='extra-condensed',
                       fontname=fontname)
        # shift other y axis down barchart_size to make room
        if yrange is not None:
            ax.set_ylim(yrange[0] - (yrange[1]-yrange[0]) * barchart_size * leave_room_scaler, yrange[1])
        else:
            ax.set_ylim(min_y-(max_y-min_y)*barchart_size * leave_room_scaler, max_y)
        ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
        for tick in ax2.get_xticklabels():
            tick.set_fontname(fontname)
        for tick in ax2.get_yticklabels():
            tick.set_fontname(fontname)
        ax2.spines['top'].set_linewidth(.5)
        ax2.spines['right'].set_linewidth(.5)
        ax2.spines['left'].set_linewidth(.5)
        ax2.spines['bottom'].set_linewidth(.5)

    if n_trials==1 and show_slope_counts:
        ax2 = ax.twinx()
        # scale y axis so the max count height is barchart_size of overall chart
        ax2.set_ylim(0, max(slope_counts_at_x) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(slope_counts_at_x)]))
        ax2.bar(x=pdpx, height=slope_counts_at_x, width=count_bar_width,
                facecolor='#BABABA', align='center', alpha=barchar_alpha)
        ax2.set_ylabel(f"slope count", labelpad=-12, fontsize=label_fontsize,
                       fontstretch='extra-condensed',
                       fontname=fontname)
        # shift other y axis down barchart_size to make room
        if yrange is not None:
            ax.set_ylim(yrange[0]-(yrange[1]-yrange[0])*barchart_size * leave_room_scaler, yrange[1])
        else:
            ax.set_ylim(min_y-(max_y-min_y)*barchart_size, max_y)
        ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
        for tick in ax2.get_xticklabels():
            tick.set_fontname(fontname)
        for tick in ax2.get_yticklabels():
            tick.set_fontname(fontname)
        ax2.spines['top'].set_linewidth(.5)
        ax2.spines['right'].set_linewidth(.5)
        ax2.spines['left'].set_linewidth(.5)
        ax2.spines['bottom'].set_linewidth(.5)

    if show_impact:
        # r = max_y - min_y
        # if max(weighted_pdpy) > 0:
        #     verticalalignment = 'bottom'
        #     y_text_shift = r*.01
        # else:
        #     verticalalignment = 'top'
        #     y_text_shift = -r*.02 # drop a bit to avoid collision with 0 line
        # ax.text(0.5, .98, f"Impact {impact:.2f}", horizontalalignment='center',
        #         verticalalignment='top', transform=ax.transAxes,
        #         fontsize=label_fontsize, fontname=fontname)
        # ax.text((max(pdpx)+1+min(pdpx))/2, 0+y_text_shift, f"Impact {impact:.2f}",
        #         horizontalalignment='center', verticalalignment=verticalalignment,
        #         fontsize=label_fontsize, fontname=fontname)
        ax.fill_between(pdpx, pdpy, [0] * len(pdpx), color=impact_fill_color)
        if show_impact_dots:
            ax.scatter(pdpx, pdpy, s=impact_marker_size, c=impact_pdp_color)
        if show_impact_line:
            ax.plot(pdpx, pdpy, lw=.3, c='grey')

    if show_xlabel:
        xl = colname
        # impact = np.sum(np.abs(pdpy * pdpx_counts)) / np.sum(pdpx_counts)
        impact = np.mean(np.abs(pdpy))
        xl += f" (Impact {impact:.2f})"
        ax.set_xlabel(xl, fontsize=label_fontsize, fontname=fontname)
    if show_ylabel:
        ax.set_ylabel(targetname, fontsize=label_fontsize, fontname=fontname)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

    return pdpx, pdpy, ignored


@jit(nopython=True)
def discrete_xc_space(x: np.ndarray, y: np.ndarray):
    """
    Use the unique x values within a leaf to dynamically compute the bins,
    rather then using a fixed nbins hyper parameter. Group the leaf x,y by x
    and collect the average y.  The unique x and y averages are the new x and y pairs.
    The slope for each x is:

        (y_{i+1} - y_i) / (x_{i+1} - x_i)

    If the ordinal/ints are exactly one unit part, then it's just y_{i+1} - y_i. If
    they are not consecutive, we do not ignore isolated x_i as it ignores too much data.
    E.g., if x is [1,3,4] and y is [9,8,10] then the x=2 coordinate is spanned as part
    of 1 to 3. The two slopes are [(8-9)/(3-1), (10-8)/(4-3)] and bin widths are [2,1].

    If there is exactly one unique x value in the leaf, the leaf provides no information
    about how x_c contributes to changes in y. We have to ignore this leaf.
    """
    ignored = 0

    # Group by x, take mean of all y with same x value (they come back sorted too)
    uniq_x = np.unique(x)
    avg_y = np.array([y[x==ux].mean() for ux in uniq_x])

    if len(uniq_x)==1:
        # print(f"ignore {len(x)} in discrete_xc_space")
        ignored += len(x)
        return np.array([[0]],dtype=x.dtype), np.array([0.0]), ignored

    bin_deltas = np.diff(uniq_x)
    y_deltas = np.diff(avg_y)
    leaf_slopes = y_deltas / bin_deltas  # "rise over run"
    leaf_xranges = np.array(list(zip(uniq_x, uniq_x[1:])))

    return leaf_xranges, leaf_slopes, ignored

def collect_discrete_slopes(rf, X, y, colname):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the leaf samples then isolate the column of interest X values
    and the target y values. Perform another partition of X[colname] vs y and do
    piecewise linear regression to get the slopes in various regions of X[colname].
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the ranges of X[colname] partitions, num obs per x range,
    associated slope for each range

    Only does discrete now after doing pointwise continuous slopes differently.
    """
    # start = timer()
    leaf_slopes = []  # drop or rise between discrete x values
    leaf_xranges = [] # drop is from one discrete value to next

    ignored = 0

    X_col = X[colname].values
    X_not_col = X.drop(colname, axis=1)
    leaves = leaf_samples(rf, X_not_col)
    y = y.values

    if False:
        nnodes = rf.estimators_[0].tree_.node_count
        print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
              f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    for samples in leaves:
        leaf_x = X_col[samples]
        # leaf_x = one_leaf_samples[]#.reshape(-1,1)
        leaf_y = y[samples]

        if np.abs(np.min(leaf_x) - np.max(leaf_x)) < 1.e-8: # faster than np.isclose()
            # print(f"ignoring xleft=xright @ {r[0]}")
            ignored += len(leaf_x)
            continue

        leaf_xranges_, leaf_slopes_, ignored_ = \
            discrete_xc_space(leaf_x, leaf_y)

        leaf_slopes.extend(leaf_slopes_)
        leaf_xranges.extend(leaf_xranges_)
        ignored += ignored_

    if len(leaf_xranges)==0:
        # make sure empty list has same shape (jit complains)
        leaf_xranges = np.array([]).reshape(0, 0)
    else:
        leaf_xranges = np.array(leaf_xranges)
    leaf_slopes = np.array(leaf_slopes)

    # stop = timer()
    # if verbose: print(f"collect_discrete_slopes {stop - start:.3f}s")
    return leaf_xranges, leaf_slopes, ignored


'''
Currently not needed
def avg_values_at_x_nojit(uniq_x, leaf_ranges, leaf_slopes):
    """
    Compute the weighted average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    nx = len(uniq_x)
    nslopes = len(leaf_slopes)
    slopes = np.zeros(shape=(nx, nslopes))
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    i = 0
    for xr, slope in zip(leaf_ranges, leaf_slopes):
        s = np.full(nx, slope, dtype=float)
        # now trim line so it's only valid in range xr;
        # don't set slope on right edge
        s[np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = np.nan
        slopes[:, i] = s
        i += 1

    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    # Oh right. We might have to ignore some leaves (those with single unique x values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        avg_value_at_x = np.nanmean(slopes, axis=1)
        # how many slopes avg'd together to get avg
        slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)

    # return average slope at each unique x value and how many slopes included in avg at each x
    return avg_value_at_x, slope_counts_at_x
'''


# We get about 20% boost from parallel but limits use of other parallelism it seems;
# i get crashes when using multiprocessing package on top of this.
# If using n_jobs=1 all the time for importances, then turn jit=False so this
# method is not used
@jit(nopython=True, parallel=True) # use prange not range.
def avg_values_at_x_jit(uniq_x, leaf_ranges, leaf_slopes):
    """
    Compute the weighted average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    nx = uniq_x.shape[0]
    nslopes = leaf_slopes.shape[0]
    slopes = np.empty(shape=(nx, nslopes), dtype=np.double)
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range

    '''
    for j in prange(nslopes):
        xl = leaf_ranges[j,0]
        xr = leaf_ranges[j,1]
        slope = leaf_slopes[j]
        # s = np.full(nx, slope)#, dtype=double)
        # s[np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = np.nan
        # slopes[:, i] = s

        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        for i in prange(nx):
            if (uniq_x[i] >= xl) or (uniq_x[i] < xr):
                slopes[i, j] = slope
            else:
                slopes[i, j] = np.nan
    '''

    for i in prange(nslopes):
        xr, slope = leaf_ranges[i], leaf_slopes[i]
        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        slopes[:, i] = np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]), np.nan, slope)

    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    # Oh right. We might have to ignore some leaves (those with single unique x values)

    # Compute:
    #   avg_value_at_x = np.mean(slopes[good], axis=1)  (numba doesn't allow axis arg)
    #   slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)
    avg_value_at_x = np.zeros(shape=nx)
    slope_counts_at_x = np.zeros(shape=nx)
    for i in prange(nx):
        row = slopes[i, :]
        n_nan = np.sum(np.isnan(row))
        avg_value_at_x[i] = np.nan if n_nan==nslopes else np.nanmean(row)
        slope_counts_at_x[i] = nslopes - n_nan

    # return average slope at each unique x value and how many slopes included in avg at each x
    return avg_value_at_x, slope_counts_at_x


# Hideous copying to get different kinds of jit'ing. This is slower by 20%
# than other version but can run in parallel with multiprocessing package.
@jit(nopython=True)
def avg_values_at_x_nonparallel_jit(uniq_x, leaf_ranges, leaf_slopes):
    """
    Compute the weighted average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point.
    """
    nx = len(uniq_x)
    nslopes = len(leaf_slopes)
    slopes = np.zeros(shape=(nx, nslopes))
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range

    for i in range(nslopes):
        xr, slope = leaf_ranges[i], leaf_slopes[i]

        # s = np.full(nx, slope)#, dtype=float)
        # s[np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]) )] = np.nan
        # slopes[:, i] = s

        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        slopes[:, i] = np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]), np.nan, slope)


    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    # Wrap nanmean() in catcher to avoid "Mean of empty slice" warning, which
    # comes from some rows being purely NaN; I should probably look at this sometime
    # to decide whether that's hiding a bug (can there ever be a nan for an x range)?
    # Oh right. We might have to ignore some leaves (those with single unique x values)

    # Compute:
    #   avg_value_at_x = np.mean(slopes[good], axis=1)  (numba doesn't allow axis arg)
    #   slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)
    avg_value_at_x = np.zeros(shape=nx)
    slope_counts_at_x = np.zeros(shape=nx)
    for i in range(nx):
        row = slopes[i, :]
        n_nan = np.sum(np.isnan(row))
        avg_value_at_x[i] = np.nan if n_nan==nslopes else np.nanmean(row)
        slope_counts_at_x[i] = nslopes - n_nan

    # return average slope at each unique x value and how many slopes included in avg at each x
    return avg_value_at_x, slope_counts_at_x


def plot_stratpd_gridsearch(X, y, colname, targetname,
                            min_samples_leaf_values=(2,5,10,20,30),
                            min_slopes_per_x_values=(5,), # Show default count only by default
                            n_trials=10,
                            nbins_values=(1,2,3,4,5),
                            nbins_smoothing=None,
                            binned=False,
                            yrange=None,
                            xrange=None,
                            show_regr_line=False,
                            show_slope_lines=True,
                            show_impact=False,
                            show_slope_counts=False,
                            show_x_counts=True,
                            marginal_alpha=.05,
                            slope_line_alpha=.1,
                            title_fontsize=8,
                            label_fontsize=7,
                            ticklabel_fontsize=7,
                            cellwidth=2.5,
                            cellheight=2.5):
    ncols = len(min_samples_leaf_values)
    if not binned:
        fig, axes = plt.subplots(len(min_slopes_per_x_values), ncols + 1,
                                 figsize=((ncols + 1) * cellwidth, len(min_slopes_per_x_values)*cellheight))
        if len(min_slopes_per_x_values)==1:
            axes = axes.reshape(1,-1)
        for row,min_slopes_per_x in enumerate(min_slopes_per_x_values):
            marginal_plot_(X, y, colname, targetname, ax=axes[row][0],
                           show_regr_line=show_regr_line, alpha=marginal_alpha,
                           label_fontsize=label_fontsize,
                           ticklabel_fontsize=ticklabel_fontsize)
            col = 1
            axes[row][0].set_title("Marginal", fontsize=title_fontsize)
            for msl in min_samples_leaf_values:
                #print(f"---------- min_samples_leaf={msl} ----------- ")
                try:
                    pdpx, pdpy, ignored = \
                        plot_stratpd(X, y, colname, targetname, ax=axes[row][col],
                                     min_samples_leaf=msl,
                                     min_slopes_per_x=min_slopes_per_x,
                                     n_trials=n_trials,
                                     xrange=xrange,
                                     yrange=yrange,
                                     n_trees=1,
                                     show_ylabel=False,
                                     slope_line_alpha=slope_line_alpha,
                                     show_slope_lines=show_slope_lines,
                                     show_impact=show_impact,
                                     show_slope_counts=show_slope_counts,
                                     show_x_counts=show_x_counts,
                                     label_fontsize=label_fontsize,
                                     ticklabel_fontsize=ticklabel_fontsize)
                    # print(f"leafsz {msl} avg abs curve value: {np.mean(np.abs(pdpy)):.2f}, mean {np.mean(pdpy):.2f}, min {np.min(pdpy):.2f}, max {np.max(pdpy)}")
                except ValueError as e:
                    print(e)
                    axes[row][col].set_title(f"Can't gen: leafsz={msl}", fontsize=8)
                else:
                    title = f"leafsz={msl}, min_slopes={min_slopes_per_x}"
                    if ignored>0:
                        title = f"leafsz={msl}, min_slopes={min_slopes_per_x},\nignored={100 * ignored / len(X):.2f}%"
                    axes[row][col].set_title(title, fontsize=title_fontsize)
                col += 1

    else:
        # more or less ignoring this branch these days
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
                #print(f"---------- min_samples_leaf={msl}, nbins={nbins:.2f} ----------- ")
                try:
                    leaf_xranges, leaf_slopes, Xbetas, plot_x, plot_y, ignored = \
                        plot_stratpd_binned(X, y, colname, targetname, ax=axes[row, col],
                                            nbins=nbins,
                                            min_samples_leaf=msl,
                                            nbins_smoothing=nbins_smoothing,
                                            yrange=yrange,
                                            show_ylabel=False,
                                            n_trees=1)
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


def marginal_plot_(X, y, colname, targetname, ax, alpha=.1, show_regr_line=True,
                   label_fontsize=7,
                   ticklabel_fontsize=7):
    ax.scatter(X[colname], y, alpha=alpha, label=None, s=10)
    ax.set_xlabel(colname, fontsize=label_fontsize)
    ax.set_ylabel(targetname, fontsize=label_fontsize)
    col = X[colname]

    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

    if show_regr_line:
        r = LinearRegression()
        r.fit(X[[colname]], y)
        xcol = np.linspace(np.min(col), np.max(col), num=100)
        yhat = r.predict(xcol.reshape(-1, 1))
        ax.plot(xcol, yhat, linewidth=1, c='orange', label=f"$\\beta_{{{colname}}}$")
        ax.text(min(xcol) * 1.02, max(y) * .95, f"$\\beta_{{{colname}}}$={r.coef_[0]:.3f}")


def marginal_catplot_(X, y, colname, targetname, ax, catnames, alpha=.1, show_xticks=True):
    catcodes, catnames_, catcode2name = getcats(X, colname, catnames)

    ax.scatter(X[colname].values, y.values, alpha=alpha, label=None, s=10)
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    # col = X[colname]
    # cats = np.unique(col)

    if show_xticks:
        ax.set_xticks(catcodes)
        ax.set_xticklabels(catnames_)
    else:
        ax.set_xticks([])

def plot_catstratpd_gridsearch(X, y, colname, targetname,
                               min_samples_leaf_values=(2, 5, 10, 20, 30),
                               min_y_shifted_to_zero=True,  # easier to read if values are relative to 0 (usually); do this for high cardinality cat vars
                               show_xticks=True,
                               show_impact=False,
                               show_all_cat_deltas=True,
                               catnames=None,
                               yrange=None,
                               sort='ascending',
                               style:('strip','scatter')='strip',
                               cellwidth=2.5,
                               cellheight=2.5):
    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(1, ncols + 1,
                             figsize=((ncols + 1) * cellwidth, cellheight))

    marginal_catplot_(X, y, colname, targetname, catnames=catnames, ax=axes[0], alpha=0.05,
                      show_xticks=show_xticks)
    axes[0].set_title("Marginal", fontsize=10)

    col = 1
    for msl in min_samples_leaf_values:
        #print(f"---------- min_samples_leaf={msl} ----------- ")
        if yrange is not None:
            axes[col].set_ylim(yrange)
        try:
            uniq_catcodes, avg_per_cat, ignored = \
                plot_catstratpd(X, y, colname, targetname, ax=axes[col],
                                min_samples_leaf=msl,
                                catnames=catnames,
                                yrange=yrange,
                                n_trees=1,
                                show_xticks=show_xticks,
                                show_impact=show_impact,
                                show_all_deltas=show_all_cat_deltas,
                                show_ylabel=False,
                                sort=sort,
                                style=style,
                                min_y_shifted_to_zero=min_y_shifted_to_zero)
        except ValueError:
            axes[col].set_title(f"Can't gen: leafsz={msl}", fontsize=8)
        else:
            axes[col].set_title(f"leafsz={msl}, ign'd={ignored / len(X):.1f}%", fontsize=9)
        col += 1


def catwise_leaves(rf, X_not_col, X_col, y, max_catcode):
    """
    Return a 2D array with the average y value for each category in each leaf
    normalized by subtracting the overall avg y value from all categories.

    The columns are the y avg value changes per cat found in a single leaf as
    they differ from the overall y average. Each row represents a category level. E.g.,

    row           leaf0       leaf1
     0       166.430176  186.796956
     1       219.590349  176.448626

    Cats are possibly noncontiguous with nan rows for cat codes not present. Not all
    values in a leaf column will be non-nan.  Only those categories mentioned in
    a leaf have values.
    Shape is (max cat + 1, num leaves).

    Previously, we subtracted the average of the leaf y not the overall y avg,
    but this failed to capture the relationship between categories when there are
    many levels.  Within a single leave, there will typically only be a few categories
    represented.
    """
    leaves = leaf_samples(rf, X_not_col)
    # y_mean = np.mean(y)

    leaf_histos = np.full(shape=(max_catcode+1, len(leaves)), fill_value=np.nan)
    refcats = np.empty(shape=(len(leaves),), dtype=int)

    # Rank the cat codes by most to least common and use the most common ref cat
    # in each leaf, given the cat codes available in the leaf.
    uniq_cats, cat_counts = np.unique(X_col, return_counts=True)
    revsort_idx = np.argsort(cat_counts)[::-1]
    cats_by_most_common = list(uniq_cats[revsort_idx])

    ignored = 0
    for leaf_i in range(len(leaves)):
        sample = leaves[leaf_i]
        leaf_cats = X_col[sample]
        leaf_y = y[sample]
        # perform a groupby(catname).mean()
        uniq_leaf_cats = np.unique(leaf_cats) # comes back sorted
        avg_y_per_cat = np.array([leaf_y[leaf_cats==cat].mean() for cat in uniq_leaf_cats])
        if len(uniq_leaf_cats) < 2:
            print(f"ignoring {len(sample)} obs for {len(avg_y_per_cat)} cat(s) in leaf")
            ignored += len(sample)
            continue

        # Find index of leaf cats in cats_by_most_common, then find min index, which
        # will correspond to most common category in X_col. Finally grab catcode
        leaf_cat_idxs = [list(cats_by_most_common).index(cat) for cat in leaf_cats]
        most_common_leaf_cat = cats_by_most_common[np.min(leaf_cat_idxs)]

        # Seems to help with grouping later by refcat for real X_col distributions
        # whereas subtracting min cat code helps to reverse sort with uniform distros.
        # Real data sets see fewer uniq cat codes after combining common refcats,
        # with increased numbers of points of course.
        refcats[leaf_i] = most_common_leaf_cat
        # refcats[leaf_i] = np.min(uniq_leaf_cats)

        # record avg y value per cat above avg y in this leaf
        # leave cats w/o representation as nan
        # Back to subtracting min of leaf_y
        # Always use smallest cat code as the reference point; since avg_y_per_cat
        # is sorted by uniq_leaf_cats, first value is the reference value
        delta_y_per_cat = avg_y_per_cat - avg_y_per_cat[0]
        # refcats[leaf_i] = uniq_leaf_cats[0]
        # Store into leaf i vector just those deltas we have data for
        leaf_histos[uniq_leaf_cats, leaf_i] = delta_y_per_cat

    return leaf_histos, refcats, ignored


def cat_partial_dependence(X, y,
                           colname,  # X[colname] expected to be numeric codes
                           max_catcode=None, # if we're bootstrapping, might see diff max's so normalize to one max
                           n_trees=1,
                           min_samples_leaf=10,
                           min_xxxxxxxxxdddddddddslopes_per_x=5,
                           max_features=1.0,
                           bootstrap=False,
                           supervised=True,
                           verbose=False):
    X_not_col = X.drop(colname, axis=1).values
    X_col = X[colname].values
    if (X_col<0).any():
        raise ValueError(f"Category codes must be > 0 in column {colname}")
    if max_catcode is None:
        max_catcode = np.max(X_col)
    if supervised:
        rf = RandomForestRegressor(n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
        rf.fit(X_not_col, y)
        if verbose:
            print(f"CatStrat Partition RF: dropping {colname} training R^2 {rf.score(X_not_col, y):.2f}")
    else:
        print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    min_samples_leaf=min_samples_leaf * 2,
                                    # there are 2x as many samples (X,X') so must double leaf size
                                    bootstrap=bootstrap,
                                    max_features=max_features,
                                    oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    # rf = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X_not_col, y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    # leaf_histos, leaf_avgs, leaf_sizes, leaf_catcounts, ignored = \
    #     catwise_leaves(rf, X, y, colname, verbose=verbose)

    leaf_histos, refcats, ignored____ = \
        catwise_leaves(rf, X_not_col, X_col, y.values, max_catcode)

    avg_per_cat, ignored = avg_values_at_cat(leaf_histos, refcats, verbose=verbose)

    # experimenting dropping those with too few averages
    # slope_counts_at_cat = leaf_histos.shape[1] - np.isnan(leaf_histos).sum(axis=1)
    # leaf_histos[slope_counts_at_cat<5,:] = np.nan # kill these

    if verbose:
        print(f"CatStratPD Num samples ignored {ignored} for {colname}")

    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     avg_per_cat = np.nanmean(leaf_histos, axis=1)
        # slope_counts_at_cat = leaf_histos.shape[1] - np.isnan(leaf_histos).sum(axis=1)

    # print("slope_counts_at_cat", colname, list(slope_counts_at_cat)[:100])
    # print("avg_per_cat", colname, list(avg_per_cat)[:100])

    return leaf_histos, avg_per_cat, ignored


def avg_values_at_cat_via_graph(leaf_histos, refcats, verbose=False):
    """
    In leaf_histos, we have information from the leaves indicating how much
    above or below each category was from the reference category of that leaf.
    The reference category is the one with the minimum cat (not y) value, so it's
    relative value in the leaf column will be 0. Categories not mentioned in the leaf,
    will have nan values in the column.

    The goal is to combine all of these relative category bumps and drops,
    despite the fact that they do not have the same reference category. We
    collect all of the leaves with a reference category level i and average them
    together (for all unique categories mentioned in refcats).  Now we have
    a list of relative value vectors, one per unique category level used as a reference.
    The list is sorted in order of unique reference category. (Hopefully this
    will be much smaller than the number of categories total for speed.) Note these
    sum vectors might have np.nan values to represent unknown category info.

    Now we have to create a result vector that combines the relative vectors.
    The problem is of course the different reference categories but if the vector for
    a unique refcat, i, is mentioned in another vector whose refcat is less than i,
    then we can use transitivity to combine.  E.g., for refcat 'a', we might have
    vector [0, 2, 5] indicating b=a+2, c=a+5.  We then have refcat 'b' vector:
    [nan, 0, 3] indicating c=b+3. The solution is to describe the relationships
    with a graph and then use "avg of sum of weights along all paths" from u to v to
    get the value of v relative to u:

    a -2-> b --|
    |          3
    |--5-> c <-|

    Walk all leaf vectors and combine to get an average for each unique refcat. From
    these, we build a graph G.  For each avg vector, for all i positions > refcat,
    add an edge from refcat -> i with weight avg_for_refcat[cat].
    """
    def nanaddvectors(A):
        "Add all vertical vectors in A but support nan+x==x and nan+nan=nan"
        s = np.nansum(A, axis=1)
        # count how many non-nan values and non-0 values across all leaves with
        # cat as reference category
        all_nan_entries = np.isnan(A)
        # if all entries for a cat are nan, make sure sum s is nan for that cat
        s[all_nan_entries.all(axis=1)] = np.nan
        return s

    ignored = 0
    avg_per_cat = None

    uniq_refcats = sorted(np.unique(refcats))
    if verbose: print("uniq_refcats =", uniq_refcats)
    # Track
    sums_for_refcats = []
    counts_for_refcats = []
    for cat in uniq_refcats:
        # collect and add up vectors from all leaves with cat as a reference category
        leaves_with_same_refcat = leaf_histos[:, np.where(refcats == cat)[0]]
        all_nan_entries = np.isnan(leaves_with_same_refcat)
        # if all entries for a cat are nan, make sure sum s is nan for that cat
        s = nanaddvectors(leaves_with_same_refcat)
        # count how many non-nan values and non-0 values across all leaves with
        # cat as reference category
        c = (~all_nan_entries).astype(int) # nan entries also get 0 count
        c[cat] = 0 # refcat doesn't get counted
        c = np.sum(c, axis=1)
        counts_for_refcats.append(c)
        sums_for_refcats.append(s)
    if verbose: print("sums_for_refcats (reordered by uniq_refcats)\n", np.array(sums_for_refcats).T)
    if verbose: print("counts\n", np.array(counts_for_refcats).T)
    # likely less memory to avoid creating 2D matrices
    avg_for_refcats = [sums_for_refcats[i] / np.where(counts_for_refcats[i]==0, 1, counts_for_refcats[i]) for i in range(len(uniq_refcats))]

    for cat in uniq_refcats:
        v = avg_for_refcats[cat]
        notnan = ~np.isnan(v)
        # beyond_cat = v[notnan]>
        for i in range(cat,len(v)):
            if np.isnan(v[i]): continue
            print("edge", v[cat], '-', v[i],'->', i)

    return avg_per_cat, ignored


def avg_values_at_cat(leaf_histos, refcats, verbose=False):
    """
    In leaf_histos, we have information from the leaves indicating how much
    above or below each category was from the reference category of that leaf.
    The reference category is the one with the minimum cat code (not y value), so the
    refcat's relative value in the leaf column will be 0. Categories not mentioned
    in the leaf, will have nan values in that column.

    The goal is to combine all of these relative category bumps and drops,
    despite the fact that they do not have the same reference category. We
    collect all of the leaves with a reference category level i and average them
    together (for all unique categories mentioned in min_cats).  Now we have
    a list of relative value vectors, one per category level used as a reference.

    The list is sorted in order of unique reference category. (Hopefully this
    will be much smaller than the number of categories total for speed.) Note these
    sum vectors might have np.nan values to represent unknown category info.
    I set all refcat values to np.nan to ease computation then set the smallest
    refcat relative value to 0 right before function exit.  This sorting is important
    so data can feed forward; refcat i uses the value of refcat i in the running sum
    of refcat vectors. (see 2nd loop)

    Now we have to create a result vector, sums_per_cat, that combines the
    relative vectors. The problem is of course the different reference categories.
    We initialize sums_per_cat to be the average relative to the first unique
    reference category. Let's assume that the first refcat is 0, which means we take
    the first element from the avg_for_refcats list to initialize sums_per_cat. To add
    in the next vector, we first have to compensate for the difference in
    reference category. refcats[i] tells us which category the vector is
    relative to so we take the corresponding value from the running sum, sums_per_cat,
    at position refcats[i] and add that to all elements of the avg_for_refcats[i]
    vector.

    BTW, it's possible that more than a single value within a leaf_histos vector will be 0.
    I.e., the reference category value is always 0 in the vector, but there might be
    another category whose value was the same y, giving a 0 relative value. I set them
    to nan, however, when combining histos for same refcat.

    Example:

    refcats: [0,1]

    sums_for_refcats
     [[nan nan]
     [ 1. nan]
     [ 2.  3.]
     [nan  2.]
     [ 0. nan]
     [nan nan]]

    counts
     [[0 0]
     [1 0]
     [1 1]
     [0 1]
     [1 0]
     [0 0]]

    Then to combine, we see a loop with an iteration per unique min cat:

    0 : initial  = [nan  1.  2. nan  0. nan] 	sums_per_cat = [nan  1.  2. nan  0. nan]
    1 : adjusted = [nan nan  4.  3. nan nan] 	sums_per_cat = [nan  1.  6.  3.  0. nan]

    Then divide by

    So we get a final avg per cat of:  [ 0.  1.  3.  3.  0. nan]

    :param leaf_histos: A 2D matrix where rows are category levels/values and
                        columns hold y values for categories.
    :param refcats: For each leaf, we must know what category was used as the reference.
                     I.e., which category had the smallest y value in the leaf?
    :return:
    """
    def nanmerge_vectors(a,b):
        "Add two vectors a+b but support nan+x==x and nan+nan=nan"
        all_nan_entries = np.isnan(a) & np.isnan(b)
        c = np.where(np.isnan(a), 0, a) + np.where(np.isnan(b), 0, b)
        # if adding nan to nan, leave as nan
        c[all_nan_entries] = np.nan
        return c

    def nanmerge_matrix_cols(A):
        "Add all vertical vectors in A but support nan+x==x and nan+nan=nan"
        s = np.nansum(A, axis=1)
        all_nan_entries = np.isnan(A)
        # if all entries for a cat are nan, make sure sum s is nan for that cat
        s[all_nan_entries.all(axis=1)] = np.nan
        return s

    def zero_as_one(a):
        return np.where(a == 0, 1, a)

    def parray(a):
        if type(a[0])==np.int64:
            return '[ ' + (' '.join([f"{x:6d}" for x in a])).strip() + ' ]'
        else:
            return '[ ' + (' '.join([f"{x:6.2f}" for x in a])).strip() + ' ]'
    def parray3(a):
        return '[ ' + (' '.join([f"{x:6.3f}" for x in a])).strip() + ' ]'

    # FIRST LOOP COMBINES LEAF VECTORS WITH SAME REFCAT
    uniq_refcats = sorted(np.unique(refcats))
    if verbose:
        print("refcats =", refcats)
        print("uniq_refcats =", uniq_refcats)
        # print("leaf_histos\n", leaf_histos[0:30])
        # print("leaf_histos reordered by refcat order\n", leaf_histos[0:30,np.argsort(refcats)])
    sums_for_refcats = []
    counts_for_refcats = []
    for cat in uniq_refcats:
        # collect and add up vectors from all leaves with cat as the reference category
        leaves_with_same_refcat = leaf_histos[:, np.where(refcats == cat)[0]]
        # Turn all refcat locations from 0 to np.nan for computation purposes
        # (reset first refcat value to 0 at end)
        # TODO: not sure we need
        leaves_with_same_refcat[cat] = np.nan
        s = nanmerge_matrix_cols(leaves_with_same_refcat)
        # count how many non-nan values values across all leaves with cat as ref category
        c = (~np.isnan(leaves_with_same_refcat)).astype(int)
        c = np.sum(c, axis=1)
        counts_for_refcats.append(c)
        sums_for_refcats.append(s)

    # FROM SUMS FOR REFCATS, COMPUTE AVERAGE
    avg_for_refcats = [sums_for_refcats[j] / zero_as_one(counts_for_refcats[j])
                       for j in range(len(uniq_refcats))]
    avg_for_refcats = np.array(avg_for_refcats).T

    # SORT REV BY WEIGHT
    # E.g., weight_for_refcats = [ 8  8  6  4  4 27  5 11  5  5  4 34 17  8  9  3  5]
    counts_for_refcats = np.array(counts_for_refcats).T
    weight_for_refcats = np.sum(counts_for_refcats, axis=0)
    # TODO: not sure it's worth sorting yet
    uniq_refcats_by_weight_idxs = np.argsort(weight_for_refcats)[::-1]
    avg_for_refcats = avg_for_refcats[:,uniq_refcats_by_weight_idxs]
    weight_for_refcats = weight_for_refcats[uniq_refcats_by_weight_idxs]

    if verbose:
        print("counts\n", counts_for_refcats[0:30])
        cats_with_values_count = np.sum(counts_for_refcats, axis=1)
        nonzero_idx = np.where(cats_with_values_count>0)[0]
        print("counts per cat>0\n", cats_with_values_count[nonzero_idx])
        # print("counts per cat\n", counts_for_refcats[np.where(np.sum(counts_for_refcats, axis=1)>0)[0]])
        print("refcat weights\n", weight_for_refcats)
        # print("sums_for_refcats (reordered by weight)\n", np.array(sums_for_refcats).T[:30])
        print("avgs per refcat\n", avg_for_refcats[0:30])


    # SECOND LOOP SUMS COMBINED VECTORS USING RELATIVE VALUE FROM RUNNING SUM
    """
    avgs per refcat
     [[   nan    nan    nan    nan    nan    nan]
      [-30.35    nan    nan    nan    nan    nan]
      [  3.5     nan    nan    nan    nan    nan]
      [ -2.78  28.46  -6.05    nan    nan    nan]
      [ -1.23    nan  -4.82    nan    nan    nan]
      [-15.84  14.94 -19.26    nan    nan    nan]
      [-13.9   16.9  -16.57    nan -12.85    nan]
      [   nan  21.98 -11.39    nan  -7.29    nan]
      [  9.43    nan   5.7   11.41  10.43  26.44]
      [ -0.23  29.57  -3.35    nan   0.78  17.04]]
     
    refcat weights
      [23 10 17  1  7  2]
    """
    # catavg is the running sum vector
    n = leaf_histos.shape[0]
    catavg = avg_for_refcats[:,0] # init with first ref category (column)
    ignored = 0
    # Need a way to restrict
    valid_idxs = np.where(weight_for_refcats>=10)[0]
    last_refcat = -1
    if len(valid_idxs)>0:
        last_refcat = valid_idxs[-1]
    # last_refcat = len(uniq_refcats)
    for j in range(1,last_refcat):      # for each refcat, avg in the vectors
        cat = uniq_refcats[j]
        relative_to_value = catavg[cat]
        v = avg_for_refcats[:,j]
        if np.isnan(relative_to_value):
            ignored += np.sum(~np.isnan(v))
            weight_for_refcats[j] = 0 # wipe out weights as we don't count these
            if verbose: print(f"cat {cat} has no value in running sum; ignored={ignored}")
            continue
        # TODO: can start at uniq_refcats[j]+1 right?
        for i in range(n): # walk down a vector
            if np.isnan(catavg[i]) and np.isnan(v[i]): # both nan
                continue
            if np.isnan(v[i]): # new vector is nan, just used old value
                continue
            # computed weighted average of two values
            prev_weight = np.sum(weight_for_refcats[0:j])
            cur_weight  = weight_for_refcats[j]
            v_ = v[i] + relative_to_value
            catavg[i] = (catavg[i] * prev_weight + v_ * cur_weight) / (prev_weight+cur_weight)

    catavg[uniq_refcats[0]] = 0.0 # first refcat always has value 0 (was nan for summation purposes)
    if verbose: print("final cat avgs", parray3(catavg))
    return catavg, ignored

"""
    # SECOND LOOP SUMS COMBINED VECTORS USING RELATIVE VALUE FROM RUNNING SUM
    # sums_per_cat is the running sum vector
    sums_per_cat = avg_for_refcats[0] # init with first ref category (column)
    ignored = 0
    cats_with_values_added_to_running_sum = (~np.isnan(avg_for_refcats[0])).astype(int)
    count_per_cat = cats_with_values_added_to_running_sum * weight_for_refcats[0]
    if verbose:
        print(f"{uniq_refcats[0]:-2d} : initial    =",parray(sums_per_cat))
        print("     weights    =", parray(count_per_cat),"\n")
    for i in range(1,len(uniq_refcats)):
        # Compensate for different reference category by adding the value
        # of this vector's reference category from the running sum
        cat = uniq_refcats[i]
        relative_to_value = sums_per_cat[cat]
        if np.isnan(relative_to_value):
            ignored += np.sum(~np.isnan(avg_for_refcats[i]))
            if verbose: print(f"cat {cat} has no value in running sum; ignored={ignored}")
            continue

        both_not_nan = (~np.isnan(avg_for_refcats[i]) & ~np.isnan(sums_per_cat)).astype(int)
        weights = both_not_nan * weight_for_refcats[i]
        prev_weights = both_not_nan * np.sum(weight_for_refcats[0:i])

        adjusted_vec = relative_to_value + avg_for_refcats[i]
        adjusted_vec = adjusted_vec * zero_as_one(weights)
        adjusted_vec[cat] = np.nan

        cur_sum = sums_per_cat * zero_as_one(prev_weights)

        cats_with_values_added_to_running_sum = (~np.isnan(adjusted_vec)).astype(int)
        count_per_cat += cats_with_values_added_to_running_sum
        # sums_per_cat = nanmerge_vectors(sums_per_cat, adjusted_vec)
        # Add weighted adjusted vec to current sum, then back to average
        sums_per_cat = (nanmerge_vectors(cur_sum, adjusted_vec)) / np.sum(weight_for_refcats[0:i+1])
        if verbose:
            print(f"{cat:-2d} : vec to add =", parray(avg_for_refcats[i]), f" + {relative_to_value:.2f}")
            # print("     count      =", parray(cats_with_values_added_to_running_sum * weight_for_refcats[i]))
            print("     adjusted   =", parray(avg_for_refcats[i]+relative_to_value))
            print("     weights    =", parray(weights))
            print("     weighted   =", parray(adjusted_vec))
            print("     prev wghtd =", parray(cur_sum))
            print("     new sum    =", parray(nanmerge_vectors(cur_sum, adjusted_vec)))
            print("     new avg    =", parray(sums_per_cat))
            print()

    # We've added vectors together for len(uniq_refcats), so get a generic average
    # avg_per_cat = sums_per_cat / zero_as_one(count_per_cat)
    avg_per_cat = sums_per_cat / np.sum(weight_for_refcats)
    avg_per_cat[uniq_refcats[0]] = 0.0 # first refcat always has value 0 (was nan for summation purposes)
    if verbose: print("final cat avgs", parray3(avg_per_cat))
    return avg_per_cat, ignored
    # avg_for_refcats[0][0] = 0.0
    # return avg_for_refcats[0], ignored
"""

def plot_catstratpd(X, y,
                    colname,  # X[colname] expected to be numeric codes
                    targetname,
                    catnames=None,  # map of catcodes to catnames; converted to map if sequence passed
                    # must pass dict or series if catcodes are not 1..n contiguous
                    # None implies use np.unique(X[colname]) values
                    # Must be 0-indexed list of names if list
                    n_trials=10,
                    ax=None,
                    sort='ascending',
                    n_trees=1,
                    min_samples_leaf=10,
                    max_features=1.0,
                    bootstrap=False,
                    yrange=None,
                    title=None,
                    supervised=True,
                    use_weighted_avg=False,
                    show_impact=False,
                    show_all_deltas=True,
                    show_x_counts=True,
                    impact_color='#D73028',
                    alpha=.15,
                    color='#2c7fb8',
                    pdp_marker_size=3,
                    pdp_marker_alpha=.6,
                    marker_size=5,
                    pdp_color='black',
                    fontname='Arial',
                    title_fontsize=11,
                    label_fontsize=10,
                    barchart_size=0.20,
                    barchar_alpha=0.9,
                    ticklabel_fontsize=10,
                    min_y_shifted_to_zero=True,
                    # easier to read if values are relative to 0 (usually); do this for high cardinality cat vars
                    show_xlabel=True,
                    show_xticks=True,
                    show_ylabel=True,
                    verbose=False,
                    figsize=(5,3)):
    """
    Warning: cat columns are assumed to be label encoded as unique integers. This
    function uses the cat code as a raw index internally. So if you have two cat
    codes 1 and 1000, this function allocates internal arrays of size 1000+1.

    only works for ints, not floats
    """
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

    uniq_catcodes = np.unique(X[colname])
    max_catcode = max(uniq_catcodes)

    X_col = X[colname]
    n = len(X_col)

    def avg_pd_catvalues(all_avg_per_cat):
        m = np.zeros(shape=(max_catcode+1,))
        c = np.zeros(shape=(max_catcode+1,), dtype=int)

        # For each unique catcode, sum and count avg_per_cat values found among trials
        for i in range(n_trials):
            avg_per_cat = all_avg_per_cat[i]
            catcodes = np.where(~np.isnan(avg_per_cat))[0]
            for code in catcodes:
                m[code] += avg_per_cat[code]
                c[code] += 1
        # Convert to average value per cat
        for code in np.where(m!=0)[0]:
            m[code] /= c[code]
        m = np.where(c==0, np.nan, m) # cats w/o values should be nan, not 0
        return m

    all_avg_per_cat = []
    ignored = 0
    for i in range(n_trials):
        # idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
        if n_trials>1:
            idxs = resample(range(n), n_samples=int(n*2/3), replace=False) # subset
            X_, y_ = X.iloc[idxs], y.iloc[idxs]
        else:
            X_, y_ = X, y

        leaf_histos, avg_per_cat, ignored_ = \
            cat_partial_dependence(X_, y_,
                                   max_catcode=np.max(X_col),
                                   colname=colname,
                                   n_trees=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   bootstrap=False,
                                   verbose=verbose)
        # avg_per_cat is currently deltas from mean(y) but we want to use min avg_per_cat
        # as reference point, zeroing that one out. All others will be relative to that
        # min cat value
        if min_y_shifted_to_zero:
            # min and mean don't work, though min is closest. really need average of
            # first few valid avg_per_cat values.
            avg_per_cat -= np.nanmin(avg_per_cat)
            # avg_per_cat += 0#np.mean(y)
        ignored += ignored_
        all_avg_per_cat.append( avg_per_cat )

    ignored /= n_trials # average number of x values ignored across trials

    combined_avg_per_cat = avg_pd_catvalues(all_avg_per_cat)

    impacts = [np.nanmean(np.abs(all_avg_per_cat[i])) for i in range(n_trials)]
    impact_order = np.argsort(impacts)
    print("impacts", impacts)
    avg_impact = np.nanmean(np.abs(combined_avg_per_cat))
    print("avg impact", avg_impact)

    cmap = plt.get_cmap('coolwarm')
    colors=cmap(np.linspace(0, 1, num=n_trials))
    min_y = 9999999999999
    max_y = -min_y
    for i in range(n_trials):
        avg_per_cat = all_avg_per_cat[i]
        if np.nanmin(avg_per_cat) < min_y:
            min_y = np.nanmin(avg_per_cat)
        if np.nanmax(avg_per_cat) > max_y:
            max_y = np.nanmax(avg_per_cat)
        trial_catcodes = np.where(~np.isnan(avg_per_cat))[0]
        # print("catcodes", trial_catcodes, "range", min(trial_catcodes), max(trial_catcodes))
        # walk each potential catcode but plot with x in 0..maxcode+1; ignore nan avg_per_cat values
        xloc = -1 # go from 0 but must count nan entries
        collect_cats = []
        collect_deltas = []
        for cat in uniq_catcodes:
            cat_delta = avg_per_cat[cat]
            xloc += 1
            if np.isnan(cat_delta): continue
            # ax.plot([xloc - .15, xloc + .15], [cat_delta] * 2, c=colors[impact_order[i]], linewidth=1)
            collect_cats.append(xloc)
            collect_deltas.append(cat_delta)
        # print("Got to xloc", xloc, "len(trial_catcodes)", len(trial_catcodes), "len(catcodes)", len(uniq_catcodes))
        # ax.scatter(collect_cats, collect_deltas, c=mpl.colors.rgb2hex(colors[impact_order[i]]),
        #            s=pdp_marker_size, alpha=pdp_marker_alpha)
        ax.plot(collect_cats, collect_deltas, '.', c=mpl.colors.rgb2hex(colors[impact_order[i]]),
                markersize=pdp_marker_size, alpha=pdp_marker_alpha)

    # show 0 line
    # ax.plot([0,len(uniq_catcodes)], [0,0], '--', c='grey', lw=.5)

    # Show avg line
    xloc = 0
    avg_delta = []
    for cat in uniq_catcodes:
        cat_delta = combined_avg_per_cat[cat]
        avg_delta.append(cat_delta)
        xloc += 1

    if n_trials>1:
        # Show combined cat values if more than one trials
        ax.plot(range(len(uniq_catcodes)), avg_delta, '.', c='k', markersize=pdp_marker_size + 1)

    if show_impact:
        ax.text(0.5, .94, f"Impact {avg_impact:.2f}",
                horizontalalignment='center',
                fontsize=label_fontsize, fontname=fontname,
                transform=ax.transAxes,
                color=impact_color)

    leave_room_scaler = 1.3

    if yrange is not None:
        ax.set_ylim(*yrange)

    if show_x_counts:
        # Only show cat counts for those which are present in X[colname] (unlike stratpd plot)
        _, cat_counts = np.unique(X_col[np.isin(X_col, uniq_catcodes)], return_counts=True)
        # x_width = len(uniq_catcodes)
        # count_bar_width = x_width / len(pdpx)
        # if count_bar_width/x_width < 0.002:
        #     count_bar_width = x_width * 0.002 # don't make them so skinny they're invisible
        count_bar_width=1
        ax2 = ax.twinx()
        # scale y axis so the max count height is 10% of overall chart
        ax2.set_ylim(0, max(cat_counts) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(cat_counts)]))
        ax2.bar(x=range(len(uniq_catcodes)), height=cat_counts, width=count_bar_width,
                facecolor='#BABABA', align='center', alpha=barchar_alpha)
        ax2.set_ylabel(f"$x$ point count", labelpad=-12, fontsize=label_fontsize,
                       fontstretch='extra-condensed',
                       fontname=fontname)
        # shift other y axis down barchart_size to make room
        if yrange is not None:
            ax.set_ylim(yrange[0]-(yrange[1]-yrange[0])*barchart_size * leave_room_scaler, yrange[1])
        else:
            ax.set_ylim(min_y-(max_y-min_y)*barchart_size * leave_room_scaler, max_y)
        # ax2.set_xticks(range(len(uniq_catcodes)))
        # ax2.set_xticklabels([])
        plt.setp(ax2.get_xticklabels(), visible=False)
        # ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
        # for tick in ax2.get_xticklabels():
        #     tick.set_visible(False)
        for tick in ax2.get_yticklabels():
            tick.set_fontname(fontname)
        ax2.spines['top'].set_linewidth(.5)
        ax2.spines['right'].set_linewidth(.5)
        ax2.spines['left'].set_linewidth(.5)
        ax2.spines['bottom'].set_linewidth(.5)

    # np.where(combined_avg_per_cat!=0)[0]
    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

    if show_xticks:
        ax.set_xticks(range(len(uniq_catcodes)))
        if catnames is not None:
            ax.set_xticklabels(catnames[uniq_catcodes])
        else:
            ax.set_xticklabels(uniq_catcodes)
        for tick in ax.get_xticklabels():
            tick.set_fontname(fontname)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    if show_xlabel:
        ax.set_xlabel(colname, fontsize=label_fontsize, fontname=fontname)
    if show_ylabel:
        ax.set_ylabel(targetname, fontsize=label_fontsize, fontname=fontname)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    return uniq_catcodes, combined_avg_per_cat, ignored

# only works for ints, not floats
def plot_catstratpd_OLD(X, y,
                    colname,  # X[colname] expected to be numeric codes
                    targetname,
                    catnames=None,  # map of catcodes to catnames; converted to map if sequence passed
                    # must pass dict or series if catcodes are not 1..n contiguous
                    # None implies use np.unique(X[colname]) values
                    # Must be 0-indexed list of names if list
                    n_trials=10,
                    ax=None,
                    sort='ascending',
                    n_trees=1,
                    min_samples_leaf=10,
                    max_features=1.0,
                    bootstrap=False,
                    yrange=None,
                    title=None,
                    supervised=True,
                    use_weighted_avg=False,
                    show_impact=False,
                    show_all_deltas=True,
                    show_x_counts=True,
                    alpha=.15,
                    color='#2c7fb8',
                    impact_color='#D73028',
                    pdp_marker_size=.5,
                    marker_size=5,
                    pdp_color='black',
                    fontname='Arial',
                    title_fontsize=11,
                    label_fontsize=10,
                    ticklabel_fontsize=10,
                    style:('strip','scatter')='strip',
                    min_y_shifted_to_zero=True,  # easier to read if values are relative to 0 (usually); do this for high cardinality cat vars
                    show_xlabel=True,
                    show_ylabel=True,
                    show_xticks=True,
                    verbose=False,
                    figsize=None):
    """
    Warning: cat columns are assumed to be label encoded as unique integers. This
    function uses the cat code as a raw index internally. So if you have two cat
    codes 1 and 1000, this function allocates internal arrays of size 1000+1.

    :param X:
    :param y:
    :param colname:
    :param targetname:
    :param catnames:
    :param ax:
    :param sort:
    :param n_trees:
    :param min_samples_leaf:
    :param max_features:
    :param bootstrap:
    :param yrange:
    :param title:
    :param supervised:
    :param use_weighted_avg:
    :param alpha:
    :param color:
    :param pdp_marker_size:
    :param marker_size:
    :param pdp_color:
    :param style:
    :param min_y_shifted_to_zero:
    :param show_xlabel:
    :param show_ylabel:
    :param show_xticks:
    :param verbose:
    :return:
    """

    catcodes, _, catcode2name = getcats(X, colname, catnames)

    all_avg_per_cat = []
    all_pdpy = []
    n = len(X)
    ignored = 0
    for i in range(n_trials):
        # idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
        idxs = resample(range(n), n_samples=int(n * 2 / 3), replace=False)  # subset
        X_, y_ = X.iloc[idxs], y.iloc[idxs]

        leaf_histos, avg_per_cat, ignored = \
            cat_partial_dependence(X_, y_,
                                   colname=colname,
                                   n_trees=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   bootstrap=bootstrap,
                                   supervised=supervised,
                                   use_weighted_avg=use_weighted_avg,
                                   verbose=verbose)
        # all_avg_per_cat =

    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

    ncats = len(catcodes)
    nleaves = leaf_histos.shape[1]

    sorted_catcodes = catcodes
    if sort == 'ascending':
        sorted_indexes = avg_per_cat[~np.isnan(avg_per_cat)].argsort()
        sorted_catcodes = catcodes[sorted_indexes]
    elif sort == 'descending':
        sorted_indexes = avg_per_cat.argsort()[::-1]  # reversed
        sorted_catcodes = catcodes[sorted_indexes]

    min_avg_value = 0
    # The category y deltas straddle 0 but it's easier to understand if we normalize
    # so lowest y delta is 0
    if min_y_shifted_to_zero:
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
        if show_all_deltas:
            ax.scatter(x_noise + xloc, leaf_histos[catcode2name[cat]] - min_avg_value,
                       alpha=alpha, marker='o', s=marker_size,
                       c=color)
        if style == 'strip':
            ax.plot([xloc - .1, xloc + .1], [avg_per_cat[catcode2name[cat]]-min_avg_value] * 2,
                    c='black', linewidth=2)
        else:
            ax.scatter(xloc, avg_per_cat[catcode2name[cat]]-min_avg_value, c=pdp_color, s=pdp_marker_size)
        xloc += 1

    ax.set_xticks(range(0, ncats))
    if show_xticks: # sometimes too many
        ax.set_xticklabels(catcode2name[sorted_catcodes])
        ax.tick_params(axis='x', which='major', labelsize=ticklabel_fontsize)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='major', labelsize=ticklabel_fontsize, bottom=False)
    ax.tick_params(axis='y', which='major', labelsize=ticklabel_fontsize)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    if show_x_counts:
        X_col = X[colname]
        _, pdpx_counts = np.unique(X_col[np.isin(X_col, pdpx)], return_counts=True)

        ax2 = ax.twinx()
        # scale y axis so the max count height is 10% of overall chart
        ax2.set_ylim(0, max(pdpx_counts) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(pdpx_counts)]))
        ax2.bar(x=pdpx, height=pdpx_counts, width=count_bar_width,
                facecolor='#BABABA', align='edge', alpha=barchar_alpha)
        ax2.set_ylabel(f"$x$ point count", labelpad=-12, fontsize=label_fontsize,
                       fontstretch='extra-condensed',
                       fontname=fontname)
        # shift other y axis down barchart_size to make room
        if yrange is not None:
            ax.set_ylim(yrange[0]-(yrange[1]-yrange[0])*barchart_size, yrange[1])
        else:
            ax.set_ylim(min_y-(max_y-min_y)*barchart_size, max_y)
        ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
        for tick in ax2.get_xticklabels():
            tick.set_fontname(fontname)
        for tick in ax2.get_yticklabels():
            tick.set_fontname(fontname)
        ax2.spines['top'].set_linewidth(.5)
        ax2.spines['right'].set_linewidth(.5)
        ax2.spines['left'].set_linewidth(.5)
        ax2.spines['bottom'].set_linewidth(.5)

    if show_impact:
        m = np.nanmean(np.abs(avg_per_cat))
        ax.plot([0, len(sorted_catcodes)], [m,m], '--', lw=.7, c=impact_color)
        # add a tick for the mean in y axis
        # ax.set_yticks(list(ax.get_yticks()) + [m])
        ax.text(0.5, .94, f"Impact {m:.2f}",
                horizontalalignment='center',
                fontsize=label_fontsize, fontname=fontname,
                transform=ax.transAxes,
                color=impact_color)

    if show_xlabel:
        ax.set_xlabel(colname, fontsize=label_fontsize, fontname=fontname)
    if show_ylabel:
        ax.set_ylabel(targetname, fontsize=label_fontsize, fontname=fontname)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

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
        X_rand[:,col] = np.random.choice(X[:,col], len(X), replace=True)
    return X_rand


def df_scramble(X : pd.DataFrame) -> pd.DataFrame:
    """
    From Breiman: https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    "...the first coordinate is sampled from the N values {x(1,n)}. The second
    coordinate is sampled independently from the N values {x(2,n)}, and so forth."
    """
    X_rand = X.copy()
    for colname in X:
        X_rand[colname] = X_rand[colname].sample(frac=1.0)
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
