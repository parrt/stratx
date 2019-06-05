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
# from scipy.stats.rv_continuous import entropy as scipy_entropy

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
        leaf_samples.extend(sample_idxs_in_leaf) # add [...sample idxs...] for each leaf
    return leaf_samples


def dtree_leaf_samples(dtree, X:np.ndarray):
    leaf_ids = dtree.apply(X)
    d = pd.DataFrame(leaf_ids, columns=['leafid'])
    d = d.reset_index() # get 0..n-1 as column called index so we can do groupby
    sample_idxs_in_leaf = d.groupby('leafid')['index'].apply(lambda x: x.values)
    return sample_idxs_in_leaf


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

def hires_slopes_from_one_leaf(x:np.ndarray, y:np.ndarray, hires_min_samples_leaf:int):
    start = time.time()
    X = x.reshape(-1,1)

    r2s = []
    r2xNs = []
    allr = (np.min(x), np.max(x))

    # hires_min_samples_leaf = int( (np.max(x) - np.min(x)) * hires_window_width )
    # print(f"setting hires_min_samples_leaf = {hires_min_samples_leaf}")

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
    # print(f"{len(leaves)} leaves")
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []
    # leaf_yranges = []
    for samples in leaves:
        leaf_x = X[samples]
        leaf_y = y[samples]
        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            # print(f"Ignoring range {r} from {leaf_x.T} -> {leaf_y}")
            continue
        lm = LinearRegression()
        lm.fit(leaf_x.reshape(-1, 1), leaf_y)
        leaf_slopes.append(lm.coef_[0])
        r2 = lm.score(leaf_x.reshape(-1, 1), leaf_y)

        r2s.append(r2)
        r2xNs.append(r2 * len(leaf_x))
        # print(f"\tHIRES {len(leaf_x)} obs, leaf R^2 {r2:.2f}, R^2*n {r2*len(leaf_x):.2f}, range {allr}")
        if dbg:
            px = np.linspace(r[0], r[1], 20)
            plt.plot(px, lm.predict(px.reshape(-1,1)), lw=.5, c='blue', label=f"R^2 {r2:.2f}")

        leaf_r2.append(r2)
        leaf_xranges.append(r)
        # leaf_yranges.append((leaf_y[0], leaf_y[-1]))

    # print(f"\tAvg leaf R^2 {np.mean(r2s):.4f}, avg x len {np.mean(r2xNs)}")

    if dbg:
        plt.legend(loc='upper left', borderpad=0, labelspacing=0)
        plt.show()

    stop = time.time()
    # print(f"hires_slopes_from_one_leaf {stop - start:.3f}s")
    return leaf_xranges, leaf_slopes, leaf_r2


# def entropy(x, base=np.e):
#     _, counts = np.unique(x, return_counts=True)
#     return scipy_entropy(counts, base=base)


def collect_leaf_slopes(rf, X, y, colname,
                        hires_r2_threshold,
                        hires_n_threshold,
                        hires_min_samples_leaf):
    """
    For each leaf of each tree of the random forest rf (trained on all features
    except colname), get the samples then isolate the column of interest X values
    and the target y values. Perform a regression to get the slope of X[colname] vs y.
    We don't need to subtract the minimum y value before regressing because
    the slope won't be different. (We are ignoring the intercept of the regression line).

    Return for each leaf, the range of X[colname] and associated slope for that range.
    """
    start = time.time()
    # ci = X.columns.get_loc(colname)
    leaf_slopes = []
    leaf_r2 = []
    leaf_xranges = []

    allr = (np.min(X[colname]), np.max(X[colname]))

    leaves = leaf_samples(rf, X.drop(colname, axis=1))
    # print(f"{len(leaves)} leaves in T")
    for samples in leaves:
        one_leaf_samples = X.iloc[samples]
        leaf_x = one_leaf_samples[colname].values
        leaf_y = y.iloc[samples].values

        r = (np.min(leaf_x), np.max(leaf_x))
        if np.isclose(r[0], r[1]):
            # print(f"ignoring xleft=xright @ {r[0]}")
            continue

        lm = LinearRegression()
        lm.fit(leaf_x.reshape(-1, 1), leaf_y)
        r2 = lm.score(leaf_x.reshape(-1, 1), leaf_y)

        # rpercent = (r[1] - r[0]) * 100.0 / (allr[1] - allr[0])
        # print(f"{len(leaf_x)} obs, R^2 y ~ X[{colname}] = {r2:.2f}, in range {r} is {rpercent:.2f}%")

        if r2 < hires_r2_threshold and len(leaf_x) > hires_n_threshold: # if linear model for y ~ X[colname] is too crappy, go hires
            print(f"BIG {len(leaf_x)}, R^2 of y ~ X[{colname}] = {r2:.2f} < {hires_r2_threshold}!!!")
            leaf_xranges_, leaf_slopes_, leaf_r2_ = \
                hires_slopes_from_one_leaf(leaf_x, leaf_y, hires_min_samples_leaf=hires_min_samples_leaf)

            if len(leaf_slopes_)>0:
                leaf_slopes.extend(leaf_slopes_)
                leaf_r2.extend(leaf_r2_)
                leaf_xranges.extend(leaf_xranges_)
                continue
            else:
                # sounds like hires_min_samples_leaf is too small and values are ints;
                # e.g., hires_min_samples_leaf=.05 and x range of 1..10. If even spread,
                # hires_min_samples_leaf will get single x value, which can't tell us about
                # change in y over x as x isn't changing. Fall back onto non-hires
                pass # keep going as if this hadn't happened

        #print(f"All R^2 {LM_r2:.3f} for {len(leaf_x)} samples with {LM.coef_[ci]:.2f} beta vs Lx,Ly beta {lm.coef_[0]:.2f}")

        leaf_slopes.append(lm.coef_[0]) # better to use univariate slope it seems
        r2 = lm.score(leaf_x.reshape(-1, 1), leaf_y)
        # print(f"Entropy of not {colname}")
        # for i in range(len(X.columns)):
        #     e = entropy(one_leaf_samples.iloc[:,i].values)
        #     print(f"\tentropy of col {X.columns[i]}[{len(leaf_x)}] values = {e:.2f}")

        leaf_r2.append(r2)
        leaf_xranges.append(r)
    leaf_slopes = np.array(leaf_slopes)
    leaf_xranges = np.array(leaf_xranges)
    stop = time.time()
    print(f"collect_leaf_slopes {stop - start:.3f}s")
    return leaf_xranges, leaf_slopes, leaf_r2


def avg_values_at_x(uniq_x, leaf_ranges, leaf_values):
    """
    Value at max(x) is NaN since we have not data beyond that point.
    :param leaf_ranges:
    :param leaf_values:
    :return:
    """
    start = time.time()
    nx = len(uniq_x)
    nslopes = len(leaf_values)
    slopes = np.zeros(shape=(nx, nslopes))
    i = 0  # leaf index; we get a line for each leaf
    # collect the slope for each range (taken from a leaf) as collection of
    # flat lines across the same x range
    for r, slope in zip(leaf_ranges, leaf_values):
        s = np.full(nx, slope) # s has value scope at all locations (flat line)
        # now trim line so it's only valid in range r
        s[np.where(uniq_x < r[0])] = np.nan
        s[np.where(uniq_x >= r[1])] = np.nan # don't set slope on right edge
        slopes[:, i] = s
        i += 1
    # The value could be genuinely zero so we use nan not 0 for out-of-range
    # Now average horiz across the matrix, averaging within each range
    avg_value_at_x = np.nanmean(slopes, axis=1)
    stop = time.time()
    # print(f"avg_value_at_x {stop - start:.3f}s")
    return avg_value_at_x


def plot_stratpd(X, y, colname, targetname=None,
                 ax=None,
                 ntrees=1,
                 min_samples_leaf=10,
                 min_r2_hires=0.5,
                 min_samples_hires=15,
                 min_samples_leaf_hires=.20,
                 xrange=None,
                 yrange=None,
                 pdp_dot_size=5,
                 title=None,
                 nlines=None,
                 show_dx_line=False,
                 show_xlabel=True,
                 show_ylabel=True,
                 connect_pdp_dots=False,
                 show_importance=False,
                 imp_color='#fdae61',
                 supervised=True,
                 bootstrap=False,
                 max_features = 1.0,
                 alpha=.5
                 ):
    if ntrees==1:
        max_features = 1.0
        bootstrap = False

    if min_r2_hires>1.0:
        min_r2_hires = 1.0

    # print(f"Unique {colname} = {len(np.unique(X[colname]))}/{len(X)}")
    if supervised:
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
        rf.fit(X.drop(colname, axis=1), y)
    else:
        """
        Wow. Breiman's trick works in most cases. Falls apart on Boston housing MEDV target vs AGE
        """
        print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = False,
                                   max_features = 1.0,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    uniq_x = np.array(sorted(np.unique(X[colname])))
    # print(f"\nModel wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_xranges, leaf_slopes, leaf_r2 = \
        collect_leaf_slopes(rf, X, y, colname, hires_r2_threshold=min_r2_hires,
                            hires_n_threshold=min_samples_hires,
                            hires_min_samples_leaf=min_samples_leaf_hires)
    slope_at_x = avg_values_at_x(uniq_x, leaf_xranges, leaf_slopes)
    r2_at_x = avg_values_at_x(uniq_x, leaf_xranges, leaf_r2)
    # Drop any nan slopes; implies we have no reliable data for that range
    # Make sure to drop uniq_x values too :)
    notnan_idx = ~np.isnan(slope_at_x) # should be same for slope_at_x and r2_at_x
    slope_at_x = slope_at_x[notnan_idx]
    uniq_x = uniq_x[notnan_idx]
    r2_at_x = r2_at_x[notnan_idx]
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

    ax.scatter(uniq_x, curve, s=pdp_dot_size, alpha=1, c='black')

    if connect_pdp_dots:
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
        other.tick_params(axis='y', colors=imp_color)
        other.set_ylabel("Feature importance", fontdict={"color":imp_color})
        other.plot(uniq_x, r2_at_x, lw=1, c=imp_color)
        a,b = ax.get_xlim()
        other.plot(b - (b-a)*.03, np.mean(r2_at_x), marker='>', c=imp_color)
        # other.plot(mx - (mx-mnx)*.02, np.mean(r2_at_x), marker='>', c=imp_color)

    return uniq_x, curve, r2_at_x


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


def plot_catstratpd(X, y, colname, targetname,
                    cats=None,
                    ax=None,
                    sort='ascending',
                    ntrees=1,
                    min_samples_leaf=10,
                    alpha=.15,
                    yrange=None,
                    title=None,
                    bootstrap=False,
                    max_features=1.0,
                    supervised=True,
                    pdp_marker_width=.5,
                    pdp_color='black',
                    style:('strip','scatter')='strip',
                    show_xlabel=True,
                    show_ylabel=True,
                    show_xticks=True):
    # if min_samples_leaf is None:
    #     # rule of thumb: for binary, 2 samples / leaf seems good
    #     # but num cats + 3 seems better for non-binary
    #     min_samples_leaf = len(np.unique(X[colname]))
    #     if min_samples_leaf>2:
    #         min_samples_leaf += 3
    #
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
    else:
        print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = False,
                                   max_features = 1.0,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    # rf = RandomForestRegressor(n_estimators=ntrees, min_samples_leaf=min_samples_leaf, oob_score=True)
    rf.fit(X.drop(colname, axis=1), y)
    # print(f"Model wo {colname} OOB R^2 {rf.oob_score_:.5f}")
    leaf_histos = catwise_leaves(rf, X, y, colname)
    avg_per_cat = np.nanmean(leaf_histos, axis=1)

    if len(cats)>50:
        show_xticks = False # failsafe

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

    # if too many categories, can't do strip plot
    if style=='strip':
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
            ax.scatter(xlocs, sorted_histos.iloc[:,i] - min_value,
                       alpha=alpha, marker='o', s=5,
                       c='#9CD1E3')
            xloc += 1
        ax.scatter(xlocs, avg_per_cat[sort_indexes]-min_value, c=pdp_color, s=pdp_marker_width)

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
