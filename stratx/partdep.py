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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
from typing import Sequence

from numba import jit, prange


def leaf_samples(rf, X_not_col:np.ndarray) -> Sequence:
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
        sample_idxs_in_leaves = [np.where(leaf_ids[:, t] == id)[0] for id in uniq_ids]
        leaf_samples.extend(sample_idxs_in_leaves)
    return leaf_samples


def partial_dependence(X:pd.DataFrame, y:pd.Series, colname:str,
                       min_slopes_per_x=5,
                       parallel_jit=True,
                       n_trees=1, min_samples_leaf=15, rf_bootstrap=False, max_features=1.0,
                       supervised=True,
                       verbose=False):
    """
    Internal computation of partial dependence information about X[colname]'s effect on y.
    Also computes partial derivative of y with respect to X[colname].

    :param X: Dataframe with all explanatory variables
    :param y: Series or vector with response variable
    :param colname: which X[colname] (a string) to compute partial dependence for
    :param min_slopes_per_x: ignore pdp y values derived from too few slopes; this is
           same count across all features (tried percentage of max slope count but was
           too variable). Important for getting good starting point of PD.

    Returns:
        leaf_xranges    The ranges of X[colname] partitions


        leaf_slopes     Associated slope for each leaf xrange

        slope_counts_at_x How many slopes are available at each x_j location

        dx              The change in x from one non-NaN unique X[colname] to the next

        slope_at_x      The slope at each non-NaN unique X[colname]

        pdpx            The non-NaN unique X[colname] values; len(pdpx)<=len(unique(X[colname]))

        pdpy            The effect of each non-NaN unique X[colname] on y; effectively
                        the cumulative sum (integration from X[colname] x to z for all
                        z in X[colname]). The first value is always 0.

        ignored         How many samples from len(X) total records did we have to
                        ignore because samples in leaves had identical X[colname]
                        values.
  """
    X_not_col = X.drop(colname, axis=1).values
    # For x floating-point numbers that are very close, I noticed that np.unique(x)
    # was treating floating-point numbers different in the 12th decimal point as different.
    # This caused a number of problems likely but I didn't notice it until I tried
    # np.gradient(), which found extremely huge derivatives. I fixed that with a hack:
    X_col = X[colname].values.round(decimals=10)

    if supervised:
        rf = RandomForestRegressor(n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=rf_bootstrap,
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
                                    min_samples_leaf=min_samples_leaf,
                                    bootstrap=rf_bootstrap,
                                    max_features=max_features,
                                    oob_score=False)
        rf.fit(X_synth.drop(colname, axis=1), y_synth)

    if verbose:
        leaves = leaf_samples(rf, X_not_col)
        nnodes = rf.estimators_[0].tree_.node_count
        print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
              f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    leaf_xranges, leaf_slopes, ignored = \
        collect_discrete_slopes(rf, X_col, X_not_col, y) # if ignored, won't have entries in leaf_* results

    # print('leaf_xranges', leaf_xranges)
    # print('leaf_slopes', leaf_slopes)

    real_uniq_x = np.unique(X_col)   # comes back sorted
    if verbose:
        print(f"discrete StratPD num samples ignored {ignored}/{len(X)} for {colname}")

    #print("uniq x =", len(real_uniq_x), "slopes.shape =", leaf_slopes.shape, "x ranges.shape", leaf_xranges.shape)
    if parallel_jit:
        slope_at_x, slope_counts_at_x = \
            avg_slopes_at_x_jit(real_uniq_x, leaf_xranges, leaf_slopes)
    else:
        slope_at_x, slope_counts_at_x = \
            avg_slopes_at_x_nonparallel_jit(real_uniq_x, leaf_xranges, leaf_slopes)

    if min_slopes_per_x <= 0:
        min_slopes_per_x = 1 # must have at least one slope value

    # Turn any slopes with weak evidence into NaNs but keep same slope_at_x length
    slope_at_x = np.where(slope_counts_at_x >= min_slopes_per_x, slope_at_x, np.nan)

    pdpx = real_uniq_x

    # At this point, slope_at_x will have at least one nan, but that's okay because
    # diff and nancumsum will treat them appropriately.  nancumsum treats NaN as 0
    # so the previous non-NaN value is carried forward until we reach next real value.
    # Keep in mind that the last slope in slope_at_x is always nan,
    # since we have no information beyond that. However, we can still produce
    # a pdpy value for that last position because we have a delta from the
    # previous position that will get us to the last position.

    # Integrate the partial derivative estimate in slope_at_x across pdpx to get dependence
    dx = np.diff(pdpx)      # for n pdpx values, there are n-1 dx values
    dydx = slope_at_x[:-1]  # for n y values, only n-1 slope values; last slope always nan

    y_deltas = dydx * dx                          # get change in y from x[i] to x[i+1]
    pdpy = np.nancumsum(y_deltas)                 # y_deltas has one less value than num x values
    pdpy = np.concatenate([np.array([0]), pdpy])  # align with x values; our PDP y always starts from zero

    # At this point pdpx, pdpy have the same length as real_uniq_x

    # Strip from pdpx,pdpy any positions for which we don't have useful slope info. If we have
    # slopes = [1, 3, nan] then cumsum will give 3 valid pdpy values. But, if we have
    # slopes = [1, 3, nan, nan], then there is no pdpy value for last position. nancumsum
    # carries same y to the right, but we wanna strip last position, in this case.
    # Detect 2 nans in a row. We don't want the second or any after that until next real value
    idx_not_adjacent_nans = np.where(~(np.isnan(slope_at_x[1:]) & np.isnan(slope_at_x[:-1])))[0] + 1
    # Deal with first position. If it's nan, drop it, else keep it
    if not np.isnan(slope_at_x[0]):
        idx_not_adjacent_nans = np.concatenate([np.array([0]), idx_not_adjacent_nans])
    pdpx = pdpx[idx_not_adjacent_nans]
    pdpy = pdpy[idx_not_adjacent_nans]

    return leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored


def finite_differences(x: np.ndarray, y: np.ndarray):
    """
    Use the unique x values within a leaf to compute finite differences. Given, n unique
    x values return n-1 derivative estimates. We start by grouping the leaf x,y by x
    and then collect the average y. The unique x and y averages are the new x and y pairs.

    The slope for each x is the forward discrete difference:

        (y_{i+1} - y_i) / (x_{i+1} - x_i)

    At right edge, there is no following x value so we could use backward difference:

        (y_i - y_{i-1}) / (x_{i+1} - x_i)

    But we don't use that last value as there's nothing to plot pass last value; slope
    at last value therefore is worthless.

    I experimented with the center difference using a secant method:

        (y_{i+1} - y_{i-1}) / (x_{i+1} - x_{i-1})

    which is Method A from Veldman, A.E.P., Rinzema, K. Playing with nonuniform grids.
    J Eng Math 26, 119–130 (1992).  https://doi.org/10.1007/BF00043231
    https://www.rug.nl/research/portal/files/3332271/1992JEngMathVeldman.pdf

    The np.gradient() function uses Method B from Veldman and Rinzema but that one seems
    to give much worse derivatives for sparse data.  In the end, forward difference seemed
    to get the best approximation for synthetic data, particularly sparse data. Since
    it is simplest, I went with it.

    If the ordinal/ints are exactly one unit part, then it's just y_{i+1} - y_i. If
    they are not consecutive, we do not ignore isolated x_i as it ignores too much data.
    E.g., if x is [1,3,4] and y is [9,8,10] then the x=2 coordinate is spanned as part
    of 1 to 3. The two slopes are [(8-9)/(3-1), (10-8)/(4-3)] and bin widths are [2,1].

    If there is exactly one unique x value in the leaf, the leaf provides no information
    about how X[colname] contributes to changes in y. We have to ignore this leaf.
    """
    ignored = 0

    # Group by x, take mean of all y with same x value (they come back sorted too)
    uniq_x = np.unique(x)
    avg_y = np.array([y[x==ux].mean() for ux in uniq_x])

    if len(uniq_x)==1:
        # print(f"ignore {len(x)} in discrete_xc_space")
        ignored += len(x)
        return np.array([[0]],dtype=x.dtype), np.array([0.0]), ignored

    # FORWARD DIFF
    x_deltas = np.diff(uniq_x)
    y_deltas = np.diff(avg_y)
    leaf_slopes = y_deltas / x_deltas  # "rise over run"

    # AVERAGE AROUND CENTER DIFF
    # At position i, take average of forward slope from y[i-1] to y[i] and
    # slope from y[i] to y[i+1].
    # leaf_slopes = [leaf_slopes[0]] + list((leaf_slopes[:-1] + leaf_slopes[1:]) / 2)
    # leaf_slopes = np.array(leaf_slopes)
    # dang, doesn't seem to work

    # CENTER DIFF
    # x_deltas2 = uniq_x[2:] - uniq_x[:-2]  # this is empty if |uniq_x|==2
    # y_deltas2 = avg_y[2:] - avg_y[:-2]
    # dydx0 = (avg_y[1] - avg_y[0]) / (uniq_x[1] - uniq_x[0]) # forward diff for i=0
    # leaf_slopes_ctr = [dydx0] + list(y_deltas2 / x_deltas2)  # "rise over run, stride 2"
    # leaf_slopes_ctr = np.array(leaf_slopes_ctr)

    # leaf_slopes = np.gradient(avg_y, uniq_x)[:-1] # drop last derivative as we won't use it
    # print("ctr",list(leaf_slopes_ctr))
    # print("grd",list(leaf_slopes))

    leaf_xranges = np.array(list(zip(uniq_x, uniq_x[1:])))

    return leaf_xranges, leaf_slopes, ignored


def collect_discrete_slopes(rf, X_col, X_not_col, y):
    """
    For each leaf of each tree of the decision tree or RF rf (trained on all features
    except colname), get the leaf samples then isolate the X[colname] values
    and the target y values.  Compute the y deltas between unique X[colname] values.
    Like performing piecewise linear regression of X[colname] vs y
    to get the slopes in various regions of X[colname].  We don't need to subtract
    the minimum y value before regressing because the slope won't be different.
    (We are ignoring the intercept of the regression line).

    Return for each leaf, the ranges of X[colname] partitions,
    associated slope for each range, and number of ignored samples.
    """
    # start = timer()
    leaf_slopes = []   # drop or rise between discrete x values
    leaf_xranges = []  # drop is from one discrete value to next

    ignored = 0

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
            finite_differences(leaf_x, leaf_y)

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


# We get about 20% boost from parallel but limits use of other parallelism it seems;
# i get crashes when using multiprocessing package on top of this.
# If using n_jobs=1 all the time for importances, then turn jit=False so this
# method is not used
@jit(nopython=True, parallel=True) # use prange not range.
def avg_slopes_at_x_jit(uniq_x, leaf_ranges, leaf_slopes):
    """
    Compute the average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point and so there is
    no forward difference. If last range is 4..5 then slope at 5 is nan since we
    don't know where it's going to go from there.
    """
    nx = uniq_x.shape[0]
    nslopes = leaf_slopes.shape[0]
    slopes = np.empty(shape=(nx, nslopes), dtype=np.double)
    for i in prange(nslopes):
        xr, slope = leaf_ranges[i], leaf_slopes[i]
        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        slopes[:, i] = np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]), np.nan, slope)

    # Slope values could be genuinely zero so we use nan not 0 for out-of-range.

    # Now average horiz across the matrix, averaging within each range
    # It's possible that some some rows would be purely NaN, indicating there are no
    # slopes for that X[colname] value. This can happen when we ignore some leaves,
    # when they have a single unique X[colname] value.

    # Compute:
    #   avg_slope_at_x = np.mean(slopes[good], axis=1)  (numba doesn't allow axis arg)
    #   slope_counts_at_x = nslopes - np.isnan(slopes).sum(axis=1)
    avg_slope_at_x = np.zeros(shape=nx)
    slope_counts_at_x = np.zeros(shape=nx)
    for i in prange(nx):
        row = slopes[i, :]
        n_nan = np.sum(np.isnan(row))
        avg_slope_at_x[i] = np.nan if n_nan==nslopes else np.nanmean(row)
        slope_counts_at_x[i] = nslopes - n_nan

    # return average slope at each unique x value and how many slopes included in avg at each x
    return avg_slope_at_x, slope_counts_at_x


# Hideous copying of avg_values_at_x_jit() to get different kinds of jit'ing. This is slower by 20%
# than other version but can run in parallel with multiprocessing package.
@jit(nopython=True)
def avg_slopes_at_x_nonparallel_jit(uniq_x, leaf_ranges, leaf_slopes):
    """
    Compute the weighted average of leaf_slopes at each uniq_x.

    Value at max(x) is NaN since we have no data beyond that point and so there is
    no forward difference.
    """
    nx = len(uniq_x)
    nslopes = len(leaf_slopes)
    slopes = np.zeros(shape=(nx, nslopes))
    for i in range(nslopes):
        xr, slope = leaf_ranges[i], leaf_slopes[i]
        # Compute slope all the way across uniq_x but then trim line so
        # slope is only valid in range xr; don't set slope on right edge
        slopes[:, i] = np.where( (uniq_x < xr[0]) | (uniq_x >= xr[1]), np.nan, slope)

    # Slope values could be genuinely zero so we use nan not 0 for out-of-range.

    # Now average horiz across the matrix, averaging within each range
    # It's possible that some some rows would be purely NaN, indicating there are no
    # slopes for that X[colname] value. This can happen when we ignore some leaves,
    # when they have a single unique X[colname] value.

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


def catwise_leaves(rf, X_not_col, X_col, y, max_catcode):
    """
    Return a 2D array with the average y value for each category in each leaf.
    Choose the cat code of smallest avg y as the reference category. I used to think it
    was arbitrary but now I realize that we have to mimic what StratPD does for
    numerical variables. By integrating the y deltas, we get back exactly the y
    values minus the minimum y. E.g., for x=[1,2,3] and y=[7,10,15], StratPD gets
    dx = [1,1] and dy=[3,5] then cumsum of dy gets [3,8] and we stick 0 on front:
    pdpy = [0,3,8].  To mimic the same thing with CatStratPD, we simply subtract the
    minimum for all categories in this leaf. If cats=[1,2,3] and y=[7,10,15], then
    we subtract 7 from y = [0,3,8] and we arrive at the same answer. For situations
    where the numerical var PD never skips over a valid x (extrapolating slope), it
    should give the same answer as categorical var PD. Subtracting the min y also
    has the benefit that debugging is easier because, within a leaf, all deltas are
    positive. During merging, we can still get negative deltas as we lineup cats/deltas,
    but those are true values. We should not shift everything to be positive for
    the final merged pdpy.

    Normalize the y values into deltas by subtracting the avg y value for the
    reference category from the avg y for all categories. The reference category is
    the category of the minimum avg y.

    The columns are the y avg value changes per cat found in a single leaf as
    they differ from the reference cat y average. Each row represents a category level. E.g.,

    row
    cat           leaf0       leaf1
     0       166.430176  186.796956
     1       219.590349  176.448626

    Cats are possibly noncontiguous with nan rows for cat codes not present. Not all
    values in a leaf column will be non-nan.  Only those categories mentioned in
    a leaf have values.  Shape is (max cat + 1, num leaves).

    Within a single leaf, there will typically only be a few categories represented.
    """
    leaves = leaf_samples(rf, X_not_col)

    leaf_deltas = np.full(shape=(max_catcode+1, len(leaves)), fill_value=np.nan)
    leaf_counts = np.zeros(shape=(max_catcode+1, len(leaves)), dtype=int)
    keep_leaf_idxs = np.full(shape=(len(leaves),), fill_value=True, dtype=bool)

    ignored = 0
    for leaf_i in range(len(leaves)):
        sample = leaves[leaf_i]
        leaf_cats = X_col[sample]
        leaf_y = y[sample]
        # perform a groupby(catname).mean()
        uniq_leaf_cats, count_leaf_cats = np.unique(leaf_cats, return_counts=True) # comes back sorted
        avg_y_per_cat = np.array([leaf_y[leaf_cats==cat].mean() for cat in uniq_leaf_cats])
        # print("uniq_leaf_cats",uniq_leaf_cats,"count_y_per_cat",count_leaf_cats)

        # if len(uniq_leaf_cats)==1 then we have single cat avg y and its delta is 0
        # but keep it as we'll treat as isolated group later and add marginal y for
        # this cat to get partial dependence value.

        # Can use any cat code as refcat; same "shape" of delta vec regardless of which we
        # pick. The vector is shifted/up or down but cat y's all still have the same relative
        # delta y. Might as well just pick the cat of smallest avg y.
        # Previously, I picked a random# reference category but that is unnecessary.
        # We will shift this vector during the merge operation so which we pick
        # here doesn't matter.
        idx_of_ref_cat_in_leaf = np.nanargmin(avg_y_per_cat)
        delta_y_per_cat = avg_y_per_cat - avg_y_per_cat[idx_of_ref_cat_in_leaf]
        # print("delta_y_per_cat",delta_y_per_cat)

        # Store into leaf i vector just those deltas we have data for
        # leave cats w/o representation as nan (uses index to figure out which rows to alter)
        leaf_deltas[uniq_leaf_cats, leaf_i] = delta_y_per_cat
        leaf_counts[uniq_leaf_cats, leaf_i] = count_leaf_cats

    # See unit test test_catwise_leaves:test_two_leaves_with_2nd_ignored()
    leaf_deltas = leaf_deltas[:,keep_leaf_idxs]
    leaf_counts = leaf_counts[:,keep_leaf_idxs]
    return leaf_deltas, leaf_counts, ignored


def cat_partial_dependence(X, y,
                           colname,  # X[colname] expected to be numeric codes
                           max_catcode=None,  # if we're bootstrapping, might see diff max's so normalize to one max
                           n_trees=1,
                           min_samples_leaf=5,
                           max_features=1.0,
                           rf_bootstrap=False,
                           supervised=True,
                           verbose=False):
    X_not_col = X.drop(colname, axis=1).values
    X_col = X[colname].values
    if (X_col<0).any():
        raise ValueError(f"Category codes must be > 0 in column {colname}")
    if not np.issubdtype(X_col.dtype, np.integer):
        raise ValueError(f"Category codes must be integers in column {colname} but is {X_col.dtype}")
    if max_catcode is None:
        max_catcode = np.max(X_col)
    if supervised:
        rf = RandomForestRegressor(n_estimators=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap = rf_bootstrap,
                                   max_features = max_features,
                                   oob_score=False)
    else:
        print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestClassifier(n_estimators=n_trees,
                                    min_samples_leaf=min_samples_leaf,  # * 2, # there are 2x as many samples (X,X') so must double leaf size
                                    bootstrap=rf_bootstrap,
                                    max_features=max_features,
                                    oob_score=False)
        rf.fit(X_synth.drop(colname,axis=1), y_synth)

    rf.fit(X_not_col, y)
    if verbose and supervised:
        print(f"CatStrat Partition RF: dropping {colname} training R^2 {rf.score(X_not_col, y):.2f}")

    leaf_deltas, leaf_counts, ignored = \
        catwise_leaves(rf, X_not_col, X_col, y.values, max_catcode)

    uniq_x = np.unique(X_col)
    # Ignoring other vars, what is average y for all records with same catcode?
    # We need if avg_values_at_cat finds disjoint sets
    marginal_avg_y_per_cat = np.full(shape=(max_catcode+1,), fill_value=np.nan)
    for cat in uniq_x:
        marginal_avg_y_per_cat[cat] = y[X_col == cat].mean()

    avg_per_cat, count_per_cat = \
        avg_values_at_cat(leaf_deltas, leaf_counts, marginal_avg_y_per_cat, verbose=verbose)

    if verbose:
        print(f"CatStratPD Num samples ignored {ignored} for {colname}")

    return leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored


def avg_values_at_cat(leaf_deltas, leaf_counts,
                      marginal_avg_y_per_cat=None, # only used if disjoint cat sets exist
                      max_iter=3, verbose=False):
    """
    Merge the deltas for categories from all of the leaves. If there
    is at least one category in common, then to groups of deltas can
    be merged. It's possible, however, that there are multiple
    disjoint groups of categories with no cats in common. Previously I
    would just ignore those, but now I realize there could be
    important groups of categories that don't get included. For
    example with the bulldozer data set it was common to see the vast
    majority of categories being ignored. Obviously this dramatically
    underestimates the impact of a categorical variable.

    We don't have categories in common between disjoint groups but we
    can still estimate their relative position, otherwise we'd be
    forced to start all disjoint groups at zero. If we treat each
    group as a meta-category, then we can compare the average y for
    categories in a group. For meta-cats A, B, C with avg y =
    [100,130,40] then the meta-deltas are [0,30,-60].  The base for
    the zeros in A, B, C will be 0, 30, -60 so we just add that to all
    cat deltas in A, B, C.  Note: the y used for mean is not the leaf
    deltas but y from training set.


    This is effectively the marginal delta between those meta-groups,
    but the fact that they are disjoint means that there were no
    situations where, all else being equal, the associated leaves had
    categories in common. While the overall variable might be highly
    codependent, such as ModelID (with YearMade, MachineHours,...) on
    bulldozer price, the individual groups WITHIN a variable could be
    totally independent. Imagine typical construction bulldozers
    versus the giant earthmovers used in mining.  It's extremely
    unlikely that stratification would yield groups containing both
    construction bulldozers and mining bulldozers. That hints that
    the average difference in price between the two groups would not be
    biased (too much). Besides, it's the best we can do given we
    literally have no information about the relative cost of a mining
    bulldozer and a construction bulldozer. All we can do is compare
    the average price of a bulldozer in both categories, and use that
    to shift the two meta-cats relative to each other.

    Rather than a complicated data structure and algorithm to find all
    disjoint subsets of intersecting categories, it's easier to use the
    existing algorithm I had already that merged leaves with categories in
    common.  After merging all possible, there remains a set of leaf
    indexes indicating what leaves remain to process. So, just call function
    avg_values_at_cat_one_disjoint_region() repeatedly until the work
    set comes back as empty. That indicates all leaves have been
    integrated. The count per category vectors can simply be added
    together to get the final count per category. The catavg is the
    running vector average computed in each pass over the leaves. For
    that, we can add them vectors together to get a single vector
    (like a set union), but we still need to keep track of the different
    disjoint category groups. We can do this by collecting a list
    of cat sets.  Then we compute the average y for those categories
    for each group. Then update the catavg vector values for each
    group.

    See comment for avg_values_at_cat_one_disjoint_region()
    """
    # catavg is the running average vector and starts out as the first column (1st leaf's deltas)
    catavg = np.full(shape=(leaf_deltas.shape[0],), fill_value=np.nan, dtype=np.float)
    catgroups = []
    count_per_cat = np.zeros(shape=(leaf_deltas.shape[0],), dtype=np.int)
    work = range(0, leaf_deltas.shape[1])  # all leaf indexes added to work list
    # print("START work", work)
    while len(work)>0:
        catavg_, count_per_cat_, work = \
            avg_values_at_cat_one_disjoint_region(work, leaf_counts, leaf_deltas, max_iter, verbose)
        # print("catavg", catavg_)
        # print("count_per_cat", count_per_cat_)
        # print("remaining work",work)
        catgroup = np.where(count_per_cat_>0)[0]
        # print("catgroup", catgroup)
        catgroups.append(catgroup)
        catavg[catgroup] = catavg_[catgroup] # merge disjoint avgs for catgroup into running vector
        count_per_cat += count_per_cat_

    if len(catgroups)>1: # more than one disjoint set of categories?
        avgs = []
        for cats in catgroups:
            m = np.mean(marginal_avg_y_per_cat[cats])
            # print("cats",cats,"mean",m)
            avgs.append(m)
        # Do what StratPD does: all values are relative to left edge
        relative_group_y = np.array(avgs) - avgs[0]
        # print("group avgs", avgs)
        # print("relative_group_y", relative_group_y)
        for cats,group_deltay in zip(catgroups, relative_group_y):
            catavg[cats] += group_deltay
        # print("adjusted catavg", catavg)

    # if verbose: print("final cat avgs", parray3(catavg))
    return catavg, count_per_cat


def avg_values_at_cat_one_disjoint_region(work, leaf_counts, leaf_deltas, max_iter, verbose):
    """
    In leaf_deltas, we have information from the leaves indicating how
    much above or below each category was from the reference category
    of that leaf.  The reference category was arbitrarily selected but
    now I realize it must be the cat code associated with the minimum
    avg cat y.  All deltas within a leaf are therefore positive values
    and the relative y for the reference category is 0.  Categories
    not mentioned in the leaf, will have NAN values.

    The goal is to merge all of the columns in leaf_deltas, despite
    the fact that they do not have the same reference category. We
    init a running average vector to be the first column of category
    deltas. Then we attempt to merge each of the other columns into
    the running average. We make multiple passes over the columns of
    leaf_deltas until nothing changes, we hit the maximum number of
    iterations, or everything has merged.

    To merge vector v (column j of leaf_deltas) into catavg, select a
    category, index ix, in common at random.  Subtract v[ix] from v so
    that ix is v's new reference and v[ix]=0. Add catavg[ix] to the
    adjusted v so that v is now comparable to catavg. We can now do a
    weighted average of catavg and v, paying careful attention of NaN.

    Previously, I was picking a random category for merging in an
    effort to reduce the effect of outliers, with the assumption that
    outliers were rare. Given 5 categories in common between the
    running average vector and a new vector, randomly picking one
    means a 1/5 chance of picking the outlier.  Outliers as reference
    categories shift the outlierness to all other categories. Boooo

    Now, I select the category in common that has the most evidence:
    the category associated with the most number of observations.

    It's possible that more than a single value within a leaf_deltas
    vector is 0.  I.e., the reference category value is always 0 in
    the vector, but there might be another category whose value was
    the same y, giving a 0 relative value.

    Example:

    leaf_deltas

    [[ 0. nan nan]
     [ 1. nan nan]
     [nan  0. nan]
     [nan  3.  0.]
     [nan  3.  2.]
     [ 4. nan nan]
     [ 5. nan nan]]

    leaf_counts

    [[1 0 0]
     [1 0 0]
     [0 1 0]
     [0 1 1]
     [0 1 1]
     [1 0 0]
     [1 0 0]]
    """
    initial_leaf_idx = work[0]
    work = set(work[1:])
    catavg = leaf_deltas[:, initial_leaf_idx]  # init with first ref category (column)
    catavg_weight = leaf_counts[:, initial_leaf_idx]
    completed = {-1}  # init to any nonempty set to enter loop
    iteration = 1
    # Three passes should be sufficient to merge all possible vectors, but
    # I'm being paranoid here and allowing it to run until completion or some maximum iterations
    while len(work) > 0 and len(completed) > 0 and iteration <= max_iter:
        # print(f"PASS {iteration} len(work)", len(work))
        completed = set()
        for j in work:  # for remaining leaf index in work list, avg in the vectors
            v = leaf_deltas[:, j]
            are_intersecting = ~np.isnan(catavg) & ~np.isnan(v)
            intersection_idx = np.where(are_intersecting)[0]

            # print(intersection_idx)
            if len(intersection_idx) == 0:  # found something to merge into catavg?
                continue

            # Merge column j into catavg vector
            # cat for merging is the one with most supporting evidence
            cur_weight = leaf_counts[:, j]

            PICK_MOST_CONF = False
            if PICK_MOST_CONF:
                ix = np.argmax(np.where((cur_weight > 0) & are_intersecting, cur_weight, 0))
                shifted_v = v - v[ix]  # make ix the reference cat in common
                relative_to_value = catavg[ix]  # corresponding value in catavg
                adjusted_v = shifted_v + relative_to_value  # adjust so v is mergeable with catavg
            else:
                adjusted_v = compute_avg_merge_candidate(catavg, v, intersection_idx)

            prev_catavg = catavg  # track only for verbose/debugging purposes
            catavg = nanavg_vectors(catavg, adjusted_v, catavg_weight, cur_weight)
            # Update weight of running avg to incorporate "mass" from v
            catavg_weight += cur_weight
            if verbose:
                # TODO: broken
                print(f"{ix:-2d} : vec to add =", parray(v), f"- {v[ix]:.2f}")
                print("     shifted    =", parray(shifted_v),
                      f"+ {relative_to_value:.2f}")
                print("     adjusted   =", parray(adjusted_v), "*", cur_weight)
                print("     prev avg   =", parray(prev_catavg), "*",
                      catavg_weight - cur_weight)
                print("     new avg    =", parray(catavg))
                print()
            completed.add(j)
        iteration += 1
        work = work - completed
    return catavg, catavg_weight, list(work)


def compute_avg_merge_candidate(catavg, v, intersection_idx):
    """
    Given intersecting deltas in catavg and v, compute average delta
    one could merge into running average.  If one cat is an outlier,
    picking that really distorts the vector we merge into running
    average vector. So, effectively merge using all as the ref
    cat in common by merging in average of all possible refcats.

    When there is no noise in y, the average merge candidate is
    the same as any single candidate. So, with no noise, we get
    exact answer; averaging here doesn't cost us anything. It
    only helps to spread noise across categories.
    """
    merge_candidates = []
    for i in intersection_idx:
        merge_candidates.append(v - v[i] + catavg[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # We get "Mean of empty slice" when all entries are Nan but we want that.
        v = np.nanmean(merge_candidates, axis=0)
    return v



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
        # X_rand[colname] = np.random.choice(X[colname], len(X), replace=True)
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


def nanavg_vectors(a, b, wa=1.0, wb=1.0):
    """
    Add two vectors a+b but support nan+x==x and nan+nan=nan
    np.nanmean works to get nan+nan=nan, but for weighted avg
    we need to divide by wa+wb after using nansum. nansum gives
    0 not nan it seems when adding nan+nan. Do it the hard way.
    """
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    c = a*wa + b*wb               # weighted average where both are non-nan
    c /= zero_as_one(wa+wb)       # weighted avg
    c[a_nan] = b[a_nan]           # copy any stuff where b has only value (unweighted into result)
    in_a_not_b = (~a_nan) & b_nan
    c[in_a_not_b] = a[in_a_not_b] # copy stuff where a has only value
    return c


def nanmerge_matrix_cols(A):
    """
    Add all vertical vectors in A but support nan+x==x and nan+nan=nan.
    """
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
