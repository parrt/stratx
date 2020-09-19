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

from sklearn.utils import resample

import stratx.ice as ice
from stratx.partdep import partial_dependence, cat_partial_dependence

from timeit import default_timer as timer
from joblib import Parallel, delayed


def importances(X: pd.DataFrame,
                y: pd.Series,
                catcolnames=set(),
                sortby='Importance',  # sort by importance or impact
                n_trials: int = 1,
                min_slopes_per_x=5,   # ignore pdp y values derived from too few slopes (usually at edges); for smallerData sets, drop this to five or so
                min_samples_leaf=15,
                cat_min_samples_leaf=5,
                drop_high_stddev=2.0,
                bootstrap=True, # boostrap by default, don't subsample
                subsample_size=.75,
                normalize=True,  # make imp values 0..1
                n_trees=1,
                rf_bootstrap=False, max_features=1.0,
                pvalues=False,  # use to get p-values for each importance; it's number trials
                pvalues_n_trials=80,
                supervised=True,
                n_jobs=1,
                verbose=False) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    print(f"PARAMETERS:")
    print(f"\tn=|X|                {len(X)}")
    print(f"\tn_trials             {n_trials}")
    print(f"\tmin_samples_leaf     {min_samples_leaf}")
    print(f"\tcat_min_samples_leaf {cat_min_samples_leaf}")
    print(f"\tmin_slopes_per_x     {min_slopes_per_x}")
    print(f"\tbootstrap            {bootstrap}")
    print(f"\tn_trees              {n_trees}")

    X = compress_catcodes(X, catcolnames)

    n,p = X.shape
    impact_trials = np.zeros(shape=(p, n_trials))
    importance_trials = np.zeros(shape=(p, n_trials))
    # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        if n_trials==1: # don't shuffle if not bootstrapping
            idxs = range(n)
        else:
            if bootstrap:
                idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
            else: # subsample
                idxs = resample(range(n), n_samples=int(n*subsample_size), replace=False)
        X_, y_ = X.iloc[idxs], y.iloc[idxs]
        impacts, importances = importances_(X_, y_, catcolnames=catcolnames,
                                            normalize=normalize,
                                            supervised=supervised,
                                            n_jobs=n_jobs,
                                            n_trees=n_trees,
                                            min_samples_leaf=min_samples_leaf,
                                            cat_min_samples_leaf=cat_min_samples_leaf,
                                            min_slopes_per_x=min_slopes_per_x,
                                            rf_bootstrap=rf_bootstrap,
                                            max_features=max_features,
                                            verbose=verbose)
        impact_trials[:,i] = impacts
        importance_trials[:,i] = importances

    I = pd.DataFrame(data={'Feature': X.columns})
    I = I.set_index('Feature')
    I['Importance'] = np.mean(importance_trials, axis=1)
    I['Impact'] = np.mean(impact_trials, axis=1)
    I['Importance sigma'] = np.std(importance_trials, axis=1)
    I['Impact sigma'] = np.std(impact_trials, axis=1)

    I['Impact p-value'] = 0.0
    I['Importance p-value'] = 0.0
    if pvalues:
        impact_pvalues, importance_pvalues = \
            importances_pvalues(X, y, catcolnames,
                                baseline_impacts=I['Impact'].values,
                                baseline_importances=I['Importance'].values,
                                supervised=supervised,
                                normalize=normalize,
                                n_jobs=n_jobs,
                                n_trials=pvalues_n_trials,
                                min_slopes_per_x=min_slopes_per_x,
                                n_trees=n_trees,
                                min_samples_leaf=min_samples_leaf,
                                cat_min_samples_leaf=cat_min_samples_leaf,
                                rf_bootstrap=rf_bootstrap,
                                max_features=max_features)
        I['Impact p-value'] = importance_pvalues
        I['Importance p-value'] = importance_pvalues
        # I['Rank'] = I['Importance'] * (1.0 - importance_pvalues)

    if sortby:
        I = Isortby(I, sortby, drop_high_stddev)

    # I['stable'] = True
    # if n_trials>1 and drop_high_stddev > 0:
    #     # stable only those features whose impact/importance is >= about 2 sigma
    #     I['stable'] = I[sortby] >= drop_high_stddev * I[sortby+' sigma']
    #
    # if sortby:
    #     I = I.sort_values(['stable',sortby], ascending=False)
    #
    # I = I.drop('stable', axis=1)

    # Set reasonable column order
    I = I[['Importance', 'Importance sigma', 'Importance p-value',
           'Impact', 'Impact sigma', 'Impact p-value']]
    if n_trials==1:
        I = I.drop(['Importance sigma', 'Impact sigma'], axis=1)
    if not pvalues:
        I = I.drop(['Importance p-value', 'Impact p-value'], axis=1)
    return I


def Isortby(I, sortby, stddev_threshold=2.0, ascending=False):
    I = I.copy()
    I['stable'] = True
    if sortby+" sigma" in I.columns.values and stddev_threshold > 0:
        # keep only those features whose impact/importance is >= about 2 sigma
        # set max_stddev to 0 to disable this filtering
        # the bigger max_stddev, the more we filter out "iffy" features
        I['stable'] = I[sortby] > stddev_threshold * I[sortby + ' sigma']
        I = I.sort_values(['stable', sortby], ascending=ascending)
        I = I.drop('stable', axis=1)
    else:
        I = I.sort_values(sortby, ascending=ascending)

    return I


def importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                 normalize=True,
                 supervised=True,
                 n_jobs=1,
                 n_trees=1,
                 min_samples_leaf=10,
                 cat_min_samples_leaf=5,
                 min_slopes_per_x=5,
                 rf_bootstrap=False, max_features=1.0,
                 verbose=False) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    all_start = timer()

    if n_jobs>1 or n_jobs==-1:
        # Do n_jobs in parallel; in case it flips to shared mem, make it readonly
        impacts_importances = Parallel(verbose=0, n_jobs=n_jobs, mmap_mode='r') \
            (delayed(single_feature_importance)(X, y, colname,
                                                catcolnames=catcolnames,
                                                supervised=supervised,
                                                n_jobs=n_jobs,
                                                n_trees=n_trees,
                                                min_samples_leaf=min_samples_leaf,
                                                cat_min_samples_leaf=cat_min_samples_leaf,
                                                min_slopes_per_x=min_slopes_per_x,
                                                max_features=max_features,
                                                verbose=verbose) for colname in X.columns)
    else:
        impacts_importances = [single_feature_importance(X, y, colname,
                                                         catcolnames=catcolnames,
                                                         supervised=supervised,
                                                         n_jobs=n_jobs,
                                                         n_trees=n_trees,
                                                         min_samples_leaf=min_samples_leaf,
                                                         cat_min_samples_leaf=cat_min_samples_leaf,
                                                         min_slopes_per_x=min_slopes_per_x,
                                                         rf_bootstrap=rf_bootstrap,
                                                         max_features=max_features,
                                                         verbose=verbose) for colname in X.columns]

    impacts_importances = np.array(impacts_importances)
    impacts = impacts_importances[:,0]
    importances = impacts_importances[:,1]

    total_impact = np.sum(impacts)
    total_importance = np.sum(importances)

    if normalize:
        impacts /= total_impact
        importances /= total_importance

    all_stop = timer()
    print(f"Impact importance time {(all_stop-all_start):.0f}s")

    return impacts, importances


def single_feature_importance(X: pd.DataFrame, y: pd.Series,
                              colname,
                              catcolnames=set(),
                              supervised=True,
                              n_jobs=1,
                              n_trees=1,
                              min_samples_leaf=10,
                              cat_min_samples_leaf=5,
                              min_slopes_per_x=5,
                              rf_bootstrap=False, max_features=1.0,
                              verbose=False):
    "Return impact=unweighted avg abs, importance=weighted avg abs"
    X_col = X[colname].values.round(decimals=10)

    #print(f"Start {'catvar' if (colname in catcolnames) else 'numerical'} {colname}")
    if colname in catcolnames:
        leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored = \
            cat_partial_dependence(X, y, colname=colname,
                                   n_trees=n_trees,
                                   min_samples_leaf=cat_min_samples_leaf,
                                   rf_bootstrap=rf_bootstrap,
                                   max_features=max_features,
                                   verbose=verbose,
                                   supervised=supervised)
        impact, importance = cat_compute_importance(avg_per_cat, count_per_cat)
    else:
        leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
            partial_dependence(X=X, y=y, colname=colname,
                               n_trees=n_trees,
                               min_samples_leaf=min_samples_leaf,
                               min_slopes_per_x=min_slopes_per_x,
                               rf_bootstrap=rf_bootstrap,
                               max_features=max_features,
                               verbose=verbose,
                               parallel_jit=n_jobs == 1,
                               supervised=supervised)
        impact, importance = compute_importance(X_col, pdpx, pdpy)
    #print("IGNORED", ignored)
    #print(f"{colname}:{impact:.3f}, {importance:.3f} mass")
    # print(f"Stop {colname}")
    return impact, importance


def compute_importance(X_col, pdpx, pdpy):
    # Weight pdpy values by how many X[colname] values there are at the associated pdpx
    _, count_at_uniq_x = np.unique(X_col[np.isin(X_col, pdpx)], return_counts=True)
    if len(count_at_uniq_x) > 0:
        # weighted average of pdpy using count_at_uniq_x
        weighted_avg_abs_pdp = np.sum(np.abs(pdpy * count_at_uniq_x)) / np.sum(count_at_uniq_x)
    else:
        weighted_avg_abs_pdp = np.mean(np.abs(pdpy))

    # unweighted
    avg_abs_pdp = np.mean(np.abs(pdpy))
    return avg_abs_pdp, weighted_avg_abs_pdp


def cat_compute_importance(avg_per_cat, count_per_cat):
    # First mean-center avg_per_cat by mean of avg y per cat not overall y average.
    # All avg y per cat are just relative to each other; there is no "zero",
    # which we need in order to compute mean(abs(y)). Can't push min to 0 as then
    # all values are positive when some clearly pull y down. For example, if
    # avg_per_cat deltas are [0,1,1,1], then this gives impact=3/4 or, if we had
    # the equivalent [-1,0,0,0] then impact=-1/4. That's a huge difference. We
    # need to normalize and best thing is to see how each cat pushes y up or down
    # from average cat y.
    #
    # When min_samples_leaf is big enough to get all samples into a single leaf,
    # then the mean-centered avg_per_cat should look exactly like mean-centered
    # marginal plot. I verified and it works on bulldozer.  Seeing all cats
    # in one leaf means we know exact deltas between categories; more specifically
    # between avg y at each category. By mean-centering a marginal plot, we take
    # the same shape down from y-intercept and make it 0. Then they look the same.
    centered_avg_per_cat = avg_per_cat.copy() - np.nanmean(avg_per_cat)

    # weight each cat value by how many were used to create it
    abs_avg_per_cat = np.abs(centered_avg_per_cat)
    weighted_avg_abs_pdp = np.nansum(abs_avg_per_cat * count_per_cat) / np.sum(count_per_cat)

    # do unweighted
    # some cats have NaN, such as 0th which is often for "missing values"
    # depending on label encoding scheme.
    avg_abs_pdp = np.nanmean(abs_avg_per_cat)
    return avg_abs_pdp, weighted_avg_abs_pdp


'''
def all_pairs_delta(avg_per_cat):
    """
    For a vector containing the average delta per category found
    by merging deltas found across regions, compute the average drop/rise in y
    if we move from all other categories. If we are talking about US state temperatures,
    and leave on an airplane from a random state, what is expected bump or drop
    in temperature when arriving in, say, AZ?  That is the average change in y temp from
    all other states to AZ.

    Return array of deltas has same size as arg avg_per_cat.
    """
    avg_pairwise_delta = np.full_like(avg_per_cat, fill_value=np.nan, dtype=float)
    # first find catcodes (same as index into avg_per_cat). Find all non-NaN
    catcodes = np.where(~np.isnan(avg_per_cat))[0]
    ncodes = len(catcodes)  # how many codes have usable data?
    #     avg_pairwise_delta = []
    for i, k in enumerate(catcodes):
        # Shift cat k to 0 to find relative bump up/down to others;
        rel_to_k = avg_per_cat - avg_per_cat[k]
        # Since not all cats are present, some of these are still NaN;
        # don't count those nor cat k to itself
        avg_delta_away_from_k = np.nansum(rel_to_k) / (ncodes - 1)
        avg_delta_to_k = -avg_delta_away_from_k
        #         print("to code",k,rel_to_k, avg_delta_to_k)
        avg_pairwise_delta[k] = avg_delta_to_k
    return avg_pairwise_delta


def new_cat_compute_importance(avg_per_cat, count_per_cat):
    all_pairwise_deltas = all_pairs_deltas_foo(avg_per_cat)
    impact = np.nanmean(np.abs(all_pairwise_deltas))

    cat_density_weight = count_per_cat / np.max(count_per_cat)
    all_pairwise_deltas = all_pairs_deltas_foo(avg_per_cat * cat_density_weight)
    importance = np.nanmean(np.abs(all_pairwise_deltas))

    # # do unweighted
    # impact = avg_all_pairs_abs_delta(avg_per_cat)
    # prob_cat = count_per_cat / np.sum(count_per_cat)
    # importance = avg_all_pairs_abs_delta(avg_per_cat * prob_cat)
    #
    # # weight impact of each cat value by how many were used to create it
    # # Note: some cats have NaN, such as 0th which is often for "missing values"
    # # depending on label encoding scheme.
    # weighted_avg_abs_pdp = np.nansum(impact * count_per_cat) / np.sum(count_per_cat)

    return impact, importance

# def avg_all_pairs_abs_delta(avg_per_cat):
#     all_pairwise_deltas = all_pairs_deltas(avg_per_cat)
#     return np.mean(np.abs(all_pairwise_deltas))

def all_pairs_deltas_foo(avg_per_cat):
    """
    Impact for a categorical variable x_j is the average magnitude of change of
    PD_j from category level A to level B, for all possible pairs A,B. NaN entries
    indicate we have no data for that category and so these are ignored.

    Mathematically, we are filling an upper triangular matrix by putting the raw
    avg_per_cat as first row, then shifting that vector by k=1, 2, 3, ... to get
    the other rows of the matrix. Then, we take the nanmean of the upper triangular
    part. The lower triangular part is the negative of the upper and the diagonal is 0

    See test_all_pairs_delta.py for unit tests.
    """
    # first find catcodes (same as index into avg_per_cat). Find all non-NaN
    catcodes = np.where(~np.isnan(avg_per_cat))[0]
    all_pairwise_deltas = []
    for k in catcodes:
        # Shift cat k to 0 to find relative bump up/down to others;
        # Since not all cats are present, some of these are still NaN
        rel_to_k = avg_per_cat - avg_per_cat[k]
        all_pairwise_deltas.extend(rel_to_k[k+1:])
        # print(rel_to_k[k+1:])
    return all_pairwise_deltas
'''

def importances_pvalues(X: pd.DataFrame,
                        y: pd.Series,
                        catcolnames=set(),
                        baseline_impacts=None,
                        baseline_importances=None, # importances to use as baseline; must be in X column order!
                        supervised=True,
                        normalize=True,
                        n_jobs=1,
                        n_trials: int = 1,
                        min_slopes_per_x=5,
                        n_trees=1,
                        min_samples_leaf=10,
                        cat_min_samples_leaf=5,
                        rf_bootstrap=False,
                        max_features=1.0):
    """
    For each feature, compute and return empirical p-values.  The idea is to shuffle y
    and then compute feature importances; do this repeatedly to get a null distribution.
    The importances for feature j form a distribution and we can count how many times the
    importance value (obtained with shuffled y) reaches the importance value computed
    using unshuffled y.
    """
    if baseline_importances is None or baseline_impacts is None:
        baseline_impacts, baseline_importances = \
            importances_(X, y, catcolnames=catcolnames,
                         normalize=normalize,
                         supervised=supervised,
                         n_jobs=n_jobs,
                         n_trees=n_trees,
                         min_samples_leaf=min_samples_leaf,
                         cat_min_samples_leaf=cat_min_samples_leaf,
                         min_slopes_per_x=min_slopes_per_x,
                         rf_bootstrap=rf_bootstrap,
                         max_features=max_features)

    impact_counts = np.zeros(shape=X.shape[1])
    importance_counts = np.zeros(shape=X.shape[1])
    for i in range(n_trials):
        impacts, importances = importances_(X, y.sample(frac=1.0, replace=False),
                                            catcolnames=catcolnames,
                                            normalize=normalize,
                                            supervised=supervised,
                                            n_jobs=n_jobs,
                                            n_trees=n_trees,
                                            min_samples_leaf=min_samples_leaf,
                                            cat_min_samples_leaf=cat_min_samples_leaf,
                                            min_slopes_per_x=min_slopes_per_x,
                                            rf_bootstrap=rf_bootstrap,
                                            max_features=max_features)
        # print("Shuffled impacts\n",impacts)
        # print("Shuffled importances\n",importances)
        # print("Counts\n", impacts >= I_baseline['Impact'].values)
        impact_counts += impacts >= baseline_impacts
        importance_counts += importances >= baseline_importances

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC379178/ says don't use r/n
    # "Typically, the estimate of the P value is obtained as equation p_hat = r/n, where n
    # is the number of replicate samples that have been simulated and r is the number
    # of these replicates that produce a test statistic greater than or equal to that
    # calculated for the actual data. However, Davison and Hinkley (1997) give the
    # correct formula for obtaining an empirical P value as (r+1)/(n+1)."
    impact_pvalues = (impact_counts + 1) / (n_trials + 1)
    importance_pvalues = (importance_counts + 1) / (n_trials + 1)

    return impact_pvalues, importance_pvalues


def pdp_importances(model,X,numx=30,normalize=True):
    """
    Use standard PDP then mean center and take average magnitude as impact. Return
    an importance dataframe
    """
    pdpxs,pdpys = ice.friedman_partial_dependences(model, X, numx=numx, mean_centered=True)
    I = pd.DataFrame(data={'Feature': X.columns})
    I = I.set_index('Feature')
    Ivals = np.mean(np.abs(pdpys), axis=1)
    if normalize:
        total = np.sum(Ivals)
        Ivals /= total
    I['Importance'] = Ivals

    return I.sort_values('Importance', ascending=False)


def compress_catcodes(X, catcolnames, inplace=False):
    "Compress categorical integers if less than 90% dense"
    X_ = X if inplace else X.copy()
    for colname in catcolnames:
        uniq_x = np.unique(X_[colname])
        if len(uniq_x) < 0.90 * len(X_):  # sparse? compress into contiguous range of x cat codes
            X_[colname] = X_[colname].rank(method='min').astype(int)
    return X_