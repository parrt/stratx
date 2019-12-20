import numpy as np
import pandas as pd
from sklearn.utils import resample
from timeit import default_timer as timer

from stratx.partdep import *
from stratx.ice import *


def impact_importances(X: pd.DataFrame,
                       y: pd.Series,
                       catcolnames=set(),
                       n_samples=None,  # use all by default
                       min_slopes_percentile_x=0.003,
                       bootstrap_sampling=True,
                       n_trials: int = 1,
                       n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                       verbose=False,
                       pdp:('stratpd','ice')='stratpd') -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    if n_trees==1:
        bootstrap_sampling = False

    n,p = X.shape
    imps = np.zeros(shape=(p, n_trials)) # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        bootstrap_sample_idxs = resample(range(n), n_samples=n_samples, replace=bootstrap_sampling)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        imps[:,i] = impact_importances_(X_, y_, catcolnames=catcolnames,
                                        n_trees=n_trees,
                                        min_samples_leaf=min_samples_leaf,
                                        min_slopes_percentile_x=min_slopes_percentile_x,
                                        bootstrap=bootstrap,
                                        max_features=max_features,
                                        verbose=verbose,
                                        pdp=pdp)

    avg_imps = np.mean(imps, axis=1)
    stddev_imps = np.std(imps, axis=1)

    I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': avg_imps,
                           "Sigma":stddev_imps})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)

    return I


def impact_importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                        n_trees=1, min_samples_leaf=10,
                        min_slopes_percentile_x=0.003,
                        bootstrap=False, max_features=1.0,
                        verbose=False,
                        pdp:('stratpd','ice')='stratpd') -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    if pdp not in {'stratpd','ice'}:
        raise ValueError("pdp must be 'stratpd' or 'ice'")

    all_start = timer()
    p = X.shape[1]
    # avg_pdp = np.zeros(shape=(p,)) # track avg pdp, not magnitude
    avg_abs_pdp = np.zeros(shape=(p,)) # like area under PDP curve but not including width
    total_avg_pdpy = 0.0

    if pdp=='ice':
        rf = RandomForestRegressor(n_estimators=30)
        rf.fit(X, y)

    for j, colname in enumerate(X.columns):
        # Ask stratx package for the partial dependence of y with respect to X[colname]
        if colname in catcolnames:
            start = timer()
            if pdp=='stratpd':
                catcodes, _, catcode2name = getcats(X, colname, incoming_cats=None)
                leaf_histos, avg_per_cat, ignored = \
                    cat_partial_dependence(X, y, colname=colname,
                                           index=catcode2name,
                                           ntrees=n_trees,
                                           min_samples_leaf=min_samples_leaf,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           verbose=verbose)
                #         print(f"Ignored for {colname} = {ignored}")
            elif pdp=='ice':
                pdpy = original_catpdp(rf, X=X, colname=colname)
            stop = timer()
            # print(f"{colname} CatStratPD time for {len(X)} records = {(stop - start) * 1000:.0f}ms")
            # no need to shift as abs(avg_per_cat) deals with negatives. The avg per cat
            # values will straddle 0, some above, some below.
            # some cats have NaN, such as 0th which is for "missing values"
            avg_abs_pdp[j] = np.nanmean(np.abs(avg_per_cat))# * (ncats - 1)
            # avg_pdp[j] = np.mean(avg_per_cat)
            total_avg_pdpy += avg_abs_pdp[j]
        else:
            start = timer()
            if pdp=='stratpd':
                leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
                    partial_dependence(X=X, y=y, colname=colname,
                                       ntrees=n_trees,
                                       min_samples_leaf=min_samples_leaf,
                                       min_slopes_percentile_x=min_slopes_percentile_x,
                                       bootstrap=bootstrap,
                                       max_features=max_features,
                                       verbose=verbose)
                #         print(f"Ignored for {colname} = {ignored}")
            elif pdp=='ice':
                pdpy = original_pdp(rf, X=X, colname=colname)
            stop = timer()
            # print(f"{colname} StratPD time for {len(X)} records = {(stop-start)*1000:.0f}ms")
            avg_abs_pdp[j] = np.mean(np.abs(pdpy))# * (np.max(pdpx) - np.min(pdpx))
            # avg_pdp[j] = np.mean(pdpy)
            total_avg_pdpy += avg_abs_pdp[j]

    # print("avg_pdp", avg_pdp, "sum", np.sum(avg_pdp), "avg y", np.mean(y), "avg y-min(y)", np.mean(y)-np.min(y))
    normalized_importances = avg_abs_pdp / total_avg_pdpy

    all_stop = timer()
    print(f"Impact importance time {(all_stop-all_start):.0f}s")

    return normalized_importances
