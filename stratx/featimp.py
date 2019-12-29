import numpy as np
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from joblib import parallel_backend, Parallel, delayed
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter
from os import getpid, makedirs
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from stratx.partdep import *
from stratx.ice import *


def impact_importances(X: pd.DataFrame,
                       y: pd.Series,
                       catcolnames=set(),
                       normalize=True,  # make imp values 0..1
                       supervised=True,
                       n_jobs=1,
                       sort=True,  # sort by importance in descending order?
                       min_slopes_per_x=15,
                       stddev=False,  # turn on to get stddev of importances via bootstrapping
                       n_stddev_trials: int = 5,
                       pvalues=False,  # use to get p-values for each importance; it's number trials
                       n_pvalue_trials=50,  # how many trials to do to get p-values
                       n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                       verbose=False,
                       pdp:('stratpd','ice')='stratpd') -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    resample_with_replacement = n_trees>1
    if not stddev:
        n_stddev_trials = 1
    n,p = X.shape
    imps = np.zeros(shape=(p, n_stddev_trials)) # track p var importances for ntrials; cols are trials
    for i in range(n_stddev_trials):
        bootstrap_sample_idxs = resample(range(n), n_samples=n, replace=resample_with_replacement)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        imps[:,i] = impact_importances_(X_, y_, catcolnames=catcolnames,
                                        normalize=normalize,
                                        supervised=supervised,
                                        n_jobs=n_jobs,
                                        n_trees=n_trees,
                                        min_samples_leaf=min_samples_leaf,
                                        min_slopes_per_x=min_slopes_per_x,
                                        bootstrap=bootstrap,
                                        max_features=max_features,
                                        verbose=verbose,
                                        pdp=pdp)

    avg_imps = np.mean(imps, axis=1)

    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': avg_imps})
    I = I.set_index('Feature')

    if stddev:
        I['Sigma'] = np.std(imps, axis=1)

    if pvalues:
        I['p-value'] = importances_pvalues(X, y, catcolnames,
                                           importances=I,
                                           supervised=supervised,
                                           n_jobs=n_jobs,
                                           n_trials=n_pvalue_trials,
                                           min_slopes_per_x=min_slopes_per_x,
                                           n_trees=n_trees,
                                           min_samples_leaf=min_samples_leaf,
                                           bootstrap=bootstrap,
                                           max_features=max_features)
    if sort is not None:
        I = I.sort_values('Importance', ascending=False)

    return I



def impact_importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                        normalize=True,
                        supervised=True,
                        n_jobs=1,
                        n_trees=1, min_samples_leaf=10,
                        min_slopes_per_x=15,
                        bootstrap=False, max_features=1.0,
                        verbose=False,
                        pdp:('stratpd','ice')='stratpd') -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    if pdp not in {'stratpd','ice'}:
        raise ValueError("pdp must be 'stratpd' or 'ice'")

    all_start = timer()

    if pdp=='ice':
        rf = RandomForestRegressor(n_estimators=30)
        rf.fit(X, y)

    def single_feature_importance(colname):
        # print(f"Start {colname}")
        if colname in catcolnames:
            if pdp=='stratpd':
                leaf_histos, avg_per_cat, ignored = \
                    cat_partial_dependence(X, y, colname=colname,
                                           n_trees=n_trees,
                                           min_samples_leaf=min_samples_leaf,
                                           bootstrap=bootstrap,
                                           max_features=max_features,
                                           verbose=verbose,
                                           supervised=supervised)
                #         print(f"Ignored for {colname} = {ignored}")
            elif pdp == 'ice':
                pdpy = original_catpdp(rf, X=X, colname=colname)
            # no need to shift as abs(avg_per_cat) deals with negatives. The avg per cat
            # values will straddle 0, some above, some below.
            # some cats have NaN, such as 0th which is for "missing values"
            avg_abs_pdp = np.nanmean(np.abs(avg_per_cat))# * (ncats - 1)
        else:
            if pdp=='stratpd':
                leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
                    partial_dependence(X=X, y=y, colname=colname,
                                       n_trees=n_trees,
                                       min_samples_leaf=min_samples_leaf,
                                       min_slopes_per_x=min_slopes_per_x,
                                       bootstrap=bootstrap,
                                       max_features=max_features,
                                       verbose=verbose,
                                       parallel_jit=n_jobs == 1,
                                       supervised=supervised)
                #         print(f"Ignored for {colname} = {ignored}")
            elif pdp=='ice':
                pdpy = original_pdp(rf, X=X, colname=colname)
            avg_abs_pdp = np.mean(np.abs(pdpy))# * (np.max(pdpx) - np.min(pdpx))
        # print(f"Stop {colname}")
        return avg_abs_pdp

    if n_jobs>1 or n_jobs==-1:
        # Do n_jobs in parallel; in case it flips to shared mem, make it readonly
        avg_abs_pdp = Parallel(verbose=0, n_jobs=n_jobs, mmap_mode='r')\
            (delayed(single_feature_importance)(colname) for colname in X.columns)
    else:
        avg_abs_pdp = [single_feature_importance(colname) for colname in X.columns]

    avg_abs_pdp = np.array(avg_abs_pdp)
    total_avg_pdpy = np.sum(avg_abs_pdp)

    normalized_importances = avg_abs_pdp
    if normalize:
        normalized_importances = avg_abs_pdp / total_avg_pdpy

    all_stop = timer()
    print(f"Impact importance time {(all_stop-all_start):.0f}s")

    return normalized_importances


def importances_pvalues(X: pd.DataFrame,
                        y: pd.Series,
                        catcolnames=set(),
                        importances=None, # importances to use as baseline; must be in X column order!
                        supervised=True,
                        n_jobs=1,
                        n_trials: int = 1,
                        min_slopes_per_x=15,
                        n_trees=1, min_samples_leaf=10, bootstrap=False,
                        max_features=1.0):
    """
    For each feature, compute and return empirical p-values.  The idea is to shuffle y
    and then compute feature importances; do this repeatedly to get a null distribution.
    The importances for feature j form a distribution and we can count how many times the
    importance value (obtained with shuffled y) reaches the importance value computed
    using unshuffled y.
    """
    I_baseline = importances
    if importances is None:
        I_baseline = impact_importances(X, y, catcolnames=catcolnames, sort=False,
                                        min_slopes_per_x=min_slopes_per_x,
                                        supervised=supervised,
                                        n_jobs=n_jobs,
                                        n_trees=n_trees,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bootstrap,
                                        max_features=max_features)

    counts = np.zeros(shape=X.shape[1])
    for i in range(n_trials):
        I = impact_importances(X, y.sample(frac=1.0, replace=False),
                               catcolnames=catcolnames, sort=False,
                               min_slopes_per_x=min_slopes_per_x,
                               supervised=supervised,
                               n_jobs=n_jobs,
                               n_trees=n_trees,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               max_features=max_features)
        counts += I['Importance'].values >= I_baseline['Importance'].values

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC379178/ says don't use r/n
    # "Typically, the estimate of the P value is obtained as equation p_hat = r/n, where n
    # is the number of replicate samples that have been simulated and r is the number
    # of these replicates that produce a test statistic greater than or equal to that
    # calculated for the actual data. However, Davison and Hinkley (1997) give the
    # correct formula for obtaining an empirical P value as (r+1)/(n+1)."
    pvalue = (counts + 1) / (n_trials + 1)

    # print(counts)
    # print(pvalue)
    return pvalue


class ImpViz:
    """
    For use with jupyter notebooks, plot_importances returns an instance
    of this class so we display SVG not PNG.
    """
    def __init__(self):
        tmp = tempfile.gettempdir()
        self.svgfilename = tmp+"/PimpViz_"+str(getpid())+".svg"
        plt.tight_layout()
        plt.savefig(self.svgfilename, bbox_inches='tight', pad_inches=0)

    def _repr_svg_(self):
        with open(self.svgfilename, "r", encoding='UTF-8') as f:
            svg = f.read()
        plt.close()
        return svg

    def save(self, filename):
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def view(self):
        plt.show()

    def close(self):
        plt.close()


def plot_importances(df_importances,
                     yrot=0,
                     title_fontsize=11,
                     label_fontsize=10,
                     fontname="Arial",
                     figsize=None,
                     width:float=3, # if no figsize, use this width
                     bar_width=13,
                     imp_range=(0, 1.0),
                     dpi=150,
                     color='#4574B4',#'#D9E6F5',
                     bgcolor=None,  # seaborn uses '#F1F8FE'
                     xtick_precision=2,
                     title=None,
                     ax=None):
    """
    Given an array or data frame of importances, plot a horizontal bar chart
    showing the importance values.

    :param df_importances: A data frame with Feature, Importance columns
    :type df_importances: pd.DataFrame
    :param width: Figure width in default units (inches I think). Height determined
                  by number of features.
    :type width: int
    :param minheight: Minimum plot height in default matplotlib units (inches?)
    :type minheight: float
    :param vscale: Scale vertical plot (default .25) to make it taller
    :type vscale: float
    :param label_fontsize: Font size for feature names and importance values
    :type label_fontsize: int
    :param yrot: Degrees to rotate feature (Y axis) labels
    :type yrot: int
    :param label_fontsize:  The font size for the column names and x ticks
    :type label_fontsize:  int
    :param scalefig: Scale width and height of image (widthscale,heightscale)
    :type scalefig: 2-tuple of floats
    :param xtick_precision: How many digits after decimal for importance values.
    :type xtick_precision: int
    :param xtick_precision: Title of plot; set to None to avoid.
    :type xtick_precision: string
    :param ax: Matplotlib "axis" to plot into
    :return: None

    SAMPLE CODE

    from stratx.featimp import *
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = importances(rf, X_test, y_test)
    viz = plot_importances(imp)
    viz.save('file.svg')
    viz.save('file.pdf')
    viz.view() # or just viz in notebook
    """
    GREY = '#444443'
    I = df_importances
    n_features = len(I)
    left_padding = 0.01

    ppi = 72 # matplotlib has this hardcoded. E.g., see https://github.com/matplotlib/matplotlib/blob/40dfc353aa66b93fd0fbc55ca1f51701202c0549/lib/matplotlib/axes/_base.py#L694
    imp = I.Importance.values

    barcounts = np.array([f.count('\n')+1 for f in I.index])
    N = np.sum(barcounts)

    ypositions = np.array(range(n_features))

    if ax is None:
        if figsize:
            fig, ax = plt.subplots(1, 1, figsize=figsize)#, dpi=dpi)
        else:
            # plt tries to make Bbox(x0=0.125, y0=0.10999999999999999, x1=0.9, y1=0.88) for canvas
            # Those are 0..1 values for part of the overall fig I think
            height_in_pixels = N * bar_width + 2 * bar_width/2 + (N-1) * 3
            # to compute figsize, add 1-.88 and .11 = .12+.11 = .23
            fudge = 15
            fig, ax = plt.subplots(1, 1, figsize=(width, (height_in_pixels + fudge) / ppi), dpi=dpi)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    # ax.spines['bottom'].set_smart_bounds(True)
    if bgcolor:
        ax.set_facecolor(bgcolor)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname, color=GREY)


    ax.invert_yaxis()  # labels read top-to-bottom
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{xtick_precision}f'))
    ax.set_xlim(*imp_range)

    ax.tick_params(axis='both', which='major', labelsize=label_fontsize, labelcolor=GREY)
    ax.set_yticks(ypositions)
    ax.set_yticklabels(list(I.index.values))

    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

    # ax.tick_params(axis='both', which='major', labelsize=label_fontsize)
    # ax.set_yticks(np.array(range(n_features)) + shift_upwards_from_axis)
    # rects = []
    # for fi,y in zip(imp,ypositions):
    #     print(fi,y)
    #     r = Rectangle([0.002, y-bar_width/2], fi, bar_width, color=color)
    #     rects.append(r)
    #
    # bars = PatchCollection(rects)
    # ax.add_collection(bars)

    ax.hlines(y=ypositions, xmin=left_padding, xmax=imp + left_padding, color=color,
              linewidth=bar_width, linestyles='solid')

    if False and 'Sigma' in I.columns:
        sigmas = I['Sigma'].values
        for fi,s,y in zip(imp, sigmas, ypositions):
            if fi < 0.005: continue
            left_whisker = fi + left_padding - s
            right_whisker = fi + left_padding + s
            if left_whisker < left_padding:
                left_whisker = left_padding + 0.004 # add fudge factor; mpl sees to draw bars a bit too far to right
            # print(fi, y, left_whisker, right_whisker)
            ax.plot([left_whisker, right_whisker], [y, y], lw=1.1, c='#F46C43')


    # barcontainer = ax.barh(y=range(n_features),
    #                        width=imp,
    #                        left=0.001,
    #                        height=0.9,
    #                        tick_label=I.index,
    #                        color=color)

    # # Alter appearance of each bar
    # for rect in barcontainer.patches:
    #     rect.set_linewidth(.1)
    #     rect.set_edgecolor(GREY)#'none')

    # rotate y-ticks
    if yrot is not None:
        ax.tick_params(labelrotation=yrot)

    print(ax.bbox)

    return ImpViz()
