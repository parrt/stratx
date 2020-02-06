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


def importances(X: pd.DataFrame,
                y: pd.Series,
                catcolnames=set(),
                normalize=True,  # make imp values 0..1
                supervised=True,
                n_jobs=1,
                sortby='Importance',  # sort by importance or impact
                min_slopes_per_x=15,  # ignore pdp y values derived from too few slopes (usually at edges); for smallerData sets, drop this to five or so
                # important for getting good starting point of PD so AUC isn't skewed.
                n_trials: int = 1,
                pvalues=False,  # use to get p-values for each importance; it's number trials
                pvalues_n_trials=80,
                n_trees=1,
                min_samples_leaf=10,
                cat_min_samples_leaf=5,
                bootstrap=False, max_features=1.0,
                verbose=False) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    X = compress_catcodes(X, catcolnames)

    n,p = X.shape
    impact_trials = np.zeros(shape=(p, n_trials))
    importance_trials = np.zeros(shape=(p, n_trials))
    # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        if n_trials==1: # don't shuffle if not bootstrapping
            bootstrap_sample_idxs = range(n)
        else:
            # bootstrap_sample_idxs = resample(range(n), n_samples=n, replace=True)
            bootstrap_sample_idxs = resample(range(n), n_samples=int(n*.75), replace=False)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        impacts, importances = importances_(X_, y_, catcolnames=catcolnames,
                                            normalize=normalize,
                                            supervised=supervised,
                                            n_jobs=n_jobs,
                                            n_trees=n_trees,
                                            min_samples_leaf=min_samples_leaf,
                                            cat_min_samples_leaf=cat_min_samples_leaf,
                                            min_slopes_per_x=min_slopes_per_x,
                                            bootstrap=bootstrap,
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
    # I['Rank'] = I['Importance']
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
                                bootstrap=bootstrap,
                                max_features=max_features)
        I['Impact p-value'] = importance_pvalues
        I['Importance p-value'] = importance_pvalues
        # I['Rank'] = I['Importance'] * (1.0 - importance_pvalues)

    # knock out any feature whose importance is less than 2 sigma
    # I['Rank'] = I['Rank'] * ((I['Importance'] - 2 * I['Importance sigma']) > 0).astype(int)

    if sortby:
        I = I.sort_values(sortby, ascending=False)

    # Set reasonable column order
    I = I[['Importance', 'Importance sigma', 'Importance p-value',
           'Impact', 'Impact sigma', 'Impact p-value']]
    if n_trials==1:
        I = I.drop(['Importance sigma', 'Impact sigma'], axis=1)
    if not pvalues:
        I = I.drop(['Importance p-value', 'Impact p-value'], axis=1)
    return I


def importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                 normalize=True,
                 supervised=True,
                 n_jobs=1,
                 n_trees=1,
                 min_samples_leaf=10,
                 cat_min_samples_leaf=5,
                 min_slopes_per_x=5,
                 bootstrap=False, max_features=1.0,
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
                                                bootstrap=bootstrap,
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
                                                         bootstrap=bootstrap,
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
                              bootstrap=False, max_features=1.0,
                              verbose=False):
    "Return impact=unweighted avg abs, importance=weighted avg abs"
    # print(f"Start {colname}")
    if colname in catcolnames:
        leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored, merge_ignored = \
            cat_partial_dependence(X, y, colname=colname,
                                   n_trees=n_trees,
                                   min_samples_leaf=cat_min_samples_leaf,
                                   bootstrap=bootstrap,
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
                               bootstrap=bootstrap,
                               max_features=max_features,
                               verbose=verbose,
                               parallel_jit=n_jobs == 1,
                               supervised=supervised)
        impact, importance = compute_importance(X[colname], pdpx, pdpy)
    # print(f"{colname}:{avg_abs_pdp:.3f} mass")
    # print(f"Stop {colname}")
    return impact, importance


def compute_importance(X_col, pdpx, pdpy):
    # TODO: might include samples not included in pdpy due to thresholding, ignored
    _, pdpx_counts = np.unique(X_col[np.isin(X_col, pdpx)], return_counts=True)
    if len(pdpx_counts) > 0:
        # weighted average of pdpy using pdpx_counts
        weighted_avg_abs_pdp = np.sum(np.abs(pdpy * pdpx_counts)) / np.sum(pdpx_counts)
    else:
        weighted_avg_abs_pdp = np.mean(np.abs(pdpy))

    # unweighted
    avg_abs_pdp = np.mean(np.abs(pdpy))
    return avg_abs_pdp, weighted_avg_abs_pdp


def cat_compute_importance(avg_per_cat, count_per_cat):
    # weight each cat value by how many were used to create it
    abs_avg_per_cat = np.abs(avg_per_cat)
    # above_threshold = np.where(count_per_cat>20)[0]
    # weighted_avg_abs_pdp = np.nansum(abs_avg_per_cat[above_threshold] * count_per_cat[above_threshold]) / np.sum(count_per_cat[above_threshold])
    weighted_avg_abs_pdp = np.nansum(abs_avg_per_cat * count_per_cat) / np.sum(count_per_cat)

    # do unweighted
    # some cats have NaN, such as 0th which is often for "missing values"
    # depending on label encoding scheme.
    # no need to shift as abs(avg_per_cat) deals with negatives.
    avg_abs_pdp = np.nanmean(abs_avg_per_cat)
    return avg_abs_pdp, weighted_avg_abs_pdp


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
                        bootstrap=False,
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
                         bootstrap=bootstrap,
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
                                            bootstrap=bootstrap,
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
                     xlabel=None,
                     sortby: ('Impact', 'Importance') = 'Impact',
                     yrot=0,
                     title_fontsize=11,
                     label_fontsize=10,
                     fontname="Arial",
                     width:float=3,  # if no figsize, use this width
                     height:float=None,
                     bar_width=13,  # in pixels
                     bar_spacing = 4,  # in pixels
                     imp_range=(0, 1.0),
                     dpi=150,
                     color='#4574B4',  #'#D9E6F5',
                     whisker_color='black',  #'#F46C43'
                     whisker_linewidth = .6,
                     whisker_barwidth = .1,
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
    if sortby not in I.index.values:
        sortby = 'Importance'
    I = I.sort_values(sortby, ascending=True)
    n_features = len(I)
    left_padding = 0.01

    ppi = 72 # matplotlib has this hardcoded. E.g., see https://github.com/matplotlib/matplotlib/blob/40dfc353aa66b93fd0fbc55ca1f51701202c0549/lib/matplotlib/axes/_base.py#L694
    imp = I[sortby].values

    barcounts = np.array([f.count('\n')+1 for f in I.index])
    N = np.sum(barcounts)

    ypositions = np.array( range(n_features) )

    if ax is None:
        if height is None:
            # we need a bar for each feature and half a bar on bottom + half a bar above
            # on top then spacing in between N bars (N-1 spaces)
            height_in_pixels = N * bar_width + 2 * bar_width / 2 + (N-1) * bar_spacing
            # space_for x axis (labels etc...)
            fudge = 25
            if xlabel is not None: fudge += 12
            if title is not None: fudge += 12
            fig, ax = plt.subplots(1, 1, figsize=(width, (height_in_pixels + fudge) / ppi), dpi=dpi)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)

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
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname, color=GREY, pad=0)

    #ax.invert_yaxis()  # labels read top-to-bottom
    ax.xaxis.set_major_formatter(FormatStrFormatter(f'%.{xtick_precision}f'))
    ax.set_xlim(*imp_range)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    ax.tick_params(axis='both', which='major', labelsize=label_fontsize, labelcolor=GREY)
    ax.set_ylim(-.6, n_features-.5) # leave room for about half a bar below
    ax.set_yticks(list(ypositions))
    ax.set_yticklabels(list(I.index.values))

    for tick in ax.get_xticklabels():
        tick.set_fontname(fontname)
    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

    # rects = []
    # for fi,y in zip(imp,ypositions):
    #     print(fi,y)
    #     r = Rectangle([0.01, y-.45], fi, .9, color=color)
    #     rects.append(r)

    # bars = PatchCollection(rects)
    # ax.add_collection(bars)

    ax.hlines(y=ypositions, xmin=left_padding, xmax=imp + left_padding, color=color,
              linewidth=bar_width, linestyles='solid')

    if sortby+' sigma' in I.columns:
        sigmas = I[sortby+' sigma'].values
        for fi,s,y in zip(imp, sigmas, ypositions):
            if fi < 0.005: continue
            s *= 2 # show 2 sigma
            left_whisker = fi + left_padding - s
            right_whisker = fi + left_padding + s
            left_edge = left_whisker
            c = whisker_color
            if left_whisker < left_padding:
                left_edge = left_padding + 0.004 # add fudge factor; mpl sees to draw bars a bit too far to right
                c = '#CB1B1F'
            # print(fi, y, left_whisker, right_whisker)
            # horiz line
            ax.plot([left_edge, right_whisker],  [y, y], lw=whisker_linewidth, c=c)
            # left vertical
            if left_whisker >= left_padding:
                ax.plot([left_whisker, left_whisker],   [y - whisker_barwidth, y + whisker_barwidth], lw=whisker_linewidth, c=c)
            # right vertical
            ax.plot([right_whisker, right_whisker], [y - whisker_barwidth, y + whisker_barwidth], lw=whisker_linewidth, c=c)


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

    return ImpViz()
