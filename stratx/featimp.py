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


'''
def feature_rank(X: pd.DataFrame,
                 y: pd.Series,
                 catcolnames=set(),
                 normalize=True,  # make imp values 0..1
                 supervised=True,
                 n_jobs=1,
                 sort=True,  # sort by importance in descending order?
                 min_slopes_per_x=5,
                 # ignore pdp y values derived from too few slopes (usually at edges)
                 # important for getting good starting point of PD so AUC isn't skewed.
                 n_trials: int = 1,
                 n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                 verbose=False) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    I = importances(X=X, y=y, catcolnames=catcolnames,
                    normalize=normalize,
                    supervised=supervised,
                    n_jobs=n_jobs,
                    sort=sort,
                    min_slopes_per_x=min_slopes_per_x,
                    n_trials=n_trials,
                    pvalues=False,
                    n_trees=n_trees, min_samples_leaf=min_samples_leaf,
                    bootstrap=bootstrap, max_features=max_features,
                    verbose=verbose)

    F = I.copy()
    F['Importance'] = F['Importance'] / F['Sigma']
    F = F.drop('Sigma', axis=1)
    if sort:
        F = F.sort_values('Importance', ascending=False)
    return F
'''

def importances(X: pd.DataFrame,
                y: pd.Series,
                catcolnames=set(),
                normalize=True,  # make imp values 0..1
                density_weighted=True,
                supervised=True,
                n_jobs=1,
                sort=True,  # sort by importance in descending order?
                min_slopes_per_x=5,  # ignore pdp y values derived from too few slopes (usually at edges)
                # important for getting good starting point of PD so AUC isn't skewed.
                n_trials: int = 1,
                pvalues=False,  # use to get p-values for each importance; it's number trials
                n_pvalue_trials=50,  # how many trials to do to get p-values
                n_trees=1,
                min_samples_leaf=10,
                cat_min_samples_leaf=10,
                bootstrap=False, max_features=1.0,
                verbose=False) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    n,p = X.shape
    imps = np.zeros(shape=(p, n_trials))
    # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        if n_trials==1: # don't shuffle if not bootstrapping
            bootstrap_sample_idxs = range(n)
        else:
            # bootstrap_sample_idxs = resample(range(n), n_samples=n, replace=True)
            bootstrap_sample_idxs = resample(range(n), n_samples=int(n*.75), replace=False)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        I = importances_(X_, y_, catcolnames=catcolnames,
                         normalize=normalize,
                         density_weighted=density_weighted,
                         supervised=supervised,
                         n_jobs=n_jobs,
                         n_trees=n_trees,
                         min_samples_leaf=min_samples_leaf,
                         cat_min_samples_leaf=cat_min_samples_leaf,
                         min_slopes_per_x=min_slopes_per_x,
                         bootstrap=bootstrap,
                         max_features=max_features,
                         verbose=verbose)
        imps[:,i] = I

    avg_imps = np.mean(imps, axis=1)

    I = pd.DataFrame(data={'Feature': X.columns, 'Importance': avg_imps})
    I = I.set_index('Feature')

    I['Sigma'] = np.std(imps, axis=1)

    if pvalues:
        I['p-value'] = importances_pvalues(X, y, catcolnames,
                                           baseline_importances=I,
                                           supervised=supervised,
                                           n_jobs=n_jobs,
                                           n_trials=n_pvalue_trials,
                                           min_slopes_per_x=min_slopes_per_x,
                                           n_trees=n_trees,
                                           min_samples_leaf=min_samples_leaf,
                                           cat_min_samples_leaf=cat_min_samples_leaf,
                                           bootstrap=bootstrap,
                                           max_features=max_features)

    # if n_trials>1:
    #     # TODO: make 0.01 an argument or something; or maybe mean?
    #     # I['Rank'] = I['Importance'] / np.where(I['Sigma']<0.01, 1, I['Sigma'])
    #
    #     # I['Rank'] = I['Importance'] / I['Sigma']
    #     # I['Rank'] /= np.sum(I['Rank']) # normalize to 0..1
    #     # I['Rank'] = I['Rank'].fillna(0)
    #     I['Rank'] = I['Importance']
    # else:
    #     I['Rank'] = I['Importance']
    #
    # if sort=='Rank':
    #     I = I[['Rank','Importance','Sigma']]
    # else:
    #     I = I[['Importance','Sigma','Rank']]

    if sort is not None:
        I = I.sort_values('Importance', ascending=False)

    return I


def importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                 normalize=True,
                 density_weighted=True,
                 supervised=True,
                 n_jobs=1,
                 n_trees=1,
                 min_samples_leaf=10,
                 cat_min_samples_leaf=10,
                 min_slopes_per_x=5,
                 bootstrap=False, max_features=1.0,
                 verbose=False) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    all_start = timer()

    def single_feature_importance(colname):
        # print(f"Start {colname}")
        X_col = X[colname]
        if colname in catcolnames:
            leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored, merge_ignored = \
                cat_partial_dependence(X, y, colname=colname,
                                       n_trees=n_trees,
                                       min_samples_leaf=cat_min_samples_leaf,
                                       bootstrap=bootstrap,
                                       max_features=max_features,
                                       verbose=verbose,
                                       supervised=supervised)
            if density_weighted:
                # weight each cat value by how many were used to create it
                abs_avg_per_cat = np.abs(avg_per_cat)
                avg_abs_pdp = np.nansum(abs_avg_per_cat * count_per_cat) / np.sum(count_per_cat)
            else:
                # some cats have NaN, such as 0th which is often for "missing values"
                # depending on label encoding scheme.
                # no need to shift as abs(avg_per_cat) deals with negatives.
                avg_abs_pdp = np.nanmean(np.abs(avg_per_cat))

            """
            #         print(f"Ignored for {colname} = {ignored}")
            # values will straddle 0, some above, some below.
            # Used to be just this but now we weight by num of each cat:
            # Ok, I'm back to this from the below because weight at each x really
            # doesn't change how much an x location pushes on y.
            avg_abs_pdp = np.nanmean(np.abs(avg_per_cat))

            # avg_per_cat is currently deltas from mean(y) but we want to use min avg_per_cat
            # as reference point, zeroing that one out. All others will be relative to that
            # min cat value
            # DON'T do this actually. seems to accentuate the impact; leave as relative to
            # overall mean
            #avg_per_cat -= np.nanmin(avg_per_cat)

            # group by cat and get count
            # abs_avg_per_cat = np.abs(avg_per_cat[~np.isnan(avg_per_cat)])
            # uniq_cats = np.where(~np.isnan(avg_per_cat))[0]
            # cat_counts = [len(np.where(X_col == cat)[0]) for cat in uniq_cats]
            # avg_abs_pdp = np.sum(abs_avg_per_cat * cat_counts) / np.sum(cat_counts)
            """
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
            if density_weighted:
                _, pdpx_counts = np.unique(X_col[np.isin(X_col, pdpx)], return_counts=True)
                if len(pdpx_counts)>0:
                    # weighted average of pdpy using pdpx_counts
                    avg_abs_pdp = np.sum(np.abs(pdpy * pdpx_counts)) / np.sum(pdpx_counts)
                else:
                    avg_abs_pdp = np.mean(np.abs(pdpy))
            else:
                avg_abs_pdp = np.mean(np.abs(pdpy))
        print(f"{colname}:{avg_abs_pdp:.3f} mass")
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
                        baseline_importances=None, # importances to use as baseline; must be in X column order!
                        supervised=True,
                        n_jobs=1,
                        n_trials: int = 1,
                        min_slopes_per_x=5,
                        n_trees=1,
                        min_samples_leaf=10,
                        cat_min_samples_leaf=10,
                        bootstrap=False,
                        max_features=1.0):
    """
    For each feature, compute and return empirical p-values.  The idea is to shuffle y
    and then compute feature importances; do this repeatedly to get a null distribution.
    The importances for feature j form a distribution and we can count how many times the
    importance value (obtained with shuffled y) reaches the importance value computed
    using unshuffled y.
    """
    I_baseline = baseline_importances
    if baseline_importances is None:
        I_baseline = importances(X, y, catcolnames=catcolnames, sort=False,
                                 min_slopes_per_x=min_slopes_per_x,
                                 supervised=supervised,
                                 n_jobs=n_jobs,
                                 n_trees=n_trees,
                                 min_samples_leaf=min_samples_leaf,
                                 cat_min_samples_leaf=cat_min_samples_leaf,
                                 bootstrap=bootstrap,
                                 max_features=max_features)

    counts = np.zeros(shape=X.shape[1])
    for i in range(n_trials):
        I = importances(X, y.sample(frac=1.0, replace=False),
                        catcolnames=catcolnames, sort=False,
                        min_slopes_per_x=min_slopes_per_x,
                        supervised=supervised,
                        n_jobs=n_jobs,
                        n_trees=n_trees,
                        min_samples_leaf=min_samples_leaf,
                        cat_min_samples_leaf=cat_min_samples_leaf,
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
                     xlabel=None,
                     show: ('Rank', 'Importance') = 'Rank',
                     yrot=0,
                     title_fontsize=11,
                     label_fontsize=10,
                     fontname="Arial",
                     width:float=3, # if no figsize, use this width
                     height:float=None,
                     bar_width=13, # in pixels
                     bar_spacing = 4, # in pixels
                     imp_range=(0, 1.0),
                     dpi=150,
                     color='#4574B4',#'#D9E6F5',
                     whisker_color='black', #'#F46C43'
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
    if show=='Rank' and 'Rank' not in I.columns:
        show='Importance'
    I = I.sort_values(show, ascending=True)
    n_features = len(I)
    left_padding = 0.01

    ppi = 72 # matplotlib has this hardcoded. E.g., see https://github.com/matplotlib/matplotlib/blob/40dfc353aa66b93fd0fbc55ca1f51701202c0549/lib/matplotlib/axes/_base.py#L694
    imp = I[show].values

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

    if show!='Rank' and 'Sigma' in I.columns:
        sigmas = I['Sigma'].values
        for fi,s,y in zip(imp, sigmas, ypositions):
            if fi < 0.005: continue
            s *= 2 # show 2 sigma
            left_whisker = fi + left_padding - s
            right_whisker = fi + left_padding + s
            if left_whisker < left_padding:
                left_whisker = left_padding + 0.004 # add fudge factor; mpl sees to draw bars a bit too far to right
            # print(fi, y, left_whisker, right_whisker)
            ax.plot([left_whisker, right_whisker], [y, y], lw=whisker_linewidth, c=whisker_color)
            ax.plot([left_whisker, left_whisker], [y - whisker_barwidth, y + whisker_barwidth], lw=whisker_linewidth, c=whisker_color)
            ax.plot([right_whisker, right_whisker], [y - whisker_barwidth, y + whisker_barwidth], lw=whisker_linewidth, c=whisker_color)


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
