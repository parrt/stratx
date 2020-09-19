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
import matplotlib.pyplot as plt
import matplotlib as mpl
from  matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from collections import defaultdict
import tempfile
from os import getpid
from matplotlib.ticker import FormatStrFormatter
import time

import stratx.featimp
import stratx.partdep
import stratx.ice


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
                     sortby: ('Impact', 'Importance') = 'Importance',
                     highlight_high_stddev=2.0,
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
    if isinstance(I, pd.DataFrame):
        if sortby not in I.columns.values:
            sortby = 'Importance'
        I = stratx.featimp.Isortby(I, sortby, stddev_threshold=highlight_high_stddev, ascending=True) # we plot in reverse order

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


def plot_stratpd(X:pd.DataFrame, y:pd.Series, colname:str, targetname:str,
                 min_slopes_per_x=5,
                 n_trials=1,
                 n_trees=1,
                 min_samples_leaf=15,
                 bootstrap=True,
                 subsample_size=.75,
                 rf_bootstrap=False,
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
                 show_slope_lines=False,
                 show_slope_counts=False,
                 show_x_counts=True,
                 show_impact=False,
                 show_impact_dots=True,
                 show_impact_line=True,
                 hide_top_right_axes=True,
                 pdp_marker_size=2,
                 pdp_marker_alpha=.5,
                 pdp_line_width=.5,
                 slope_line_color='#2c7fb8',
                 slope_line_width=.5,
                 slope_line_alpha=.3,
                 pdp_line_color='black',
                 pdp_marker_color='black',
                 pdp_marker_cmap='coolwarm',
                 impact_fill_color='#FEF5DC',
                 impact_pdp_color='#D73028',
                 impact_marker_size=3,
                 fontname='Arial',
                 title_fontsize=11,
                 label_fontsize=10,
                 ticklabel_fontsize=10,
                 barchart_size=0.20,
                 barchar_alpha=1.0, # if show_slope_counts, what ratio of vertical space should barchart use at bottom?
                 barchar_color='#BABABA',
                 verbose=False,
                 figsize=None
                 ):
    """
    Plot the partial dependence of X[colname] on y for numerical X[colname].

    Key parameters:

    :param X: Dataframe with all explanatory variables

    :param y: Series or vector with response variable

    :param colname: which X[colname] (a string) to compute partial dependence for

    :param targetname: for plotting purposes, will what is the y axis label?

    :param n_trials:  How many times should we run the stratpd algorithm and get PD
                      curves, using bootstrapped or subsample sets? Default is 1.

    :param min_samples_leaf Key hyper parameter to the stratification
                            process. The default is 15 and usually
                            works out pretty well.  It controls the
                            minimum number of observations in each
                            decision tree leaf used to stratify X other than colname.
                            Generally speaking,
                            smaller values lead to more confidence
                            that fluctuations in y are due solely to
                            X[colname], but more observations per leaf allow
                            StratPD to capture more nonlinearities and
                            make it less susceptible to noise. As the
                            leaf size grows, however, one risks
                            introducing contributions from X not colname into
                            the relationship between X[colname] and y. At the
                            extreme, the decision tree would consist
                            of a single leaf node containing all
                            observations, leading to a marginal not
                            partial dependence curve.

    :param min_slopes_per_x: ignore any partial derivatives estimated
                             with too few observations. Dropping uncertain partial derivatives
                             greatly improves accuracy and stability. Partial dependences
                             computed by integrating over local partial derivatives are highly
                             sensitive to partial derivatives computed at the left edge of any
                             X[colname]’s range because imprecision at the left edge affects the entire
                             curve. This presents a problem when there are few samples with X[colname]
                             values at the extreme left. Default is 5.

    Returns:

        pdpx            The non-NaN unique X[colname] values

        pdpy            The effect of each non-NaN unique X[colname] on y; effectively
                        the cumulative sum of the partial derivative of y with respect to
                        X[colname]. The first value is always 0.

        ignored         How many samples from len(X) total records did we have to
                        ignore because samples in leaves had identical X[colname]
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

    X_col = X[colname].values.round(decimals=10)

    all_pdpx = []
    all_pdpy = []
    impacts = []
    importances = []
    n = len(X)
    ignored = 0
    for i in range(n_trials):
        if n_trials>1:
            if bootstrap:
                idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
            else: # subsample
                idxs = resample(range(n), n_samples=int(n*subsample_size), replace=False)
            X_, y_ = X.iloc[idxs], y.iloc[idxs]
        else:
            X_, y_ = X, y

        leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored_ = \
            stratx.partdep.partial_dependence(X=X_, y=y_, colname=colname,
                                              min_slopes_per_x=min_slopes_per_x,
                                              n_trees=n_trees,
                                              min_samples_leaf=min_samples_leaf,
                                              rf_bootstrap=rf_bootstrap,
                                              max_features=max_features,
                                              supervised=supervised,
                                              verbose=verbose)
        ignored += ignored_
        # print("ignored", ignored_, "pdpy", pdpy)
        all_pdpx.append(pdpx)
        all_pdpy.append(pdpy)
        impact, importance = stratx.featimp.compute_importance(X_col, pdpx, pdpy)
        impacts.append(impact)
        importances.append(importance)

    ignored /= n_trials # average number of x values ignored across trials

    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

    avg_pdp_marker_size = pdp_marker_size
    if show_all_pdp and n_trials>1:
        sorted_by_imp = np.argsort([np.mean(np.abs(v)) for v in all_pdpy])
        cmap = plt.get_cmap(pdp_marker_cmap)
        ax.set_prop_cycle(color=cmap(np.linspace(0,1,num=n_trials)))
        for i in range(n_trials):
            ax.plot(all_pdpx[sorted_by_imp[i]], all_pdpy[sorted_by_imp[i]],
                    '.', markersize=pdp_marker_size, alpha=pdp_marker_alpha)
        avg_pdp_marker_size += 2

    # Get avg curve, reset pdpx and pdpy to the average
    pdpx, pdpy = avg_pd_curve(all_pdpx, all_pdpy)
    ax.plot(pdpx, pdpy, '.', c=pdp_marker_color, markersize=avg_pdp_marker_size, label=colname)

    if show_pdp_line:
        ax.plot(pdpx, pdpy, lw=pdp_line_width, c=pdp_line_color)

    if len(pdpy)==0:
        raise ValueError("No partial dependence y values, often due to value of min_samples_leaf that is too small or min_slopes_per_x that is too large")

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
    # else:
    #     ax.set_xlim(*domain)
    if yrange is not None:
        ax.set_ylim(*yrange)
    else:
        ax.set_ylim(min_y, max_y)

    leave_room_scaler = 1.3
    x_width = max(pdpx) - min(pdpx) + 1
    count_bar_width = x_width / len(pdpx)
    if count_bar_width/x_width < 0.002:
        count_bar_width = x_width * 0.002 # don't make them so skinny they're invisible
    # print(f"x_width={x_width:.2f}, count_bar_width={count_bar_width}")
    if show_x_counts:
        _, pdpx_counts = np.unique(X_col[np.isin(X_col, pdpx)], return_counts=True)
        ax2 = ax.twinx()
        # scale y axis so the max count height is 10% of overall chart
        ax2.set_ylim(0, max(pdpx_counts) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(pdpx_counts)]))
        ax2.bar(x=pdpx, height=pdpx_counts, width=count_bar_width,
                facecolor=barchar_color, align='center', alpha=barchar_alpha)
        ax2.set_ylabel(f"$x$ count, ignored={ignored:.0f}", labelpad=-10, fontsize=label_fontsize,
                       fontstretch='extra-condensed',
                       fontname=fontname,
                       horizontalalignment='right',
                       y=1.0)

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
    else:
        if hide_top_right_axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    if n_trials==1 and show_slope_counts:
        ax2 = ax.twinx()
        # scale y axis so the max count height is barchart_size of overall chart
        ax2.set_ylim(0, max(slope_counts_at_x) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(slope_counts_at_x)]))
        ax2.bar(x=pdpx, height=slope_counts_at_x, width=count_bar_width,
                facecolor=barchar_color, align='center', alpha=barchar_alpha)
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
        ax.fill_between(pdpx, pdpy, [0] * len(pdpx), color=impact_fill_color)
        if show_impact_dots:
            ax.scatter(pdpx, pdpy, s=impact_marker_size, c=impact_pdp_color)
        if show_impact_line:
            ax.plot(pdpx, pdpy, lw=.3, c='grey')

    if show_xlabel:
        xl = colname
        if show_impact:
            xl += f" (Impact {np.mean(impact):.2f}, importance {np.mean(importance):.2f})"
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


def plot_catstratpd(X, y,
                    colname,  # X[colname] expected to be numeric codes
                    targetname,
                    catnames=None,
                    n_trials=1,
                    subsample_size = .75,
                    bootstrap=True,
                    ax=None,
                    n_trees=1,
                    rf_bootstrap=False,
                    min_samples_leaf=5,
                    max_features=1.0,
                    yrange=None,
                    title=None,
                    show_x_counts=True,
                    show_all_pdp=True,
                    pdp_marker_size=6,
                    pdp_marker_alpha=.6,
                    pdp_color='#A5D9B5',
                    fontname='Arial',
                    title_fontsize=11,
                    label_fontsize=10,
                    barchart_size=0.20,
                    barchar_alpha=0.9,
                    ticklabel_fontsize=10,
                    min_y_shifted_to_zero=False,
                    leftmost_shifted_to_zero=False,  # either this or min_y_shifted_to_zero can be true
                    # easier to read if values are relative to 0 (usually); do this for high cardinality cat vars
                    mean_centered=False,
                    show_xlabel=True,
                    show_unique_cat_xticks=False,
                    show_xticks=True,
                    show_ylabel=True,
                    show_impact=False,
                    verbose=False,
                    sort_by_y=False,
                    figsize=(5,3)):
    """
    Plot the partial dependence of categorical variable X[colname] on y.
    Warning: cat columns are assumed to be label encoded as unique integers. This
    function uses the cat code as a raw index internally. So if you have two cat
    codes 1 and 1000, this function allocates internal arrays of size 1000+1.

    Key parameters:

    :param X: Dataframe with all explanatory variables

    :param y: Series or vector with response variable

    :param colname: which X[colname] (a string) to compute partial dependence for

    :param targetname: for plotting purposes, will what is the y axis label?

    :param catnames: dict or array mapping catcode to catname, used for plotting x axis

    :param n_trials:  How many times should we run the catstratpd algorithm and get PD
                      curves, using bootstrapped or subsample sets? Default is 1.

    :param min_samples_leaf Key hyper parameter to the stratification
                            process. The default is 5 and usually
                            works out pretty well.  It controls the
                            minimum number of observations in each
                            decision tree leaf used to stratify X other than colname.
                            Generally speaking, smaller values lead to more confidence
                            that fluctuations in y are due solely to
                            X[colname], but more observations per leaf allow
                            CatStratPD to capture more relationships and
                            make it less susceptible to noise. As the
                            leaf size grows, however, one risks
                            introducing contributions from X not colname into
                            the relationship between X[colname] and y. At the
                            extreme, the decision tree would consist
                            of a single leaf node containing all
                            observations, leading to a marginal not
                            partial dependence curve.
    """
    if ax is None:
        if figsize is not None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1, 1)

    uniq_catcodes = np.unique(X[colname])

    X_col = X[colname]
    n = len(X_col)

    ''' replaced with np.nanmean
    def avg_pd_catvalues(all_avg_per_cat):
        """For each unique catcode, sum and count avg_per_cat values found among trials"""
        m = np.zeros(shape=(max_catcode+1,))
        c = np.zeros(shape=(max_catcode+1,), dtype=int)
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
    '''

    impacts = []
    importances = []
    all_avg_per_cat = []
    ignored = 0
    for i in range(n_trials):
        if n_trials>1:
            if bootstrap:
                idxs = resample(range(n), n_samples=n, replace=True)
            else: # use subsetting
                idxs = resample(range(n), n_samples=int(n * subsample_size), replace=False)
            X_, y_ = X.iloc[idxs], y.iloc[idxs]
        else:
            X_, y_ = X, y

        leaf_deltas, leaf_counts, avg_per_cat, count_per_cat, ignored_ = \
            stratx.partdep.cat_partial_dependence(X_, y_,
                                                  max_catcode=np.max(X_col),
                                                  colname=colname,
                                                  n_trees=n_trees,
                                                  min_samples_leaf=min_samples_leaf,
                                                  max_features=max_features,
                                                  rf_bootstrap=rf_bootstrap,
                                                  verbose=verbose)
        impact, importance = stratx.featimp.cat_compute_importance(avg_per_cat, count_per_cat)
        impacts.append(impact)
        importances.append(importance)
        ignored += ignored_
        all_avg_per_cat.append( avg_per_cat )

    all_avg_per_cat = np.array(all_avg_per_cat)

    ignored /= n_trials # average number of x values ignored across trials

    # average down the matrix of all_avg_per_cat across trials to get average per cat
    # combined_avg_per_cat = avg_pd_catvalues(all_avg_per_cat)
    if n_trials>1:
        combined_avg_per_cat = np.nanmean(all_avg_per_cat, axis=0)
    else:
        combined_avg_per_cat = all_avg_per_cat.flatten()
    # print("start of combined_avg_per_cat =", combined_avg_per_cat[uniq_catcodes][0:20])
    # print("mean(pdpy)", np.nanmean(combined_avg_per_cat))

    if leftmost_shifted_to_zero:
        combined_avg_per_cat -= combined_avg_per_cat[np.isfinite(combined_avg_per_cat)][0]
    if min_y_shifted_to_zero:
        combined_avg_per_cat -= np.nanmin(combined_avg_per_cat)
    if mean_centered:
        combined_avg_per_cat -= np.nanmean(combined_avg_per_cat)

    impact_order = np.argsort(impacts)
    # print("impacts", impacts)
    # print("avg impact", np.mean(impacts))

    cmap = plt.get_cmap('coolwarm')
    colors=cmap(np.linspace(0, 1, num=n_trials))
    min_y = 9999999999999
    max_y = -min_y

    for i in range(0,n_trials): # find min/max from all trials
        avg_per_cat = all_avg_per_cat[i]
        if np.nanmin(avg_per_cat) < min_y:
            min_y = np.nanmin(avg_per_cat)
        if np.nanmax(avg_per_cat) > max_y:
            max_y = np.nanmax(avg_per_cat)

    # Show a dot for each cat in all trials
    n_catcodes = len(uniq_catcodes)
    cat_x = range(n_catcodes)
    if show_all_pdp and n_trials>1:
        for i in range(1,n_trials): # only do if > 1 trial
            ax.plot(cat_x, all_avg_per_cat[i][uniq_catcodes], '.', c=mpl.colors.rgb2hex(colors[impact_order[i]]),
                    markersize=pdp_marker_size, alpha=pdp_marker_alpha)

    '''
    # Show avg line
    segments = []
    for cat, delta in zip(range(len(uniq_catcodes)), combined_avg_per_cat[uniq_catcodes]):
        one_line = [(cat-0.5, delta), (cat+0.5, delta)]
        segments.append(one_line)
        # ax.plot([cat-0.5,cat+0.5], [delta,delta], '-',
        #         lw=1.0, c=pdp_color, alpha=pdp_marker_alpha)
        # ax.plot(range(len(uniq_catcodes)), avg_delta, '.', c='k', markersize=pdp_marker_size + 1)
    lines = LineCollection(segments, alpha=pdp_marker_alpha, color=pdp_color, linewidths=pdp_marker_lw)
    ax.add_collection(lines)
    '''

    cat_heights = combined_avg_per_cat[uniq_catcodes]
    if sort_by_y and not show_unique_cat_xticks:
        cat_heights = sorted(cat_heights)

    barcontainer = ax.bar(x=cat_x,
                          height=cat_heights,
                          color=pdp_color)
    # Alter appearance of each bar
    for rect in barcontainer.patches:
        rect.set_linewidth(.1)
        rect.set_edgecolor('#444443')

    leave_room_scaler = 1.3

    if yrange is not None:
        ax.set_ylim(*yrange)
    else:
        ax.set_ylim(min_y*1.05, max_y*1.05)

    if show_x_counts:
        # Only show cat counts for those which are present in X[colname] (unlike stratpd plot)
        _, cat_counts = np.unique(X_col[np.isin(X_col, uniq_catcodes)], return_counts=True)
        count_bar_width=1
        ax2 = ax.twinx()
        # scale y axis so the max count height is 10% of overall chart
        ax2.set_ylim(0, max(cat_counts) * 1/barchart_size)
        # draw just 0 and max count
        ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(cat_counts)]))
        ax2.bar(x=cat_x, height=cat_counts, width=count_bar_width,
                facecolor='#BABABA', align='center', alpha=barchar_alpha)
        ax2.set_ylabel(f"$x$ count, ignored={ignored:.0f}", labelpad=-5, fontsize=label_fontsize,
                       fontstretch='extra-condensed',
                       fontname=fontname)
        # shift other y axis down barchart_size to make room
        if yrange is not None:
            ax.set_ylim(yrange[0]-(yrange[1]-yrange[0])*barchart_size * leave_room_scaler, yrange[1])
        else:
            ax.set_ylim(min_y-(max_y-min_y)*barchart_size * leave_room_scaler, max_y)
        plt.setp(ax2.get_xticklabels(), visible=False)
        for tick in ax2.get_yticklabels():
            tick.set_fontname(fontname)
        ax2.spines['top'].set_linewidth(.5)
        ax2.spines['right'].set_linewidth(.5)
        ax2.spines['left'].set_linewidth(.5)
        ax2.spines['bottom'].set_linewidth(.5)

    ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)

    # leave .8 on either size of graph
    ax.set_xlim(0 - .8, n_catcodes - 1 + 0.8)
    if show_unique_cat_xticks:
        ax.set_xticks(cat_x)
        if catnames is not None:
            labels = [catnames[c] for c in uniq_catcodes]
            ax.set_xticklabels(labels)
        else:
            ax.set_xticklabels(uniq_catcodes)
        for tick in ax.get_xticklabels():
            tick.set_fontname(fontname)
    elif not show_xticks:
        ax.set_xticks([])
        ax.set_xticklabels([])

    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if show_xlabel:
        label = colname
        if show_impact:
            label += f" (Impact {np.nanmean(np.abs(combined_avg_per_cat)):.2f})"
        ax.set_xlabel(label, fontsize=label_fontsize, fontname=fontname)
    if show_ylabel:
        ax.set_ylabel(targetname, fontsize=label_fontsize, fontname=fontname)
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

    for tick in ax.get_yticklabels():
        tick.set_fontname(fontname)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)

    return uniq_catcodes, combined_avg_per_cat, ignored


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


def plot_catstratpd_gridsearch(X, y, colname, targetname,
                               n_trials=1,
                               min_samples_leaf_values=(2, 5, 10, 20, 30),
                               min_y_shifted_to_zero=True,  # easier to read if values are relative to 0 (usually); do this for high cardinality cat vars
                               show_xticks=True,
                               show_impact=False,
                               show_all_cat_deltas=True,
                               catnames=None,
                               yrange=None,
                               cellwidth=2.5,
                               cellheight=2.5):

    ncols = len(min_samples_leaf_values)
    fig, axes = plt.subplots(1, ncols + 1,
                             figsize=((ncols + 1) * cellwidth, cellheight),
                             sharey=True)

    marginal_catplot_(X, y, colname, targetname, catnames=catnames, ax=axes[0], alpha=0.05,
                      show_xticks=show_xticks)
    axes[0].set_title("Marginal", fontsize=10)

    col = 1
    for msl in min_samples_leaf_values:
        #print(f"---------- min_samples_leaf={msl} ----------- ")
        if yrange is not None:
            axes[col].set_ylim(yrange)
        try:
            uniq_catcodes, combined_avg_per_cat, ignored = \
                plot_catstratpd(X, y, colname, targetname, ax=axes[col],
                                n_trials=n_trials,
                                min_samples_leaf=msl,
                                catnames=catnames,
                                yrange=yrange,
                                n_trees=1,
                                show_impact=show_impact,
                                show_unique_cat_xticks=show_xticks,
                                show_ylabel=False,
                                min_y_shifted_to_zero=min_y_shifted_to_zero)
        except ValueError:
            axes[col].set_title(f"Can't gen: leafsz={msl}", fontsize=8)
        else:
            axes[col].set_title(f"leafsz={msl}, ign'd={ignored / len(X):.1f}%", fontsize=9)
        col += 1


def plot_stratpd_gridsearch(X, y, colname, targetname,
                            min_samples_leaf_values=(2,5,10,20,30),
                            min_slopes_per_x_values=(5,), # Show default count only by default
                            n_trials=1,
                            yrange=None,
                            xrange=None,
                            show_regr_line=False,
                            show_slope_lines=False,
                            show_impact=False,
                            show_slope_counts=False,
                            show_x_counts=True,
                            marginal_alpha=.05,
                            slope_line_alpha=.1,
                            pdp_marker_size=2,
                            title_fontsize=8,
                            label_fontsize=7,
                            ticklabel_fontsize=7,
                            cellwidth=2.5,
                            cellheight=2.5):
    ncols = len(min_samples_leaf_values)
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
                xrange_ = xrange
                if xrange is None:
                    xrange_ = (np.min(X[colname]), np.max(X[colname]))
                pdpx, pdpy, ignored = \
                    plot_stratpd(X, y, colname, targetname, ax=axes[row][col],
                                 min_samples_leaf=msl,
                                 min_slopes_per_x=min_slopes_per_x,
                                 n_trials=n_trials,
                                 xrange=xrange_,
                                 yrange=yrange,
                                 n_trees=1,
                                 show_ylabel=False,
                                 pdp_marker_size=pdp_marker_size,
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


def plot_ice(ice, colname, targetname="target", ax=None, linewidth=.5, linecolor='#9CD1E3',
             min_y_shifted_to_zero=False, # easier to read if values are relative to 0 (usually)
             alpha=.1, title=None, xrange=None, yrange=None, pdp=True, pdp_linewidth=.5, pdp_alpha=1,
             pdp_color='black', show_xlabel=True, show_ylabel=True):
    start = time.time()
    if ax is None:
        fig, ax = plt.subplots(1,1)

    avg_y = np.mean(ice[1:], axis=0)

    min_pdp_y = avg_y[0]
    # if 0 is in x feature and not on left/right edge, get y at 0
    # and shift so that is x,y 0 point.
    linex = ice.iloc[0,:] # get unique x values from first row
    nx = len(linex)
    if linex[int(nx*0.05)]<0 or linex[-int(nx*0.05)]>0:
        closest_x_to_0 = np.argmin(np.abs(np.array(linex - 0.0))) # do argmin w/o depr warning
        min_pdp_y = avg_y[closest_x_to_0]

    lines = stratx.ice.ice2lines(ice)
    if min_y_shifted_to_zero:
        lines[:,:,1] = lines[:,:,1] - min_pdp_y
    # lines[:,:,0] scans all lines, all points in a line, and gets x column
    minx, maxx = np.min(lines[:,:,0]), np.max(lines[:,:,0])
    miny, maxy = np.min(lines[:,:,1]), np.max(lines[:,:,1])
    if yrange is not None:
        ax.set_ylim(*yrange)
    # else:
    #     ax.set_ylim(miny, maxy)
    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)
    lines = LineCollection(lines, linewidth=linewidth, alpha=alpha, color=linecolor)
    ax.add_collection(lines)

    if xrange is not None:
        ax.set_xlim(*xrange)
    # else:
    #     ax.set_xlim(minx, maxx)

    uniq_x = ice.iloc[0, :]
    if min_y_shifted_to_zero:
        pdp_curve = avg_y - min_pdp_y
    else:
        pdp_curve = avg_y

    if pdp:
        ax.plot(uniq_x, pdp_curve,
                alpha=pdp_alpha, linewidth=pdp_linewidth, c=pdp_color)

    stop = time.time()
    # print(f"plot_ICE {stop - start:.3f}s")
    return uniq_x, pdp_curve


def plot_catice(ice, colname, targetname,
                catnames,  # cat names indexed by cat code
                ax=None,
                min_y_shifted_to_zero=False,
                color='#9CD1E3',
                alpha=.1, title=None, yrange=None, pdp=True,
                pdp_marker_size=.5, pdp_alpha=1,
                pdp_color='black',
                marker_size=10,
                show_xlabel=True, show_ylabel=True,
                show_xticks=True):
    start = time.time()
    if ax is None:
        fig, ax = plt.subplots(1,1)

    ncats = len(catnames)

    avg_y = np.mean(ice[1:], axis=0)

    lines = stratx.ice.ice2lines(ice)

    nobs = lines.shape[0]

    catcodes, _, catcode2name = getcats(None, colname, catnames)

    avg_y = np.mean(ice[1:], axis=0)
    min_pdp_y = np.min(avg_y)
    # min_pdp_y = 0

    lines[:,:,1] = lines[:,:,1] - min_pdp_y
    pdp_curve = avg_y - min_pdp_y

    # plot predicted values for each category at each observation point
    if isinstance(list(catnames.keys())[0], bool):
        xlocs = np.arange(0, 1+1)
    else:
        xlocs = np.arange(1,ncats+1)
    # print(f"shape {lines.shape}, ncats {ncats}, nx {nx}, len(pdp) {len(pdp_curve)}")
    for i in range(nobs): # for each observation
        ax.scatter(xlocs, lines[i,:,1], # lines[i] is ith observation
                   alpha=alpha, marker='o', s=marker_size,
                   c=color)

    pdpy = pdp_curve
    if min_y_shifted_to_zero:
        avg_y = avg_y - min_pdp_y

    if pdp:
        ax.scatter(xlocs, avg_y, c=pdp_color, s=pdp_marker_size, alpha=pdp_alpha)

    if yrange is not None:
        ax.set_ylim(*yrange)
    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)

    ax.set_xticks(xlocs)

    if show_xticks: # sometimes too many
        ax.set_xticklabels(catcode2name[catcodes])
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False)

    stop = time.time()
    print(f"plot_catice {stop - start:.3f}s")