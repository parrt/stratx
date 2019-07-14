import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.collections import LineCollection
import time

"""
This code was built just to generate ICE plots for comparison in the paper.
We just hacked it together.
"""

from stratx.partdep import getcats

def predict_catice(model, X:pd.DataFrame, colname:str, targetname, cats=None, ncats=None):
    if cats is None:
        cats = np.unique(X[colname]) # get unique codes
    return predict_ice(model=model, X=X, colname=colname, targetname=targetname,
                       cats=cats, nlines=ncats)


def predict_ice(model, X:pd.DataFrame, colname:str, targetname="target", cats=None, numx=50, nlines=None):
    """
    Return dataframe with one row per observation in X and one column
    per unique value of column identified by colname.
    Row 0 is actually the sorted unique X[colname] values used to get predictions.
    It's handy to have so we don't have to pass X around to other methods.
    Points in a single ICE line are the unique values of colname zipped
    with one row of returned dataframe. E.g.,

    	predicted weight          predicted weight         ...
    	height=62.3638789416112	  height=62.78667197542318 ...
    0	62.786672	              70.595222                ... unique X[colname] values
    1	109.270644	              161.270843               ...
    """
    start = time.time()
    save = X[colname].copy()

    if nlines is not None and nlines > len(X):
        nlines = len(X)

    if cats is not None:
        linex = np.unique(cats)
        numx = None
    elif numx is not None:
        linex = np.linspace(np.min(X[colname]), np.max(X[colname]), numx, endpoint=True)
    else:
        linex = sorted(X[colname].unique())

    lines = np.zeros(shape=(len(X) + 1, len(linex)))
    lines[0, :] = linex
    i = 0
    for v in linex:
        X[colname] = v
        y_pred = model.predict(X)
        lines[1:, i] = y_pred
        i += 1
    X[colname] = save
    columns = [f"predicted {targetname}\n{colname}={str(v)}"
               for v in linex]
    df = pd.DataFrame(lines, columns=columns)

    if nlines is not None:
        # sample lines (first row is special: linex)
        df_ = pd.DataFrame(lines)
        df_ = df_.sample(n=nlines, axis=0, replace=False)
        lines = df_.values
        lines = np.concatenate([linex.reshape(1,-1),lines], axis=0)
        df = pd.DataFrame(lines, columns=columns)

    stop = time.time()
    print(f"ICE_predict {stop - start:.3f}s")
    return df


def ice2lines(ice:np.ndarray) -> np.ndarray:
    """
    Return a 3D array of 2D matrices holding X coordinates in col 0 and
    Y coordinates in col 1. result[0] is first 2D matrix of [X,Y] points
    in a single ICE line for single observations. Shape of result is:
    (nobservations,nuniquevalues,2)
    """
    start = time.time()
    linex = ice.iloc[0,:] # get unique x values from first row
    # If needed, apply_along_axis() is faster than the loop
    # def getline(liney): return np.array(list(zip(linex, liney)))
    # lines = np.apply_along_axis(getline, axis=1, arr=ice.iloc[1:])
    lines = []
    for i in range(1,len(ice)): # ignore first row
        liney = ice.iloc[i].values
        line = np.array(list(zip(linex, liney)))
        lines.append(line)
    stop = time.time()
    # print(f"ICE_lines {stop - start:.3f}s")
    return np.array(lines)


def plot_ice(ice, colname, targetname="target", ax=None, linewidth=.5, linecolor='#9CD1E3',
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
        closest_x_to_0 = np.abs(linex - 0.0).argmin()
        min_pdp_y = avg_y[closest_x_to_0]

    lines = ice2lines(ice)
    lines[:,:,1] = lines[:,:,1] - min_pdp_y
    # lines[:,:,0] scans all lines, all points in a line, and gets x column
    minx, maxx = np.min(lines[:,:,0]), np.max(lines[:,:,0])
    miny, maxy = np.min(lines[:,:,1]), np.max(lines[:,:,1])
    if yrange is not None:
        ax.set_ylim(*yrange)
    else:
        ax.set_ylim(miny, maxy)
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
    else:
        ax.set_xlim(minx, maxx)

    uniq_x = ice.iloc[0, :]
    pdp_curve = avg_y - min_pdp_y
    if pdp:
        ax.plot(uniq_x, pdp_curve,
                alpha=pdp_alpha, linewidth=pdp_linewidth, c=pdp_color)

    stop = time.time()
    # print(f"plot_ICE {stop - start:.3f}s")
    return uniq_x, pdp_curve

def plot_catice(ice, colname, targetname,
                catnames,  # cat names indexed by cat code
                ax=None,
                color='#9CD1E3',
                alpha=.1, title=None, yrange=None, pdp=True,
                pdp_marker_size=.5, pdp_alpha=1,
                pdp_color='black',
                marker_size=10,
                show_xlabel=True, show_ylabel=True,
                show_xticks=True,
                sort='ascending'):
    start = time.time()
    if ax is None:
        fig, ax = plt.subplots(1,1)

    ncats = len(catnames)

    avg_y = np.mean(ice[1:], axis=0)

    lines = ice2lines(ice)

    nobs = lines.shape[0]
    nx = lines.shape[1]

    catcodes, _, catcode2name = getcats(None, colname, catnames)
    sorted_catcodes = catcodes
    if sort == 'ascending':
        sorted_indexes = avg_y.argsort()
        sorted_catcodes = catcodes[sorted_indexes]
    elif sort == 'descending':
        sorted_indexes = avg_y.argsort()[::-1] # reversed
        sorted_catcodes = catcodes[sorted_indexes]

    # find leftmost value (lowest value if sorted ascending) and shift by this
    min_pdp_y = avg_y[sorted_indexes[0]]
    lines[:,:,1] = lines[:,:,1] - min_pdp_y
    pdp_curve = avg_y - min_pdp_y

    # plot predicted values for each category at each observation point
    if True in catnames or False in catnames:
        xlocs = np.arange(0, ncats)
    else:
        xlocs = np.arange(1,ncats+1)
    # print(f"shape {lines.shape}, ncats {ncats}, nx {nx}, len(pdp) {len(pdp_curve)}")
    for i in range(nobs): # for each observation
        ax.scatter(xlocs, lines[i,sorted_indexes,1], # lines[i] is ith observation
                   alpha=alpha, marker='o', s=marker_size,
                   c=color)

    if pdp:
        ax.scatter(xlocs, pdp_curve[sorted_indexes], c=pdp_color, s=pdp_marker_size, alpha=pdp_alpha)

    if yrange is not None:
        ax.set_ylim(*yrange)
    if show_xlabel:
        ax.set_xlabel(colname)
    if show_ylabel:
        ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)

    if True in catnames or False in catnames:
        ax.set_xticks(range(0, 1+1))
    else:
        ax.set_xticks(range(1, ncats+1))

    if show_xticks: # sometimes too many
        ax.set_xticklabels(catcode2name[sorted_catcodes])
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='both', bottom=False)

    stop = time.time()
    print(f"plot_catice {stop - start:.3f}s")