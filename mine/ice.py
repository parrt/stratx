import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  matplotlib.collections import LineCollection
import time

def ice_predict(model, X:pd.DataFrame, colname:str, targetname="target", numx=50, nlines=None):
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
    if nlines is not None:
        X = X.sample(nlines, replace=False)
    if numx is not None:
        linex = np.linspace(np.min(X[colname]), np.max(X[colname]), numx)
    else:
        linex = sorted(X[colname].unique())
    lines = np.zeros(shape=(len(X) + 1, len(linex)))
    lines[0, :] = linex
    i = 0
    for v in linex:
        #         print(f"{colname}.{v}")
        X[colname] = v
        y_pred = model.predict(X)
        lines[1:, i] = y_pred
        i += 1
    X[colname] = save
    columns = [f"predicted {targetname}\n{colname}={str(v)}"
               for v in linex]
    df = pd.DataFrame(lines, columns=columns)
    stop = time.time()
    print(f"ICE_predict {stop - start:.3f}s")
    return df


def ice2lines(ice:np.ndarray) -> np.ndarray:
    """
    Return a 3D array of 2D matrices holding X coordinates in col 0 and
    Y coordinates in col 1. result[0] is first 2D matrix of [X,Y] points
    in a single ICE line for single sample. Shape of result is:
    (nsamples,nuniquevalues,2)
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


def ice_plot(ice, colname, targetname="target", cats=None, ax=None, linewidth=.7, color='#9CD1E3',
             alpha=.1, title=None, yrange=None, pdp=True, pdp_linewidth=1, pdp_alpha=1,
             pdp_color='black'):
    start = time.time()
    if ax is None:
        fig, ax = plt.subplots(1,1)

    avg_y = np.mean(ice[1:], axis=0)

    min_pdp_y = avg_y[0] if cats is None else 0
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
    ax.set_xlabel(colname)
    ax.set_ylabel(targetname)
    if title is not None:
        ax.set_title(title)
    lines = LineCollection(lines, linewidth=linewidth, alpha=alpha, color=color)
    ax.add_collection(lines)

    if cats is not None:
        if True in cats or False in cats:
            ax.set_xticks(range(0, 1+1))
            ax.set_xticklabels(cats)
            ax.set_xlim(0, 1)
        else:
            ncats = len(cats)
            ax.set_xticks(range(1, ncats+1))
            ax.set_xticklabels(cats)
            ax.set_xlim(1, ncats)
    else:
        ax.set_xlim(minx, maxx)

    if pdp:
        uniq_values = ice.iloc[0,:]
        ax.plot(uniq_values, avg_y - min_pdp_y,
                alpha=pdp_alpha, linewidth=pdp_linewidth, c=pdp_color)

    stop = time.time()
    # print(f"plot_ICE {stop - start:.3f}s")
