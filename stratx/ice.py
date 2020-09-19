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
from  matplotlib.collections import LineCollection
import time

"""
This code was built just to generate ICE plots for comparison in the paper.
We just hacked it together.
"""

def friedman_partial_dependences(model,X,numx=100,mean_centered=True):
    """
    Plot with stuff like:

    pdpy = friedman_partial_dependences(rf, X)
    uniq_x1 = np.unique(X['x1'])
    uniq_x2 = np.unique(X['x2'])
    fig, ax = plt.subplots(1,1)
    ax.plot(uniq_x1, pdpy[0], '.', markersize=1, label=f"x1 area={np.mean(np.abs(pdpy1))*3:2f}")
    ax.plot(uniq_x2, pdpy[1], '.', markersize=1, label=f"x2 area={np.mean(np.abs(pdpy2))*3:2f}")
    plt.legend()
    plt.show()
    """
    pdpxs = []
    pdpys = []
    for i, colname in enumerate(X.columns):
        print(colname)
        pdpx, pdpy = friedman_partial_dependence(model,X,colname,numx=numx,mean_centered=mean_centered)
        pdpxs.append(pdpx)
        pdpys.append(pdpy)
    return pdpxs, pdpys


def friedman_partial_dependence(model,X,colname,numx=100,mean_centered=True):
    """
    Return the partial dependence curve for y on X[colname] using all
    unique x values. For each unique x, replace entire X[colname] with
    it then compute average prediction. That is PDP for that x.
    """
    save_x = X[colname].copy()
    if numx is not None:
        uniq_x = np.random.choice(X[colname], numx)
    else:
        uniq_x = np.unique(X[colname])
    pdpx = uniq_x
    pdpy = np.empty(shape=(len(uniq_x),))
    for i,x in enumerate(uniq_x):
        X[colname] = x
    #     print(X)
        y_pred = model.predict(X)
        pdpy[i] = y_pred.mean()
    X[colname] = save_x
    if mean_centered:
        pdpy = pdpy - np.mean(pdpy)
    return pdpx, pdpy


def original_pdp(model, X, colname):
    """
    Return an ndarray with relative partial dependence line (average of ICE lines).
    Attempt is made to get first pdp y to 0.
    """
    ice = predict_ice(model, X, colname)
    #  Row 0 is actually the sorted unique X[colname] values used to get predictions.
    pdp_curve = np.mean(ice[1:], axis=0)
    min_pdp_y = pdp_curve[0]
    # if 0 is in x feature and not on left/right edge, get y at 0
    # and shift so that is x,y 0 point.
    linex = ice.iloc[0, :]  # get unique x values from first row
    nx = len(linex)
    if linex[int(nx * 0.05)] < 0 or linex[-int(nx * 0.05)] > 0:
        closest_x_to_0 = np.argmin(
            np.abs(np.array(linex - 0.0)))  # do argmin w/o depr warning
        min_pdp_y = pdp_curve[closest_x_to_0]

    pdp_curve -= min_pdp_y
    return pdp_curve.values


def original_catpdp(model, X, colname):
    """
    Return an ndarray with relative partial dependence line (average of ICE lines).
    Attempt is made to get first pdp y to 0.
    """
    ice = predict_catice(model, X, colname)
    #  Row 0 is actually the sorted unique X[colname] values used to get predictions.
    pdp_curve = np.mean(ice[1:], axis=0)

    return pdp_curve.values


def predict_catice(model, X:pd.DataFrame, colname:str, targetname="target", cats=None, ncats=None):
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
        lines[1:, i] = y_pred.flatten()
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


