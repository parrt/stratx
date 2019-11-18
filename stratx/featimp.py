import numpy as np
import pandas as pd
from typing import Sequence

from sklearn.ensemble import RandomForestRegressor

from stratx.partdep import PD, discrete_xc_space

def standardize(df):
    # standardize variables
    Z = df.copy()
    for colname in df.columns:
        Z[colname] = (Z[colname] - np.mean(Z[colname])) / np.std(Z[colname])
    return Z


def importances(X:pd.DataFrame, y:pd.Series, colnames:Sequence=None,
                ntrees=1, min_samples_leaf=10, bootstrap=False,
                max_features=1.0,
                verbose=False):
    """
    """
    if colnames is None:
        colnames = X.columns.values

    #TODO: check standardized vars
    # # standardize variables
    # Z = X.copy()
    # for colname in colnames:
    #     Z[colname] = (Z[colname] - np.mean(Z[colname])) / np.std(Z[colname])

    n, p = X.shape
    df = pd.DataFrame()
    #df['x'] = sorted(X.iloc[0:-1,0])
    # pick any standardized variable (column) for shared x
    # ignore last x coordinate as we have no partial derivative data at the end
    avgs = np.zeros(shape=(p,))
    for i, colname in enumerate(colnames):
        leaf_xranges, leaf_slopes, pdpx, pdpy, ignored = \
            PD(X=X, y=y, colname=colname, ntrees=ntrees, min_samples_leaf=min_samples_leaf,
               bootstrap=bootstrap, max_features=max_features, supervised=True,
               verbose=verbose)
        x = X[colname]
        x_filtered = x[np.isin(x, pdpx)]
        uniq_x_counts = np.unique(x_filtered, return_counts=True)[1]
        # print(len(x), len(pdpx), len(x_filtered), np.sum(uniq_x_counts))
        # print(list(zip(pdpx,uniq_x_counts)))
        #y_filtered = y[np.isin(x, pdpx)]
        # print(np.sum(y_filtered), np.sum(y))
        avg_pdpy = np.mean(pdpy)
        avgs[i] = np.sum(np.abs(pdpy-avg_pdpy) * uniq_x_counts)# / np.sum(uniq_x_counts) # weighted avg abs pdpy
        # df[f"pd_{colname}"] = np.abs(pdpy)
        print("len uniq pdpx", len(np.unique(pdpx)))
        print("len y", len(y))
        print("len uniq x", len(np.unique(x)))
        print("max pdpy", np.max(pdpy))

    # TODO: probably should make smallest pd value 0 to shift all up from 0 lest
    # things cancel

    # avgs /= np.sum(avgs) # normalize to 0..1
    # print(avgs)
    # avgs /= (np.sum(y) / len(y)) # normalize to 0..1
    # avgs /= np.max(avgs)
    print("Avgs", avgs, "sum avgs", np.sum(avgs))

    # don't need unique_y_counts as summing all y gives us total sum we need
    avgs /= np.sum(np.abs(y-np.mean(y))) # normalize 0..1 where 1.0 is mass of y
    # sum(avgs) will be less than 1 if partial dep are correct
    print("normalized avgs", avgs)
    print("Mean y", np.mean(y))
    print('avg abs y', np.mean(np.abs(y)))
    print('avg abs mean-centered sum', np.sum(np.abs(y-np.mean(y))))

    # TODO maybe mean(y) should really only count x values for which we have values

    I = pd.DataFrame(data={'Feature':colnames, 'Importance':avgs})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    # print(I)

    return I