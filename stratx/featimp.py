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
    df['x'] = sorted(X.iloc[0:-1,0])
    # pick any standardized variable (column) for shared x
    # ignore last x coordinate as we have no partial derivative data at the end
    for colname in colnames:
        leaf_xranges, leaf_slopes, pdpx, pdpy, ignored = \
            PD(X=X, y=y, colname=colname, ntrees=ntrees, min_samples_leaf=min_samples_leaf,
               bootstrap=bootstrap, max_features=max_features, supervised=True,
               verbose=verbose)
        df[f"pd_{colname}"] = np.abs(pdpy)

    # TODO: probably should make smallest pd value 0 to shift all up from 0 lest
    # things cancel


    df['sum_pd'] = df.iloc[:,1:].sum(axis=1)

    # do ratios for importance
    # for colname in X.columns:
    #     df[f'I_{colname}'] = df[f'pd_{colname}'] / df[f'sum_pd']


    print(df)
    avgs = [np.mean(df[f"pd_{colname}"]) for colname in colnames]
    avgs /= np.sum(avgs) # normalize to 0..1

    I = pd.DataFrame(data={'Feature':colnames, 'Importance':avgs})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    print(I)

    return I