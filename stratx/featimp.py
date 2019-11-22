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

    n, p = X.shape
    avgs = np.zeros(shape=(p,))
    for i, colname in enumerate(colnames):
        #plot_catstratpd
        leaf_xranges, leaf_slopes, pdpx, pdpy, ignored = \
            PD(X=X, y=y, colname=colname, ntrees=ntrees, min_samples_leaf=min_samples_leaf,
               bootstrap=bootstrap, max_features=max_features, supervised=True,
               verbose=verbose)
        x = X[colname]
        x_filtered = x[np.isin(x, pdpx)]
        uniq_x_counts = np.unique(x_filtered, return_counts=True)[1]
        # x_mass = np.sum(np.abs(pdpy) * uniq_x_counts) / np.sum(uniq_x_counts)
        # weighted average was way over-counting for repeated x values
        # maybe we want to know how much each x pushes up y, but don't count
        # repeated values at x.
        # x_mass = np.mean(pdpy - np.min(pdpy))
        x_mass = np.mean(np.abs(pdpy))
        print(f"{colname} mass", x_mass)
        avgs[i] = x_mass
        print("min,max pdpy", np.min(pdpy), np.max(pdpy))

    pdpy_mass = np.sum(avgs)
    print("Avgs", avgs, "sum avgs", pdpy_mass)

    # What is mass of y after clipping so min(y) is 0?
    y_mass = np.mean(np.abs(y-np.min(y)))

    print("### remaining mass %", 100 * (y_mass - pdpy_mass) / y_mass)

    # Normalize x_i masses to be ratios of y_mass
    avgs /= np.sum(y_mass)
    print("normalized avgs", avgs, 'ratios', avgs[0]/avgs[1])
    # don't need unique_y_counts as summing all y gives us total sum we need
    #avgs /= np.sum(np.abs(y-np.mean(y))) # normalize 0..1 where 1.0 is mass of y
    # sum(avgs) will be less than 1 if partial dep are correct
    print('mean abs y', np.mean(np.abs(y)))
    print('y_mass = mean abs min-clipped', y_mass)
    print('mean abs mean-centered', np.mean(np.abs(y-np.mean(y))))

    # TODO maybe mean(y) should really only count x values for which we have values

    I = pd.DataFrame(data={'Feature':colnames, 'Importance':avgs})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    # print(I)

    return I, y_mass, pdpy_mass