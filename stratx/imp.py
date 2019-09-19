import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

def importances(X, y, colname, targetname,
                ntrees=1, min_samples_leaf=10, bootstrap=False,
                max_features=1.0,
                verbose=False):
    """
    """
    rf = RandomForestRegressor(n_estimators=ntrees,
                               min_samples_leaf=min_samples_leaf,
                               bootstrap=bootstrap,
                               max_features=max_features)
    rf.fit(X.drop(colname, axis=1), y)
    if verbose:
        print(
            f"Strat Partition RF: missing {colname} training R^2 {rf.score(X.drop(colname, axis=1), y)}")


