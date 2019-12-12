from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from impimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

def bulldozer_top(top_range=(1, 7),
                  n_estimators=40,
                  trials=3,
                  n=10_000,
                  min_samples_leaf=10):
    n_shap = 300

    X, y = load_bulldozer()

    X = X.iloc[-n:]
    y = y.iloc[-n:]

    rf = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"Sanity check: R^2 OOB on {X.shape[0]} records: {rf.oob_score_:.3f}")


    ols_I, rf_I, our_I = get_multiple_imps(X, y, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_shap=n_shap)
    # print("OLS\n", ols_I)
    # print("RF\n",rf_I)
    # print("OURS\n",our_I)

    top = top_range[1]

    print("OUR FEATURES", our_I.index.values)

    print("n_top, n_estimators, n, n_shap, min_samples_leaf", top, n_estimators, n, n_shap, min_samples_leaf)
    for top in range(top_range[0], top+1):
        ols_top = ols_I.iloc[:top, 0].index.values
        rf_top = rf_I.iloc[:top, 0].index.values
        our_top = our_I.iloc[:top, 0].index.values
        features_names = ['OLS', 'RF', 'OUR']
        features_set = [ols_top, rf_top, our_top]
        all = []
        for i in range(trials):
            # print(i, end=' ')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            results = []
            for name, features in zip(features_names, features_set):
                # print(f"Train with {features} from {name}")
                X_train_ = X_train[features]
                X_test_ = X_test[features]
                s = avg_model_for_top_features(name, X_test_, X_train_, y_test, y_train)
                results.append(s)
                # print(f"{name} valid R^2 {s:.3f}")
            all.append(results)
        # print(pd.DataFrame(data=all, columns=['OLS','RF','Ours']))
        # print()
        print(f"Avg top-{top} valid R^2 {np.mean(all, axis=0)}")#, stddev {np.std(all, axis=0)}")

bulldozer_top(min_samples_leaf=10)
