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

def bulldozer_top(top_range=(1, 9),
                  n_estimators=40,
                  trials=1,
                  n=10_000,
                  min_samples_leaf=10):
    n_shap = 300

    X, y = load_bulldozer()

    X = X.iloc[-n:]
    y = y.iloc[-n:]

    metric = mean_absolute_error # or rmse or mean_squared_error or r2_score
    #metric = r2_score
    use_oob = False

    rf = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=-1)
    rf.fit(X, y)
    print(f"Sanity check: R^2 OOB on {X.shape[0]} records: {rf.oob_score_:.3f}, training {metric.__name__}={metric(y, rf.predict(X))}")


    ols_I, shap_ols_I, rf_I, our_I = get_multiple_imps(X, y,
                                                       min_samples_leaf=min_samples_leaf,
                                                       n_estimators=n_estimators,
                                                       n_shap=n_shap,
                                                       catcolnames={'AC'})
    print("OLS\n", ols_I)
    print("OLS SHAP\n", shap_ols_I)
    print("RF SHAP\n",rf_I)
    print("OURS\n",our_I)

    features_names = ['OLS', 'OLS SHAP', 'RF SHAP', 'OUR']

    print("OUR FEATURES", our_I.index.values)

    print("n_top, n_estimators, n, n_shap, min_samples_leaf", top_range[1], n_estimators, n, n_shap, min_samples_leaf)
    topscores = []
    for top in range(top_range[0], top_range[1]+1):
        ols_top = ols_I.iloc[:top, 0].index.values
        shap_ols_top = shap_ols_I.iloc[:top, 0].index.values
        rf_top = rf_I.iloc[:top, 0].index.values
        our_top = our_I.iloc[:top, 0].index.values
        features_set = [ols_top, shap_ols_top, rf_top, our_top]
        all = []
        for i in range(trials):
            # print(i, end=' ')
            results = []
            for name, features in zip(features_names, features_set):
                # print(f"Train with {features} from {name}")
                s = avg_model_for_top_features(X[features], y, metric=metric, use_oob=use_oob)
                results.append(s)
                # print(f"{name} valid R^2 {s:.3f}")
            all.append(results)
        # print(pd.DataFrame(data=all, columns=['OLS','RF','Ours']))
        # print()
        topscores.append( [round(m,2) for m in np.mean(all, axis=0)] )

        # avg = [f"{round(m,2):9.3f}" for m in np.mean(all, axis=0)]
        # print(f"Avg top-{top} valid {metric.__name__} {', '.join(avg)}")

    A = pd.DataFrame(data=topscores, columns=features_names)
    A.index = [f"top-{top} {'OOB' if use_oob else 'training'} {metric.__name__}" for top in range(top_range[0], top_range[1]+1)]
    print(A)


bulldozer_top(min_samples_leaf=10)
