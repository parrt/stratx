import numpy as np
import pandas as pd
from typing import Mapping, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import time
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, \
    is_bool_dtype
from scipy.integrate import cumtrapz
from dtreeviz.trees import *
from stratx.partdep import *


def strat_importances(X, y,
                      ntrees=1,
                      min_samples_leaf=50,
                      hires_min_samples_leaf=50,
                      hires_r2_threshold=.3,
                      hires_n_threshold=5,
                      bootstrap=False,
                      max_features=1.0):

    def getimp(X, y, colname):
        # print(f"Unique {colname} = {len(np.unique(X[colname]))}/{len(X)}")
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features,
                                   oob_score=False)
        rf.fit(X.drop(colname, axis=1), y)
        leaf_xranges, leaf_slopes, leaf_r2 = \
            collect_leaf_slopes(rf, X, y, colname, hires_r2_threshold=hires_r2_threshold,
                                hires_min_samples_leaf=hires_min_samples_leaf,
                                hires_n_threshold=hires_n_threshold)
        uniq_x = np.array(sorted(np.unique(X[colname])))
        r2_at_x = avg_values_at_x(uniq_x, leaf_xranges, leaf_r2)
        imp = np.nanmean(r2_at_x)
        # print(f'{colname:15s} uniq_x = [{", ".join([f"{x:4.1f}" for x in uniq_x])}]')
        # print(f'{colname:15s} r2_at_x = [{", ".join([f"{s:4.2f}" for s in r2_at_x])}]')
        return imp

    #TODO: deal with r2 threshold

    if ntrees==1:
        max_features = 1.0
        bootstrap = False


    colnames = X.columns.values
    imp = []
    for colname in colnames:
        # print(colname)
        i = getimp(X, y, colname)
        imp.append(i)

    I = pd.DataFrame(data={'Feature':colnames, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I


# def plot_strat_importances(X, y,
#                            ax=None,
#                            ntrees=1,
#                            min_samples_leaf=10,
#                            hires_min_samples_leaf=5,
#                            hires_threshold=50,
#                            xrange=None,
#                            yrange=None,
#                            title=None,
#                            show_xlabel=True,
#                            show_ylabel=True,
#                            color='#fee090',
#                            bootstrap=False,
#                            max_features=1.0):
#
#     I = strat_importances(X, y, ntrees, min_samples_leaf, hires_min_samples_leaf, hires_threshold,
#                           bootstrap, max_features)
#     if ax is None:
#         fig, ax = plt.subplots(1,1)
#
#     from rfpimp import plot_importances
#     plot_importances(I, ax=ax, color=color)
    # ax.set_xlim(0, 1.0)
    # barcontainer = ax.barh(y=range(len(colnames)), width=I['Importance'],
    #                        #                                height=barcounts * unit,
    #                        tick_label=colnames,
    #                        color=color, align='center')
    #
    # # Alter appearance of each bar
    # for rect in barcontainer.patches:
    #     rect.set_linewidth(.5)
    #     rect.set_edgecolor(GREY)
