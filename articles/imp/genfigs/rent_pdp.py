import support
from stratx import plot_stratpd, plot_catstratpd, \
    plot_catstratpd_gridsearch, plot_stratpd_gridsearch
import matplotlib.pyplot as plt
from stratx.featimp import importances
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#np.random.seed(5)

X, y, X_train, X_test, y_train, y_test = support.load_dataset("rent", 'price')

# X, y = X[:100], y[:100]

# X, y = support.load_rent(n=15_000)
# print(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# tuned_params = support.models[("rent", "RF")]
# rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
# rf.fit(X_train, y_train)
# print("R^2 test",rf.score(X_test,y_test))

I = importances(X, y,
                n_trials=1,
                normalize=False,

                bootstrap=True,
                # bootstrap=False,
                # subsample_size=.7,

                min_samples_leaf=20,
                cat_min_samples_leaf=20,
                )
print(I)
exit()
#
# pdpx, pdpy, ignored = \
#     plot_stratpd(X, y, colname='longitude', targetname='price',
#                 #yrange=(-500,3500),
#                  n_trials = 1,
#                  pdp_marker_size=3,
#                  min_samples_leaf=20,
#                  min_slopes_per_x=20,
#                  show_impact=True
#                  )
# plt.show()
# print("pdpy", list(pdpy))

# print(pdpx)
# print("ignored", ignored)
# plt.show()

# plot_catstratpd(X, y, colname='bedrooms', targetname='price',
#                 # yrange=(-500,3500),
#                 min_y_shifted_to_zero=True)
#

#np.random.seed(1)

# uniq_catcodes, combined_avg_per_cat, ignored, merge_ignored = \
#     plot_catstratpd(X, y, colname='bedrooms', targetname='price',
#                     n_trials=1,
#                     yrange=(-500,6000),
#                     min_y_shifted_to_zero=True,
#                     # show_avg_pairwise_effect=True,
#                     min_samples_leaf=10,
#                     show_impact=True
#                     )
# plt.show()

# m = np.nanmean(combined_avg_per_cat)
# i = np.nanmean(np.abs(combined_avg_per_cat - m))
# # print("impact from avg", i)
# print("mean cat y", m)
# print("size", len(X), "ignored", ignored, merge_ignored)
# plot_catstratpd_gridsearch(X, y, colname='bathrooms', targetname='price',
#                            n_trials=1,
#                         yrange=(-200,5000),
#                         min_samples_leaf_values=(2, 5, 10,15,20))
# plt.show()