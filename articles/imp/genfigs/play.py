from support import *
from stratx.featimp import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25_000

X, y = load_rent(n=n)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

lm = LinearRegression()#normalize=True)
lm.fit(X_train,y_train)

OLS_I,_ = linear_model_importance(lm, X_train, y_train)
print("OLS I\n",OLS_I)

I = importances(X_train, y_train, min_slopes_per_x = 15)
print("OURS\n",I)

kf = KFold(n_splits=5)
kfold_indexes = list(kf.split(X))
for i in range(len(X.columns)):
    columns = X.columns[0:i+1]
    scores = cv_features(kfold_indexes, X, y, columns,
                         metric=mean_absolute_error,
                         model="OLS")
    print(columns, scores, np.mean(scores))

print(mean_absolute_error(y_test, lm.predict(X_test)))



# X, y, _ = load_flights(n=n)
# print(X.shape)

#
# I = importances(X, y, n_trials = 1,
#                 cat_min_samples_leaf=5,
#                 min_slopes_per_x=15,
#                 catcolnames={'AIRLINE',
#                              'ORIGIN_AIRPORT',
#                              'DESTINATION_AIRPORT',
#                              'FLIGHT_NUMBER',
#                              'DAY_OF_WEEK'}
#                 )
# print(I)
# plot_importances(I, imp_range=(0,.4), width=4)
# plt.show()
#
# colname='bedrooms'
# colname='bathrooms'
# colname='astoria'
# colname='UpperEast'

# plot_stratpd(X, y, colname=colname, targetname='price',
#              min_slopes_per_x=15,
#              n_trials=5,
#              show_slope_lines=False,
#              show_impact=False,
#              figsize=(3.8,3.2)
#              )
# plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/rent-{colname}.pdf", pad_inches=0)
# plt.show()

# X, y = load_bulldozer()
# print(X.shape)

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
# X = X.iloc[-50_000:]
# y = y.iloc[-50_000:]
#
# I = importances(X, y, n_trials = 1,
#                 cat_min_samples_leaf=5,
#                 min_slopes_per_x=15,
#                 catcolnames={'AC',
#                              'ModelID'}
#                 )
# print(I)
# plot_importances(I, imp_range=(0,.4), width=4)
# plt.show()

# trials=20
# colname = "YearMade"
# min_samples_leaf=10
# min_slopes_per_x=5
#
# idxs = resample(range(50_000), n_samples=n, replace=False)
# X, y = X.iloc[idxs], y.iloc[idxs]  # get sample from last part of time range
#
# # we have a sample now
# for i in range(trials):
#     print(i)
#     # idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
#     idxs = resample(range(n), n_samples=int(n*2/3), replace=False) # subset
#     X_, y_ = X.iloc[idxs], y.iloc[idxs]
#
#     I = importances(X_, y_,
#                     catcolnames={'AC', 'ModelID'},
#                     n_trials=5,
#                     min_samples_leaf=5,
#                     # min_slopes_per_x=5
#                     )
#     print(I[0:10])
