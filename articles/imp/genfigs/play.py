from support import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 20_000

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

trials=50

m = 0
a = 0
for i in range(trials):
    idxs = resample(range(50_000), n_samples=n, replace=False)
    X_, y_ = X.iloc[idxs], y.iloc[idxs]

    I = importances(X_, y_,
                    catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                    min_samples_leaf=8,
                    min_slopes_per_x=5
                    )
    I = I.reset_index()
    print(I.iloc[0:2])
    if I.iloc[0]['Feature']=='ModelID':
        m += 1
    if I.iloc[0]['Feature']=='age':
        a += 1

print(f"{trials} = {m}/{m+a} ModelID and {a}/{m+a} age")
