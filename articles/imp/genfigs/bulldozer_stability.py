from support import *

from stratx import *

metric = mean_absolute_error

X, y = load_bulldozer(n=50_000)

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

n = 10000
technique='RFSHAP'
technique='StratImpact'
idxs = resample(range(50_000), n_samples=n, replace=False)
X_, y_ = X.iloc[idxs], y.iloc[idxs]

I = importances(X, y,
                catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                n_trials = 10
                )
# I = stability(X, y, n, 10, technique=technique,
#               catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
#               min_samples_leaf=10,
#               min_slopes_per_x=5,
#               imp_n_trials=1
#               )
print("\nFinal")
print(I)

plot_importances(I)
# plt.title(f"Bulldozer stability for {technique}")
plt.savefig(f"/Users/parrt/Desktop/bulldozer-stability.pdf")
plt.show()