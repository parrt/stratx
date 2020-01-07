from support import *

metric = mean_absolute_error

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

n = 10000
technique='RFSHAP'
idxs = resample(range(50_000), n_samples=n, replace=False)
X_, y_ = X.iloc[idxs], y.iloc[idxs]

I = stability(X, y, n, 10, technique=technique)
print("\nFinal")
print(I)

plot_importances(I)
plt.title(f"Bulldozer stability for {technique}")
plt.savefig(f"/Users/parrt/Desktop/bulldozer-stability-{technique}.pdf")
plt.show()