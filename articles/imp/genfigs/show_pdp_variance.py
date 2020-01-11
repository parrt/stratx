from support import *
from stratx.featimp import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 30_000

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

trials=20
colname = "YearMade"
min_samples_leaf=10
min_slopes_per_x=5
all_pdpx = []
all_pdpy = []

idxs = resample(range(50_000), n_samples=n, replace=False)
X, y = X.iloc[idxs], y.iloc[idxs]  # get sample from last part of time range

# we have a sample now
for i in range(trials):
    print(i)
    # idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
    idxs = resample(range(n), n_samples=int(n*2/3), replace=False) # subset
    X_, y_ = X.iloc[idxs], y.iloc[idxs]

    leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
        partial_dependence(X=X_, y=y_, colname=colname,
                           n_trees=1,
                           min_samples_leaf=min_samples_leaf,
                           min_slopes_per_x=min_slopes_per_x,
                           bootstrap=False,
                           max_features=1.0,
                           parallel_jit=True,)

    all_pdpx.append( pdpx )
    all_pdpy.append( pdpy )
    # I = importances(X_, y_,
    #                 catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
    #                 min_samples_leaf=8,
    #                 min_slopes_per_x=5
    #                 )

fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))
for i in range(trials):
    ax.scatter(all_pdpx[i], all_pdpy[i], s=3)

pd.DataFrame()


def avg_pd_curve(all_pdpx, all_pdpy):
    m = defaultdict(float)
    c = defaultdict(int)
    count = 0
    for i in range(trials):
        for px, py in zip(all_pdpx, all_pdpy):
            for x, y in zip(px, py):
                m[x] += y
                c[x] += 1
    for x in m.keys():
        m[x] /= c[x]
    return m

# might be diff len, do manually; how much does importance vary?
m = avg_pd_curve(all_pdpx, all_pdpy)

ax.scatter(m.keys(), m.values(), c='k', s=7)

std_imp = np.std( [np.mean(np.abs(v)) for v in all_pdpy] )
ax.set_xlabel(colname)
ax.set_ylabel('SalePrice')
ax.set_title(f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9)
plt.show()
