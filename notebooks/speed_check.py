from timeit import default_timer as timer

from support import load_rent, load_bulldozer
from impimp import *
from stratx.partdep import *

np.random.seed(999)

n = 50_000
min_samples_leaf = 5
min_slopes_per_x = n*3/1000

X, y = load_rent(n=n)

# X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]


for i in range(3):
    start = timer()

    I = impact_importances(X, y)
    # leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
    #     partial_dependence(X=X, y=y, colname="Wvillage",
    #                        min_samples_leaf=min_samples_leaf,
    #                        min_slopes_per_x=min_slopes_per_x)

    stop = timer()
    print(f"Time {i+1}: {stop-start:.1f}s")


plot_stratpd(X, y, "latitude", "price")
plt.show()

# catcodes, _, catcode2name = getcats(X, "ModelID", None)
#
# leaf_histos, avg_per_cat, ignored = \
#     cat_partial_dependence(X, y, colname="ModelID",
#                            index=catcode2name,
#                            min_samples_leaf=min_samples_leaf)

