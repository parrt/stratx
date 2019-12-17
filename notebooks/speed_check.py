from timeit import default_timer as timer

from support import load_rent, load_bulldozer
from impimp import partial_dependence, cat_partial_dependence

n = 10000
min_samples_leaf = 5
min_slopes_per_x = n*3/1000

#X, y = load_rent(n=n)

X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]


start = timer()

# leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
#     partial_dependence(X=X, y=y, colname="latitude",
#                        min_samples_leaf=min_samples_leaf,
#                        min_slopes_per_x=min_slopes_per_x)

leaf_histos, avg_per_cat, ignored = \
    cat_partial_dependence(X, y, colname="ModelID",
                           min_samples_leaf=min_samples_leaf)

stop = timer()
print(f"Warmup Time {stop-start:.1f}s")
