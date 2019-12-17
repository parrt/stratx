from timeit import default_timer as timer

from support import load_rent
from impimp import partial_dependence

n = 60000
min_samples_leaf = 5
min_slopes_per_x = n*3.5/1000

X, y = load_rent(n=n)

start = timer()

leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
    partial_dependence(X=X, y=y, colname="latitude",
                       min_samples_leaf=min_samples_leaf,
                       min_slopes_per_x=min_slopes_per_x)

stop = timer()
print(f"Warmup Time {stop-start:.1f}s")


start = timer()

leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
    partial_dependence(X=X, y=y, colname="latitude",
                       min_samples_leaf=min_samples_leaf,
                       min_slopes_per_x=min_slopes_per_x)

stop = timer()
print(f"Time {stop-start:.1f}s")

start = timer()

leaf_xranges, leaf_slopes, slope_counts_at_x, dx, dydx, pdpx, pdpy, ignored = \
    partial_dependence(X=X, y=y, colname="latitude",
                       min_samples_leaf=min_samples_leaf,
                       min_slopes_per_x=min_slopes_per_x)

stop = timer()
print(f"Time {stop-start:.1f}s")
