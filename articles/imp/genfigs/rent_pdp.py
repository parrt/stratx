import support
from stratx.partdep import plot_stratpd
import matplotlib.pyplot as plt

X, y = support.load_rent(n=20_000)
print(X.shape)

pdpx, pdpy, ignored = \
    plot_stratpd(X, y, colname='bedrooms', targetname='price',
                 # min_samples_leaf=15
                 )

print(pdpx)
print("ignored", ignored)
plt.show()

# plot_stratpd_gridsearch(X, y, colname='bathrooms', targetname='price',
#                         yrange=(-200,2500),
#                         pdp_marker_size=4,
#                         min_samples_leaf_values=(5,10,15,20),
#                         min_slopes_per_x_values=(5,10,15))
# plt.show()