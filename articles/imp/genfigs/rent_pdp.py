from support import *

X, y = load_rent(n=20_000)
print(X.shape)

# pdpx, pdpy, ignored = \
#     plot_stratpd(X, y, colname='bedrooms', targetname='price')

# print(pdpx)
# print("ignored", ignored)

plot_stratpd_gridsearch(X, y, colname='bathrooms', targetname='price',
                        yrange=(-200,2500),
                        pdp_marker_size=4,
                        min_samples_leaf_values=(5,10,15,20),
                        min_slopes_per_x_values=(5,10,15))
plt.show()