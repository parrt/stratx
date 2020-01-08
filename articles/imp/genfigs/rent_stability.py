from support import *

metric = mean_absolute_error
X, y = load_rent(n=20_000, clean_prices=True)

technique = 'StratImpact'
I = stability(X, y, 10000, 10, technique=technique,
              catcolnames=['bathrooms'],  # numeric version ignores too much data
              min_samples_leaf=3,
              imp_n_trials=1,
              min_slopes_per_x=5,
              n_trees=5, bootstrap=True, max_features=1.0
              )
print("\nFinal")
print(I)

plot_importances(I)
plt.title(f"Rent stability for {technique}")
plt.savefig(f"/Users/parrt/Desktop/rent-stability-{technique}.pdf")
plt.show()