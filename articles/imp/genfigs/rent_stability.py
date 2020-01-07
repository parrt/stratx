from support import *

metric = mean_absolute_error
X, y = load_rent(n=20_000, clean_prices=True)

I = stability(X, y, 10000, 10, technique='RF SHAP')
print("\nFinal")
print(I)

plot_importances(I)
plt.show()