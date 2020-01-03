from support import *

metric = mean_absolute_error
X, y = load_rent(clean_prices=False)

I = stability(X, y, 10000, 10)
print("\nFinal")
print(I)

plot_importances(I)
plt.show()