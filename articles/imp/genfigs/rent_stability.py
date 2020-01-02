from support import *

metric = mean_absolute_error
X, y = load_rent()

I = stability(X, y, 20000, 10)
print("\nFinal")
print(I)

plot_importances(I)
plt.show()