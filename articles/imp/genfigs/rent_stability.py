from support import *

X, y = load_rent(n=20_000)
print(X.shape)

I = importances(X, y, n_trials=10, min_slopes_per_x=15)

print(I)

plot_importances(I[0:8], imp_range=(0, 0.4), sortby='Importance')
plt.savefig(f"../images/rent-stability-importance.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()

plot_importances(I[0:8], imp_range=(0, 0.4), sortby='Impact')
plt.savefig(f"../images/rent-stability-impact.pdf", bbox_inches="tight", pad_inches=0)
plt.show()