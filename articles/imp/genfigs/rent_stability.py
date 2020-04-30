import support
import stratx.featimp as featimp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(1)

# run test on ALL data in rent
X, y = support.load_rent(n=50_000)
print(X.shape)

I = featimp.importances(X, y, bootstrap=False, n_trials=30,
                        min_samples_leaf=20, # use same hyper parameters as top-k plots
                        cat_min_samples_leaf=10,
                        subsample_size=.75,
                        drop_high_stddev=2.0)

print(I)

featimp.plot_importances(I[0:10], imp_range=(0, 0.4), sortby='Importance')
plt.savefig(f"../images/rent-stability-importance.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
plt.close()

featimp.plot_importances(I[0:10], imp_range=(0, 0.4), sortby='Impact')
plt.savefig(f"../images/rent-stability-impact.pdf", bbox_inches="tight", pad_inches=0)
plt.show()