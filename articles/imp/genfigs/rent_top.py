import support
import numpy as np

np.random.seed(1)

n = 25_000
X, y = support.load_rent(n=n)

support.gen_topk_figs(X, y, kfolds=5, n_trials=5, dataset="rent", title="NYC rent prices",
                      yrange=(300, 900), yunits="$")
