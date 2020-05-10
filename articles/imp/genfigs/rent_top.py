import support
import numpy as np

np.random.seed(1)

support.gen_topk_figs(n_trials=1, dataset="rent",
                      targetname='price',
                      min_samples_leaf=15,
                      cat_min_samples_leaf=10,  # unused; only numerical
                      title="NYC rent prices",
                      yrange=(200, 1100),
                      normalize=False,
                      yunits="$")
