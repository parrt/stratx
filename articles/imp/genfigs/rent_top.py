import support
import numpy as np

np.random.seed(1)

support.gen_topk_figs(n_trials=1, dataset="rent",
                      targetname='price',
                      title="NYC rent prices",
                      yrange=(100, 900),
                      yunits="$")
