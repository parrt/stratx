from support import *
from stratx.featimp import *
from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300)#, threshold=1e10)

np.random.seed(3)

X, y, X_train, X_test, y_train, y_test = load_dataset("flights", "ARRIVAL_DELAY")

colname = 'ORIGIN_AIRPORT'
uniq_catcodes, combined_avg_per_cat, ignored = \
    plot_catstratpd(X, y, colname, 'ARRIVAL_DELAY',
                    min_samples_leaf=15,
                    yrange=(-125,125),
                    figsize=(14,4),
                    n_trials=1,
                    min_y_shifted_to_zero=False,
                    show_unique_cat_xticks=False,
                    show_impact=True,
                    verbose=False)

print("IGNORED", ignored)
plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/flight-fnum-cat-most_common.pdf", pad_inches=0)
plt.show()
