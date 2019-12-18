from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from impimp import *
from stratx.partdep import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rfpimp import plot_importances, dropcol_importances, importances

X, y = load_cancer_regr()

# plot_stratpd(X, y, 'worst concave points', 'cancer proba',
#              show_slope_counts=True,
#              min_slopes_per_x=0,
#              min_samples_leaf=5,
#              show_slope_lines=True)

plot_stratpd_gridsearch(X, y, 'worst concave points', 'cancer proba',
                        min_samples_leaf_values=(2, 3, 4, 5, 8, 10, 15),
                        min_slopes_per_x=0)

plt.tight_layout()
# rent_pdp()
plt.savefig("/Users/parrt/Desktop/cancer.png", pad_inches=0, dpi=150)
plt.show()