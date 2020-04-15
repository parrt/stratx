from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap
from sympy.simplify.radsimp import fraction_expand

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300, threshold=2000)

n = 20_000
X, y = load_bulldozer(n)

plot_stratpd(X, y, colname='YearMade', targetname='SalePrice',
             n_trials=10,
             show_slope_lines=False,
             show_impact=False,
             pdp_marker_alpha=.7,
             pdp_marker_size=4,
             figsize=(3.8,3.2)
             )
plt.xlim(1970,2010)
plt.tight_layout()
plt.savefig(f"../images/bulldozer-YearMade.pdf", bbox_inches="tight",
            pad_inches=0)
plt.show()

