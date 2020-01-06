from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
from sklearn.utils import resample

import shap

from stratx.featimp import *
from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 20_000

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

idxs = resample(range(50_000), n_samples=n, replace=False)
X_, y_ = X.iloc[idxs], y.iloc[idxs]

figsize=(3.1, 2.3)

# for display purposes, drop machine hours beyond 30 years
fig, ax = plt.subplots(1,1,figsize=figsize)
plot_stratpd(X_, y_, colname='age', targetname='SalePrice',
             show_slope_lines=False,
             show_impact=True,
             ticklabel_fontsize=10,
             xrange=(0,30),
             yrange=(-8000,1000),
             ax=ax
             )
plt.tight_layout()
plt.savefig("../images/bulldozer-impact-age.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

# for display purposes, drop machine hours beyond 70,000
fig, ax = plt.subplots(1,1,figsize=figsize)
plot_stratpd(X_, y_, colname='MachineHours', targetname='SalePrice',
             show_slope_lines=False,
             show_impact=True,
             ticklabel_fontsize=10,
             xrange=(0,70_000),
             ax=ax
             )
plt.tight_layout()
plt.savefig("../images/bulldozer-impact-MachineHours.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

fig, ax = plt.subplots(1,1,figsize=figsize)
plot_stratpd(X_, y_, colname='saledayofyear', targetname='SalePrice',
             show_slope_lines=False,
             show_impact=True,
             ticklabel_fontsize=10,
             ax=ax
             )
plt.tight_layout()
plt.savefig("../images/bulldozer-impact-saledayofyear.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

fig, ax = plt.subplots(1,1,figsize=figsize)
plot_stratpd(X_, y_, colname='YearMade', targetname='SalePrice',
             show_slope_lines=False,
             show_impact=True,
             ticklabel_fontsize=10,
             ax=ax
             )
plt.tight_layout()
plt.savefig("../images/bulldozer-impact-YearMade.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
