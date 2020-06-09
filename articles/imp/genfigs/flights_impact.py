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

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 20_000

X, y, _ = load_flights(n=n)

fig, ax = plt.subplots(1,1,figsize=(3.5, 2.7))
plot_stratpd(X, y, colname='SCHEDULED_DEPARTURE', targetname='Flight arrival delay (min)',
             show_slope_lines=False,
             show_impact=True,
             ticklabel_fontsize=10,
             pdp_marker_size=1,
             show_impact_dots=False,
             show_impact_line=False,
             ax=ax
             )
plt.tight_layout()
plt.savefig("../images/flights-impact-SCHEDULED_DEPARTURE.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

fig, ax = plt.subplots(1,1,figsize=(3.5, 2.7))
plot_stratpd(X, y, colname='DEPARTURE_TIME', targetname='Flight arrival delay (min)',
             show_slope_lines=False,
             show_impact=True,
             ticklabel_fontsize=10,
             pdp_marker_size=1,
             show_impact_dots=False,
             show_impact_line=False,
             ax=ax
             )
plt.tight_layout()
plt.savefig("../images/flights-impact-DEPARTURE_TIME.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
