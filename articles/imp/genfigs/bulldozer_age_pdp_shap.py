from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

GREY = '#444443'

n = 20_000
shap_test_size = 1000

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

idxs = resample(range(50_000), n_samples=n, replace=False)
X_, y_ = X.iloc[idxs], y.iloc[idxs]

fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
ax.scatter(X_['age'], y_, s=3, alpha=.1, c='#1E88E5')
ax.set_xlabel("age\n(a)", fontsize=13)
ax.set_ylabel("SalePrice ($)", fontsize=13)
ax.set_title("Marginal plot", fontsize=13)
ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
plt.tight_layout()
plt.savefig("../images/bulldozer-age-marginal.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

rf = RandomForestRegressor(n_estimators=40, n_jobs=-1)
rf.fit(X_, y_)

explainer = shap.TreeExplainer(rf, data=shap.sample(X_, 100),
                               feature_perturbation='interventional')
shap_values = explainer.shap_values(X.sample(n=shap_test_size),
                                    check_additivity=False)

fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
shap.dependence_plot("age", shap_values, X_.sample(n=shap_test_size),
                     interaction_index=None, ax=ax, dot_size=5,
                     show=False, alpha=.5)

ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

ax.set_ylabel("Impact on SalePrice\n(age SHAP)")
ax.set_xlabel("age\n(b)")
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.savefig("../images/bulldozer-age-shap.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

# Now show StratPD

fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
plot_stratpd(X_, y_, colname='age', targetname='SalePrice',
             show_slope_lines=False,
             show_impact=False,
             label_fontsize=13,
             figsize=(3.8,3.2),
             ax=ax
             )

ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

ax.set_ylabel("SalePrice ($)", fontsize=13)
ax.set_xlabel("age\n(c)", fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig("../images/bulldozer-age-stratpd.pdf", bbox_inches="tight", pad_inches=0)
plt.show()