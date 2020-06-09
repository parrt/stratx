from support import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

GREY = '#444443'

shap_test_size = 1000
n = 20_000
X, y = load_bulldozer(n)

fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
ax.scatter(X['YearMade'], y, s=3, alpha=.1, c='#1E88E5')
ax.set_xlim(1970,2010)
ax.set_xlabel("YearMade\n(a)", fontsize=11)
ax.set_ylabel("SalePrice ($)", fontsize=11)
ax.set_title("Marginal plot", fontsize=13)
ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
plt.tight_layout()
plt.savefig("../images/bulldozer-YearMade-marginal.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
#
tuned_params = models[("bulldozer", "RF")]
rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
rf.fit(X, y)

explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                               feature_perturbation='interventional')
shap_sample = X.sample(n=shap_test_size)
shap_values = explainer.shap_values(shap_sample,
                                    check_additivity=False)

fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.2))
shap.dependence_plot("YearMade", shap_values, shap_sample,
                     interaction_index=None, ax=ax, dot_size=5,
                     show=False, alpha=.5)

ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

ax.set_title("SHAP", fontsize=13)
ax.set_ylabel("Impact on SalePrice\n(YearMade SHAP)", fontsize=11)
ax.set_xlabel("YearMade\n(b)", fontsize=11)
ax.set_xlim(1970,2010)
ax.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.savefig("../images/bulldozer-YearMade-shap.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))
plot_stratpd(X, y, colname='YearMade', targetname='SalePrice',
             n_trials=10,
             show_slope_lines=False,
             show_x_counts=False,
             show_xlabel=False,
             show_impact=False,
             pdp_marker_alpha=.7,
             pdp_marker_size=4,
             ax=ax
             )
ax.set_title("StratPD", fontsize=13)
ax.set_xlabel("YearMade\n(c)", fontsize=11)
ax.set_xlim(1970,2010)
plt.tight_layout()
plt.savefig(f"../images/bulldozer-YearMade-stratpd.pdf", bbox_inches="tight",
            pad_inches=0)
plt.show()
