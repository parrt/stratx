import shap
from stratx.ice import friedman_partial_dependence
from stratx.partdep import plot_stratpd
from articles.pd.support import synthetic_interaction_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor

np.random.seed(1)  # pick seed for reproducible article images

# reuse exact same data used by ALE
# n = 1000
# df = synthetic_interaction_data(n)

df = pd.read_csv("interaction.csv")

X, y = df[['x1', 'x2', 'x3']].copy(), df['y'].copy()
X1 = X.iloc[:, 0]
X2 = X.iloc[:, 1]
X3 = X.iloc[:, 2] # UNUSED in y

rf = RandomForestRegressor(n_estimators=30, oob_score=True)
rf.fit(X, y)
print("R^2 training", rf.score(X, y))
print("R^2 OOB", rf.oob_score_)

print("mean(y) =", np.mean(y))
print("mean(X_1), mean(X_2) =", np.mean(X1), np.mean(X2))

pdp_x1 = friedman_partial_dependence(rf, X, 'x1', numx=None, mean_centered=False)
pdp_x2 = friedman_partial_dependence(rf, X, 'x2', numx=None, mean_centered=False)
pdp_x3 = friedman_partial_dependence(rf, X, 'x3', numx=None, mean_centered=False)
m1 = np.mean(pdp_x1[1])
m2 = np.mean(pdp_x2[1])
m3 = np.mean(pdp_x3[1])
print("mean(PDP_1) =", np.mean(pdp_x1[1]))
print("mean(PDP_2) =", np.mean(pdp_x2[1]))
print("mean(PDP_2) =", np.mean(pdp_x3[1]))

print("mean abs PDP_1-ybar", np.mean(np.abs(pdp_x1[1] - m1)))
print("mean abs PDP_2-ybar", np.mean(np.abs(pdp_x2[1] - m2)))
print("mean abs PDP_3-ybar", np.mean(np.abs(pdp_x3[1] - m3)))

explainer = shap.TreeExplainer(rf, data=X,
                               feature_perturbation='interventional')
shap_values = explainer.shap_values(X, check_additivity=False)
shapavg = np.mean(shap_values, axis=0)
print("SHAP avg x1,x2,x3 =", shapavg)
shapimp = np.mean(np.abs(shap_values), axis=0)
print("SHAP avg |x1|,|x2|,|x3| =", shapimp)

fig, axes = plt.subplots(1,2,figsize=(5.5,2.8))

x1_color = '#1E88E5'
x2_color = 'orange'
x3_color = '#A22396'

axes[0].plot(pdp_x1[0], pdp_x1[1], '.', markersize=1, c=x1_color, label='$FPD_1$', alpha=1)
axes[0].plot(pdp_x2[0], pdp_x2[1], '.', markersize=1, c=x2_color, label='$FPD_2$', alpha=1)
axes[0].plot(pdp_x3[0], pdp_x3[1], '.', markersize=1, c=x3_color, label='$FPD_3$', alpha=1)

axes[0].text(0, 75, f"$\\bar{{y}}={np.mean(y):.1f}$", fontsize=13)
axes[0].set_xticks([0,2,4,6,8,10])
axes[0].set_xlabel("$x_1, x_2, x_3$", fontsize=10)
axes[0].set_ylabel("y")
axes[0].set_yticks([0, 25, 50, 75, 100, 125, 150])
axes[0].set_ylim(-10,160)
axes[0].set_title(f"(a) Friedman FPD")

axes[0].spines['top'].set_linewidth(.5)
axes[0].spines['right'].set_linewidth(.5)
axes[0].spines['left'].set_linewidth(.5)
axes[0].spines['bottom'].set_linewidth(.5)
axes[0].spines['top'].set_color('none')
axes[0].spines['right'].set_color('none')

x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
axes[0].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=10)

# axes[0].legend(fontsize=10)

#axes[1].plot(shap_values)
shap.dependence_plot("x1", shap_values, X,
                     interaction_index=None, ax=axes[1], dot_size=4,
                     show=False, alpha=.5, color=x1_color)
shap.dependence_plot("x2", shap_values, X,
                     interaction_index=None, ax=axes[1], dot_size=4,
                     show=False, alpha=.5, color=x2_color)
shap.dependence_plot("x3", shap_values, X,
                     interaction_index=None, ax=axes[1], dot_size=4,
                     show=False, alpha=.5, color=x3_color)
axes[1].set_xticks([0,2,4,6,8,10])
axes[1].set_xlabel("$x_1, x_2, x_3$", fontsize=12)
axes[1].set_ylim(-95,110)
axes[1].set_title("(b) SHAP")
axes[1].set_ylabel("SHAP values", fontsize=11)
x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
axes[1].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=12)

if False:
    df_x1 = pd.read_csv("../images/x1_ale.csv")
    df_x2 = pd.read_csv("../images/x2_ale.csv")
    df_x3 = pd.read_csv("../images/x3_ale.csv")
    axes[2].plot(df_x1['x.values'],df_x1['f.values'],'.',color=x1_color,markersize=2)
    axes[2].plot(df_x2['x.values'],df_x2['f.values'],'.',color=x2_color,markersize=2)
    axes[2].plot(df_x3['x.values'],df_x3['f.values'],'.',color=x3_color,markersize=2)
    axes[2].set_title("(c) ALE")
    # axes[2].set_ylabel("y", fontsize=12)
    axes[2].set_xlabel("$x_1, x_2, x_3$", fontsize=12)
    axes[2].set_ylim(-95,110)
    # axes[2].tick_params(axis='both', which='major', labelsize=10)
    axes[2].set_xticks([0,2,4,6,8,10])
    axes[2].spines['top'].set_linewidth(.5)
    axes[2].spines['right'].set_linewidth(.5)
    axes[2].spines['left'].set_linewidth(.5)
    axes[2].spines['bottom'].set_linewidth(.5)
    axes[2].spines['top'].set_color('none')
    axes[2].spines['right'].set_color('none')
    x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
    x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
    x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
    axes[2].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=12, loc='upper left')

    plot_stratpd(X, y, "x1", "y", ax=axes[3], pdp_marker_size=1,
                 pdp_marker_color=x1_color,
                 show_x_counts=False, n_trials=1, show_slope_lines=False)
    plot_stratpd(X, y, "x2", "y", ax=axes[3], pdp_marker_size=1,
                 pdp_marker_color=x2_color,
                 show_x_counts=False, n_trials=1, show_slope_lines=False)
    plot_stratpd(X, y, "x3", "y", ax=axes[3], pdp_marker_size=1,
                 pdp_marker_color=x3_color,
                 show_x_counts=False, n_trials=1, show_slope_lines=False)
    axes[3].set_xticks([0,2,4,6,8,10])
    axes[3].set_ylim(-20,160)
    axes[3].set_yticks([0, 25, 50, 75, 100, 125, 150])
    axes[3].set_xlabel("$x_1, x_2, x_3$", fontsize=12)
    # axes[3].set_ylabel("y", fontsize=12)
    axes[3].set_title("(d) StratPD")
    axes[3].spines['top'].set_linewidth(.5)
    axes[3].spines['right'].set_linewidth(.5)
    axes[3].spines['left'].set_linewidth(.5)
    axes[3].spines['bottom'].set_linewidth(.5)
    axes[3].spines['top'].set_color('none')
    axes[3].spines['right'].set_color('none')
    x1_patch = mpatches.Patch(color=x1_color, label='$x_1$')
    x2_patch = mpatches.Patch(color=x2_color, label='$x_2$')
    x3_patch = mpatches.Patch(color=x3_color, label='$x_3$')
    axes[3].legend(handles=[x1_patch,x2_patch,x3_patch], fontsize=12)

plt.tight_layout()
plt.savefig("../images/FPD-SHAP-PD.pdf")
plt.show()