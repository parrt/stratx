import shap
from stratx.ice import friedman_partial_dependence
from stratx.partdep import plot_stratpd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def synthetic_interaction_data(n, yintercept = 10):
    df = pd.DataFrame()
    df[f'x1'] = np.random.random(size=n)*10+1
    df[f'x2'] = np.random.random(size=n)*10+1
    df['y'] = df['x1']**2 + df['x1']*df['x2'] + np.sin(df['x2']*2)*10 + yintercept
#     df['y'] = df['x1'] * df['x2'] + yintercept
    return df

# def f(x,y_intercept=10):
#     if x.ndim==2:
#         return np.array([f(row) for row in x])
#     return x[0]**2 + x[1] + y_intercept

df = synthetic_interaction_data(400)

X, y = df[['x1','x2']].copy(), df['y'].copy()
X1 = X.iloc[:,0]
X2 = X.iloc[:,1]

rf = RandomForestRegressor(n_estimators=10)
rf.fit(X, y)
print("R^2 ",rf.score(X,y))

print("mean(y) =", np.mean(y))
print("mean(X_1), mean(X_2) =", np.mean(X1), np.mean(X2))

pdp_x1 = friedman_partial_dependence(rf, X, 'x1', numx=None, mean_centered=False)
pdp_x2 = friedman_partial_dependence(rf, X, 'x2', numx=None, mean_centered=False)
print("mean(PDP_1) =", np.mean(pdp_x1[1]))
print("mean(PDP_2) =", np.mean(pdp_x2[1]))

explainer = shap.TreeExplainer(rf, data=shap.sample(X, 400), feature_perturbation='interventional')
shap_values = explainer.shap_values(X, check_additivity=False)
shapavg = np.mean(shap_values, axis=0)
print("SHAP avg x1,x2 =", shapavg)
shapimp = np.mean(np.abs(shap_values), axis=0)
print("SHAP avg |x1|,|x2| =", shapimp)

fig, axes = plt.subplots(1,3,figsize=(10,4.5))

axes[0].plot(pdp_x1[0], pdp_x1[1], '.', markersize=5, c='#1E88E5', label='$PDP_1$', alpha=.5)
axes[0].plot(pdp_x2[0], pdp_x2[1], '.', markersize=5, c='orange', label='$PDP_2$', alpha=.5)
axes[0].plot([min(pdp_x1[0]),max(pdp_x1[0])], [np.mean(pdp_x1[1])]*2, ':', c='#1E88E5', label=r"$\overline{PDP_1}$")
axes[0].plot([min(pdp_x2[0]),max(pdp_x2[0])], [np.mean(pdp_x2[1])]*2, ':', c='orange', label=r"$\overline{PDP_2}$")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("y")
axes[0].set_title(f"Friedman PDP\n$\\bar{{y}}={np.mean(y):.2f}$\n$y = x_1^2 + x_1 x_2 + sin(2 x_2) + 10$", fontsize=10)
axes[0].legend(fontsize=9)

#axes[1].plot(shap_values)
shap.dependence_plot("x1", shap_values, X,
                     interaction_index=None, ax=axes[1], dot_size=10,
                     show=False, alpha=.5)
shap.dependence_plot("x2", shap_values, X,
                     interaction_index=None, ax=axes[1], dot_size=10,
                     show=False, alpha=.5, color='orange')
axes[1].set_xticks([2,4,6,8])
axes[1].set_title("SHAP x1, x2")
axes[1].set_ylabel("SHAP values")


plot_stratpd(X, y, "x1", "y", ax=axes[2], pdp_marker_size=1,
             pdp_marker_color='#1E88E5',
             show_x_counts=False, n_trials=1, show_slope_lines=False)
plot_stratpd(X, y, "x2", "y", ax=axes[2], pdp_marker_size=1,
             pdp_marker_color='orange',
             show_x_counts=False, n_trials=1, show_slope_lines=False)
axes[2].set_ylim(-10,180)
axes[2].set_xlabel("x1,x2")
axes[2].set_ylabel("y")
axes[2].set_title("StratImpact x1, x2")

plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/all.pdf")
plt.show()