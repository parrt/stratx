import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

shap_test_size = 300
TUNE_RF = False
df = pd.read_csv("bulldozer10k.csv")
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

fig, axes = plt.subplots(1, 2, figsize=(6,2.8))
axes[0].scatter(X['YearMade'], y, s=3, alpha=.1, c='#1E88E5')
axes[0].set_xlim(1960,2010)
axes[0].set_xlabel("YearMade", fontsize=11)
axes[0].set_ylabel("SalePrice ($)", fontsize=11)
axes[0].set_title("Marginal plot", fontsize=13)


rf = RandomForestRegressor(n_estimators=150, n_jobs=-1,
                           max_features=0.9,
                           min_samples_leaf=1, oob_score=True)
rf.fit(X, y)
print("Bulldozer RF OOB R^2", rf.oob_score_)

explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100),
                               feature_perturbation='interventional')
X_test = X.sample(n=shap_test_size)
shap_values = explainer.shap_values(X_test, check_additivity=False)

shap.dependence_plot("YearMade", shap_values, X_test,
                     interaction_index=None, ax=axes[1], dot_size=5,
                     show=False, alpha=.5)
axes[1].set_xlim(1960,2010)
plt.show()