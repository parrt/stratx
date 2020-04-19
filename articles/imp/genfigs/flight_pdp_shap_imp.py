from support import *
from stratx.featimp import *

np.random.seed(1)

n = 2_000

_, _, df_flights = load_flights(n=n)
X, y = df_flights.drop('ARRIVAL_DELAY', axis=1), df_flights['ARRIVAL_DELAY']
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
backing = shap.sample(X_test, 300)
to_explain = X_test.sample(300)

tuned_params = models[("flights", "RF")]
rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
rf.fit(X_train, y_train)

pdp_I = pdp_importances(rf, backing.copy(), numx=300)
print("PDP\n",pdp_I)

shap_I1 = get_shap(rf, to_explain)
shap_I2 = get_shap(rf, to_explain, backing, assume_independence=False)

# print("SHAP", normalized_shap)
print("SHAP\n",shap_I1)

fig, axes = plt.subplots(1,3,figsize=(12,4))
plot_importances(pdp_I, ax=axes[0], imp_range=(0,.4))
plot_importances(shap_I1, ax=axes[1], imp_range=(0,.4))
plot_importances(shap_I2, ax=axes[2], imp_range=(0,.4))
axes[0].set_xlabel("(a) Friedman $\overline{|PD_j|}$ importance")
axes[1].set_xlabel("(b) SHAP $j$ importance")
axes[2].set_xlabel("(c) SHAP $j$ importance\ninterventional")
#plt.suptitle(f"NYC rent feature importance rankings", fontsize=10)
plt.tight_layout()
plt.savefig(f"/Users/parrt/Desktop/flight-pdp-vs-shap.pdf", pad_inches=0)

plt.show()
