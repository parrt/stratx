from support import *
from stratx.featimp import *

np.random.seed(1)

X, y, X_train, X_test, y_train, y_test = load_dataset("rent", "price")
backing = shap.sample(X_test, 100)
to_explain = X_test.sample(300)

tuned_params = models[("rent", "RF")]
rf = RandomForestRegressor(**tuned_params, n_jobs=-1)
rf.fit(X_train, y_train)
print("R^2 test",rf.score(X_test,y_test))

pdp_I = pdp_importances(rf, backing.copy(), numx=300)
print("PDP\n",pdp_I)

# shap_I1 = get_shap(rf, to_explain)
shap_I2 = get_shap(rf, to_explain, backing, assume_independence=False)

# print("SHAP", normalized_shap)
print("SHAP\n",shap_I2)

fig, axes = plt.subplots(1,2,figsize=(6,2.7))
plot_importances(pdp_I[0:8], ax=axes[0], imp_range=(0,.4))
# plot_importances(shap_I1, ax=axes[1], imp_range=(0,.4))
plot_importances(shap_I2[0:8], ax=axes[1], imp_range=(0,.4))
axes[0].set_xlabel("(a) $\overline{|FPD_j-\overline{y}|}$ importance")
# axes[1].set_xlabel("(b) SHAP $j$ importance")
axes[1].set_xlabel("(b) SHAP $x_j$ importance\n(interventional)")
#plt.suptitle(f"NYC rent feature importance rankings", fontsize=10)
plt.tight_layout()
plt.suptitle("Rent FPD vs SHAP", fontsize=10)
plt.savefig(f"../images/rent-pdp-vs-shap.pdf", bbox_inches="tight", pad_inches=0)

plt.show()
