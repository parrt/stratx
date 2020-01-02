from support import *

use_oob=False
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]
metric = mean_absolute_error

# need small or 1 min_slopes_per_x given tiny toy dataset
R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=n,
                         metric=metric,
                         use_oob=use_oob,
                         min_slopes_per_x=5,
                         drop=['Spearman','PCA'])

plot_importances(our_I.iloc[:8], imp_range=(0,0.4), width=3,
                 title="Boston StratImpact importances")
plt.tight_layout()
plt.savefig("../images/boston-features.pdf")
plt.show()

plot_importances(rf_I.iloc[:8], imp_range=(0,0.4), width=3,
                 title="Boston SHAP RF importances")
plt.tight_layout()
plt.savefig("../images/boston-features-shap-rf.pdf")
plt.show()

print(R)
R.reset_index().to_feather("/tmp/boston.feather")

fig, ax = plt.subplots(1,1,figsize=(3.5,3))
plot_topk(R, ax, k=8)
if use_oob:
    ax.set_ylabel("RF Out-of-bag $1-R^2$")
else:
    ax.set_ylabel("20% Validation MAE (k$)")
ax.set_title(f"{'OOB Error: ' if use_oob else ''}Boston housing prices")
plt.tight_layout()
plt.savefig(f"../images/boston-topk{'-oob' if use_oob else ''}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()