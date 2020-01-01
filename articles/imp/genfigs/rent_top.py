from support import *

n = 30_000 # more and shap gets bus error it seems
use_oob=False
metric = mean_absolute_error
X, y = load_rent(n=n)

R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         metric=metric,
                         use_oob=use_oob,
                         top_features_range=(1, 8),
                         drop=['Spearman','PCA'])

plot_importances(our_I.iloc[:8], imp_range=(0,0.4), width=3,
                 title="Rent StratImpact importances")
plt.tight_layout()
plt.savefig("../images/rent-features.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

plot_importances(rf_I.iloc[0:8], imp_range=(0, .4), width=3,
                 title="Rent RF SHAP importances")
plt.tight_layout()
plt.savefig("../images/rent-features-shap-rf.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


print(R)

fig, ax = plt.subplots(1,1,figsize=(4,3.5))
plot_topk(R, ax, k=8)
ax.set_ylabel("Training MAE ($)")
ax.set_title("NYC rent prices")
plt.tight_layout()
plt.savefig("../images/rent-topk.pdf", bbox_inches="tight", pad_inches=0)
plt.show()