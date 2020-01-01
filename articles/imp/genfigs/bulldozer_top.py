from support import *

use_oob=True
metric = mean_absolute_error
n = 20_000 # shap crashes above this; 20k works

X, y = load_bulldozer()
print("Loaded...")

X = X.iloc[-n:]
y = y.iloc[-n:]

R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         stratpd_min_samples_leaf=20, # gridsearch showed 20 better than 15
                         # min_slopes_per_x=15, # default
                         catcolnames={'AC', 'ModelID'},
                         metric=metric,
                         use_oob=use_oob,
                         top_features_range=(1, 8),
                         drop=['Spearman','PCA'])

plot_importances(our_I.iloc[:8], imp_range=(0,0.4), width=3,
                 title="Bulldozer StratImpact importances")
plt.tight_layout()
plt.savefig("../images/bulldozer-features.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

plot_importances(rf_I.iloc[0:8], imp_range=(0, .4), width=3,
                 title="Bulldozer RF SHAP importances")
plt.tight_layout()
plt.savefig("../images/bulldozer-features-shap-rf.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

print(R)

fig, ax = plt.subplots(1,1,figsize=(4,3.5))

plot_topk(R, ax, k=8)
if use_oob:
    ax.set_ylabel("RF Out-of-bag $1-R^2$")
else:
    ax.set_ylabel("Training MAE ($)")
ax.set_title(f"{'OOB Error: ' if use_oob else ''}Bulldozer auction prices")
plt.tight_layout()
plt.savefig(f"../images/bulldozer-topk{'-oob' if use_oob else ''}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()