from support import *

figsize = (3.2, 2.8)
use_oob=False
n = 30_000 # more and shap gets bus error it seems
metric = mean_absolute_error
X, y = load_rent(n=n)

R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         metric=metric,
                         use_oob=use_oob,
                         min_slopes_per_x=5, # a bit less than usual (gridsearch showed how to get value)
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

plot_topk(R, k=8, title="NYC rent prices",
          ylabel="20% Validation MAE ($)",
          title_fontsize=15, # make font a bit bigger as we shrink this one is paper a bit
          label_fontsize=15,
          ticklabel_fontsize=15,
          figsize=figsize)
plt.tight_layout()
plt.savefig(f"../images/rent-topk{'-oob' if use_oob else ''}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()