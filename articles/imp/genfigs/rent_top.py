from support import *

figsize = (3.5, 3.0)
use_oob=False
n = 25_000
metric = mean_absolute_error
model='RF' # ('RF','SVM','GBM')

X, y = load_rent(n=n)

R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         metric=metric,
                         use_oob=use_oob,
                         kfolds=5,
                         model=model,
                         stratpd_min_samples_leaf=5, # overcome colinearity
                         imp_n_trials=10,
                         catcolnames=['bathrooms'], # numeric version ignores too much data
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

plot_topk(R, k=8, title=f"{model} NYC rent prices",
          ylabel="20% 5-fold CV MAE ($)",
          title_fontsize=14,
          label_fontsize=14,
          ticklabel_fontsize=10,
          figsize=figsize)
plt.tight_layout()
plt.savefig(f"../images/rent-topk-{model}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()