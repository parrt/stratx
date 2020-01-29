from support import *
import numpy as np

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25_000
model='RF' # ('RF','SVM','GBM','OLS','Lasso')

# np.seterr(divide='raise')

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

idxs = resample(range(50_000), n_samples=n, replace=False)
X_, y_ = X.iloc[idxs], y.iloc[idxs]

R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X_, y_, n_shap=300,
                         catcolnames={'AC', 'ModelID',
                                      #'ProductSize'
                                      },
                         metric=metric,
                         use_oob=use_oob,
                         kfolds=1,
                         imp_n_trials=3,
                         imp_pvalues_n_trials=20,
                         model=model,
                         stratpd_min_samples_leaf=10,
                         stratpd_cat_min_samples_leaf=3,
                         normalize=False,
                         # min_slopes_per_x=8,
                         top_features_range=(1, 8),
                         #include=['StratImpact']
                         drop=['Spearman','PCA']
                         )

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

plot_topk(R, k=8, title=f"{model} Bulldozer auction prices",
          ylabel="20% 5-fold CV MAE ($)",
          title_fontsize=14,
          label_fontsize=14,
          ticklabel_fontsize=10,
          figsize=figsize)
plt.tight_layout()
plt.savefig(f"../images/bulldozer-topk-{model}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()