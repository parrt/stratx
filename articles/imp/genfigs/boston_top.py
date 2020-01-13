from support import *

figsize = (3.5, 3.0)
use_oob=False
model='RF' # ('RF','SVM','GBM')

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]
metric = mean_absolute_error

# need small or 1 min_slopes_per_x given tiny toy dataset
R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=n,
                         metric=metric,
                         imp_n_trials=10,
                         use_oob=use_oob,
                         kfolds=1,
                         model=model,
                         top_features_range=(1,8),
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

plot_topk(R, k=8, title=f"{model} Boston housing prices",
          ylabel="20% 5-fold CV MAE (k$)",
          title_fontsize=14,
          label_fontsize=14,
          ticklabel_fontsize=10,
          figsize=figsize)
plt.tight_layout()
plt.savefig(f"../images/boston-topk-{model}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()