from support import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25_000 # shap crashes above this; 20k works

X, y = load_bulldozer()

X = X.iloc[-n:]
y = y.iloc[-n:]

R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
    compare_top_features(X, y, n_shap=300,
                         catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                         metric=metric,
                         use_oob=use_oob,
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

plot_topk(R, k=8, title="Bulldozer auction prices",
          ylabel="20% 5-fold CV MAE ($)",
          title_fontsize=14,
          label_fontsize=14,
          ticklabel_fontsize=14,
          figsize=figsize)
plt.tight_layout()
plt.savefig(f"../images/bulldozer-topk{'-oob' if use_oob else ''}.pdf", bbox_inches="tight", pad_inches=0)
plt.show()