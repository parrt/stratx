from support import *

figsize = (3.5, 3.0)
use_oob=False
n = 25_000
metric = mean_absolute_error
model='RF' # ('RF','SVM','GBM','OLS','Lasso')

# np.random.seed(999) # set for testing effects

X, y = load_rent(n=n)

def gen(model, rank):
    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             sortby=rank,
                             metric=metric,
                             use_oob=use_oob,
                             kfolds=1,
                             model=model,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=10,
                             imp_pvalues_n_trials=0,
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
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/rent-topk-{model}-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

gen(model='RF', rank='Importance')
gen(model='RF', rank='Impact')
gen(model='GBM', rank='Importance')
