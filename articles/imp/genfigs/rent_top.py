from support import *

figsize = (3.5, 3.0)
use_oob=False
n = 25_000
metric = mean_absolute_error
model='RF' # ('RF','SVM','GBM','OLS','Lasso')

# np.random.seed(999) # set for testing effects

X, y = load_rent(n=n)

gen_topk_figs(X,y,kfolds=5,n_trials=5,dataset="rent",title="NYC rent prices",
              min_slopes_per_x=15,
              yrange=(300,900), yunits="$")

"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# use same set of folds for all techniques
kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True)
kfold_indexes = list(kf.split(X))

def gen(model, rank):
    R, imps = \
        compare_top_features(X, y,
                             X_train, X_test, y_train, y_test,
                             kfold_indexes,
                             n_shap=300,
                             sortby=rank,
                             metric=metric,
                             use_oob=use_oob,
                             model=model,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=10,
                             imp_pvalues_n_trials=0,
                             top_features_range=(1, 8),
                             drop=['Spearman','PCA'])

    plot_importances(imps['StratImpact'].iloc[:8], imp_range=(0,0.4), width=3,
                     title="Rent StratImpact importances")
    plt.tight_layout()
    plt.savefig("../images/rent-features.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close()

    plot_importances(imps['RF SHAP'].iloc[0:8], imp_range=(0, .4), width=3,
                     title="Rent RF SHAP importances")
    plt.tight_layout()
    plt.savefig("../images/rent-features-shap-rf.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close()

    print(R)

    plot_topk(R, k=8, title=f"{model} NYC rent prices",
              ylabel="5-fold CV MAE ($)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              yrange=(300,900),
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/rent-topk-{model}-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

def baseline(rank):
    R, imps = \
        compare_top_features(X, y,
                             X_train, X_test, y_train, y_test,
                             kfold_indexes,
                             sortby=rank,
                             n_shap=300,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=5,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])

    plot_topk(R, k=8, title="RF NYC rent prices",
              ylabel="5-fold CV MAE ($)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14, # make font a bit bigger as we shrink this one is paper a bit
              label_fontsize=14,
              ticklabel_fontsize=10,
              yrange=(300,900),
              figsize=figsize)
    plt.savefig(f"../images/rent-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

gen(model='RF', rank='Importance')
gen(model='RF', rank='Impact')
gen(model='GBM', rank='Importance')

baseline(rank='Importance')
"""
