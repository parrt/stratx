from support import *
import numpy as np

np.random.seed(1)

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25_000
model='RF' # ('RF','SVM','GBM','OLS','Lasso')
X, y = load_bulldozer(n)

gen_topk_figs(X,y,kfolds=5,n_trials=1,dataset="bulldozer",title="Bulldozer auction prices",
              # cat_min_samples_leaf=5,
              # min_slopes_per_x=15,
              catcolnames={'AC', 'ModelID'},
              yrange=(5000,20000), yunits="$")

"""
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2)
# use same set of folds for all techniques
kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True)
kfold_indexes = list(kf.split(X_))

def gen(model, rank):
    R, imps = \
        compare_top_features(X_, y_,
                             X_train, X_test, y_train, y_test,
                             kfold_indexes,
                             n_shap=300,
                             catcolnames={'AC', 'ModelID',
                                          #'ProductSize'
                                          },
                             sortby=rank,
                             metric=metric,
                             use_oob=use_oob,
                             imp_n_trials=10,
                             imp_pvalues_n_trials=0,
                             model=model,
                             stratpd_min_samples_leaf=10,
                             stratpd_cat_min_samples_leaf=5,
                             normalize=True,
                             # min_slopes_per_x=8,
                             top_features_range=(1, 8),
                             #include=['StratImpact']
                             drop=['Spearman','PCA']
                             )

    plot_importances(imps['StratImpact'].iloc[:8], imp_range=(0,0.4), width=3,
                     title="Bulldozer StratImpact importances")
    plt.tight_layout()
    plt.savefig("../images/bulldozer-features.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close()

    plot_importances(imps['RF SHAP'].iloc[0:8], imp_range=(0, .4), width=3,
                     title="Bulldozer RF SHAP importances")
    plt.tight_layout()
    plt.savefig("../images/bulldozer-features-shap-rf.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close()

    print(R)

    plot_topk(R, k=8, title=f"{model} Bulldozer auction prices",
              ylabel=f"5-fold CV MAE ({yunits})",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              yrange=(5000,20000),
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/bulldozer-topk-{model}-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def baseline(rank):
    R, imps = \
        compare_top_features(X_, y_,
                             X_train, X_test, y_train, y_test,
                             kfold_indexes,
                             sortby=rank,
                             n_shap=300,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=5,
                             catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    plot_topk(R, k=8, title="RF Bulldozer auction prices",
              ylabel=f"5-fold CV MAE ({yunits})",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              yrange=(5000,20000),
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/bulldozer-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


gen(model='RF', rank='Importance')
gen(model='RF', rank='Impact')
gen(model='GBM', rank='Importance')

baseline(rank='Importance')
"""
