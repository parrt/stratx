from support import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25_000
model='RF' # ('RF','SVM','GBM','OLS','Lasso')

X, y, _ = load_flights(n=n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# use same set of folds for all techniques
kfolds = 2
kf = KFold(n_splits=kfolds, shuffle=True)


def gen(model, rank):
    R, imps = \
        compare_top_features(X, y,
                             X_train, X_test, y_train, y_test,
                             kf,
                             n_shap=300,
                             catcolnames={'AIRLINE',
                                          'ORIGIN_AIRPORT',
                                          'DESTINATION_AIRPORT',
                                          'FLIGHT_NUMBER',
                                          'DAY_OF_WEEK'},
                             sortby=rank,
                             model=model,
                             metric=mean_absolute_error,
                             stratpd_min_samples_leaf=10,
                             stratpd_cat_min_samples_leaf=3,
                             use_oob=use_oob,
                             imp_n_trials=10,
                             imp_pvalues_n_trials=0,
                             #min_slopes_per_x=1,
                             normalize=True,
                             top_features_range=(1, 8),
                             drop=['Spearman','PCA'])

    plot_importances(imps['StratImpact'].iloc[0:8], imp_range=(0, .4), width=4.1,
                     title="Flight delay StratImpact importances")
    plt.tight_layout()
    plt.savefig("../images/flights-features.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    plot_importances(imps['RF SHAP'].iloc[0:8], imp_range=(0, .4), width=4.1,
                     title="Flight delay RF SHAP importances")
    plt.tight_layout()
    plt.savefig("../images/flights-features-shap-rf.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    print(R)

    # R = R.reset_index(drop=True)
    # R.reset_index().to_feather("/tmp/flights.feather")

    plot_topk(R, k=8, title=f"{model} Flight arrival delay",
              ylabel="5-fold CV MAE (mins)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/flights-topk-{model}-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

gen(model='RF', rank='Importance')
gen(model='RF', rank='Impact')
gen(model='GBM', rank='Importance')
