from support import *

figsize = (3.5, 3.0)
use_oob=False
model='RF' # ('RF','SVM','GBM','OLS','Lasso')

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]
metric = mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def gen(model, rank):
    # need small or 1 min_slopes_per_x given tiny toy dataset
    R, imps = \
        compare_top_features(X, y, n_shap=n,
                             sortby=rank,
                             metric=metric,
                             imp_n_trials=10,
                             use_oob=use_oob,
                             kfolds=1,
                             model=model,
                             top_features_range=(1,8),
                             drop=['Spearman','PCA'])

    plot_importances(imps['StratImpact'].iloc[:8], imp_range=(0,0.4), width=3,
                     title="Boston StratImpact importances")
    plt.tight_layout()
    plt.savefig("../images/boston-features.pdf")
    plt.show()

    plot_importances(imps['RF SHAP'].iloc[:8], imp_range=(0,0.4), width=3,
                     title="Boston SHAP RF importances")
    plt.tight_layout()
    plt.savefig("../images/boston-features-shap-rf.pdf")
    plt.show()

    print(R)
    R.reset_index().to_feather("/tmp/boston.feather")

    plot_topk(R, k=8, title=f"{model} Boston housing prices",
              ylabel="20% 5-fold CV MAE (k$)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/boston-topk-{model}-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


gen(model='RF', rank='Importance')
gen(model='RF', rank='Impact')
gen(model='GBM', rank='Importance')
