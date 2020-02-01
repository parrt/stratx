from support import *

figsize = (3.5, 3.0)
use_oob=False
model='RF' # ('RF','SVM','GBM','OLS','Lasso')

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target)
n = X.shape[0]
metric = mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# use same set of folds for all techniques
kfolds = 5
kf = KFold(n_splits=kfolds, shuffle=True)
kfold_indexes = list(kf.split(X))

R_cache = {} # save R per rank

def gen(model, rank):
    if rank in R_cache:
        R = R_cache[rank]
    else:
        R, imps = \
            compare_top_features(X, y,
                                 X_train, X_test, y_train, y_test,
                                 kfold_indexes,
                                 n_shap=n,
                                 sortby=rank,
                                 metric=metric,
                                 imp_n_trials=10,
                                 use_oob=use_oob,
                                 model=model,
                                 top_features_range=(1,8),
                                 drop=['Spearman','PCA'])
        R_cache[rank] = R
        plot_importances(imps['StratImpact'].iloc[:8], imp_range=(0,0.4), width=3,
                         title="Boston StratImpact importances")
        plt.tight_layout()
        plt.savefig("../images/boston-features.pdf")
        # plt.show()
        plt.close()

        plot_importances(imps['RF SHAP'].iloc[:8], imp_range=(0,0.4), width=3,
                         title="Boston SHAP RF importances")
        plt.tight_layout()
        plt.savefig("../images/boston-features-shap-rf.pdf")
        # plt.show()
        plt.close()

    print(R)

    R_ = {k:v for k,v in R.items() if k not in ['Spearman','PCA']}
    plot_topk(R_, k=8, title=f"{model} Boston housing prices",
              ylabel="5-fold CV MAE (k$)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              yrange=(2, 5.5),
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/boston-topk-{model}-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

# Show BOSTON Spearman's vs ours
def baseline(rank):
    R = R_cache[rank]
    R_ = {k:v for k,v in R.items() if k in ['Spearman','PCA', 'OLS', 'StratImpact']}
    plot_topk(R_, k=8, title="RF Boston housing prices",
              ylabel="5-fold CV MAE (k$)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              yrange=(2, 5.5),
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/boston-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


gen(model='RF', rank='Importance')
gen(model='RF', rank='Impact')
gen(model='GBM', rank='Importance')

baseline(rank='Importance')
