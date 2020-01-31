from support import *

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

np.set_printoptions(precision=4, suppress=True, linewidth=150)

figsize = (3.5, 3)

# Show BOSTON Spearman's vs ours
def boston(rank):
    boston = load_boston()
    data = boston.data
    # data = normalize(boston.data)
    data = StandardScaler().fit_transform(data)
    X = pd.DataFrame(data, columns=boston.feature_names)
    y = pd.Series(boston.target)

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             kfolds=5,
                             imp_n_trials=10,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    plot_topk(R, k=8, title="RF Boston housing prices",
              ylabel="20% 5-fold CV MAE (k$)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/boston-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def bulldozer(rank):
    n = 25_000 # shap crashes above this

    X, y = load_bulldozer()
    X = X.iloc[-n:]
    y = y.iloc[-n:]

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             kfolds=5,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=5,
                             catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    plot_topk(R, k=8, title="RF Bulldozer auction prices",
              ylabel="20% 5-fold CV MAE ($)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/bulldozer-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def rent(rank):
    n = 25_000
    X, y = load_rent(n=n)

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             kfolds=5,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=5,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])

    plot_topk(R, k=8, title="RF NYC rent prices",
              ylabel="20% 5-fold CV MAE ($)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14, # make font a bit bigger as we shrink this one is paper a bit
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.savefig(f"../images/rent-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def flight(rank):
    n = 25_000

    X, y, _ = load_flights(n=n)

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             catcolnames={'AIRLINE',
                                          'ORIGIN_AIRPORT',
                                          'DESTINATION_AIRPORT',
                                          'FLIGHT_NUMBER',
                                          'DAY_OF_WEEK'},
                             metric=mean_squared_error,
                             kfolds=5,
                             imp_n_trials=10,
                             stratpd_min_samples_leaf=10,
                             # a bit less than usual (gridsearch showed how to get value)
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])
    plot_topk(R, k=8, title="RF Flight arrival delay",
              ylabel="20% 5-fold CV MAE (mins)",
              xlabel=f"Top $k$ feature {rank}",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig(f"../images/flights-topk-baseline-{rank}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


flight(rank='Importance')
boston(rank='Importance')
bulldozer(rank='Importance')
rent(rank='Importance')
