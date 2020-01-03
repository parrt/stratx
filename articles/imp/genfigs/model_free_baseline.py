from support import *

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

np.set_printoptions(precision=4, suppress=True, linewidth=150)

figsize = (3.5, 3)

# Show BOSTON Spearman's vs ours
def boston():
    boston = load_boston()
    data = boston.data
    # data = normalize(boston.data)
    data = StandardScaler().fit_transform(data)
    X = pd.DataFrame(data, columns=boston.feature_names)
    y = pd.Series(boston.target)

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             min_slopes_per_x=5,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    plot_topk(R, k=8, title="Boston housing prices",
              ylabel="20% 5-fold CV MAE (k$)",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig("../images/boston-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def bulldozer():
    n = 25_000  # shap crashes above this; 20k works

    X, y = load_bulldozer()
    X = X.iloc[-n:]
    y = y.iloc[-n:]

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             min_slopes_per_x=5,
                             catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    plot_topk(R, k=8, title="Bulldozer auction prices",
              ylabel="20% 5-fold CV MAE ($)",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig("../images/bulldozer-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def rent():
    n = 25_000  # more and shap gets bus error it seems
    X, y = load_rent(n=n)

    R, Rstd, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             min_slopes_per_x=5,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])

    plot_topk(R, k=8, title="NYC rent prices",
              ylabel="20% 5-fold CV MAE ($)",
              title_fontsize=15, # make font a bit bigger as we shrink this one is paper a bit
              label_fontsize=15,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.savefig("../images/rent-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def flight():
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
                             min_slopes_per_x=5,
                             # a bit less than usual (gridsearch showed how to get value)
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])
    plot_topk(R, k=8, title="Flight arrival delay",
              ylabel="20% 5-fold CV MAE (mins)",
              title_fontsize=14,
              label_fontsize=14,
              ticklabel_fontsize=10,
              figsize=figsize)
    plt.tight_layout()
    plt.savefig("../images/flights-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


flight()
boston()
bulldozer()
rent()
