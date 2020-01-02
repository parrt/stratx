from support import *

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

np.set_printoptions(precision=4, suppress=True, linewidth=150)


# Show BOSTON Spearman's vs ours
def boston():
    boston = load_boston()
    data = boston.data
    # data = normalize(boston.data)
    data = StandardScaler().fit_transform(data)
    X = pd.DataFrame(data, columns=boston.feature_names)
    y = pd.Series(boston.target)

    R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             min_slopes_per_x=5,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    plot_topk(R, ax, k=8)
    ax.set_ylabel("20% Validation MAE (k$)")
    ax.set_title("Boston housing prices")
    plt.tight_layout()
    plt.savefig("../images/boston-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def bulldozer():
    n = 25_000  # shap crashes above this; 20k works

    X, y = load_bulldozer()
    X = X.iloc[-n:]
    y = y.iloc[-n:]

    R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             min_slopes_per_x=5,
                             catcolnames={'AC', 'ModelID', 'YearMade', 'ProductSize'},
                             top_features_range=(1, 8),
                             include=['Spearman','PCA', 'OLS', 'StratImpact'])

    print(R)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    plot_topk(R, ax, k=8)
    ax.set_ylabel("20% Validation MAE ($)")
    ax.set_title("Bulldozer auction prices")
    plt.tight_layout()
    plt.savefig("../images/bulldozer-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def rent():
    n = 30_000  # more and shap gets bus error it seems
    X, y = load_rent(n=n)

    R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
        compare_top_features(X, y, n_shap=300,
                             min_slopes_per_x=5,
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    plot_topk(R, ax, k=8)
    ax.set_ylabel("20% Validation MAE ($)")
    ax.set_title("NYC rent prices")
    plt.tight_layout()
    plt.savefig("../images/rent-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def flight():
    n = 25_000  # 30k crashes shap so try 20k

    X, y, _ = load_flights(n=n)

    R, spear_I, pca_I, ols_I, shap_ols_I, rf_I, perm_I, our_I = \
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
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    plot_topk(R, ax, k=8)
    ax.set_ylabel("20% Validation MAE (minutes)")
    ax.set_title(f"Flight arrival delay")
    plt.tight_layout()
    plt.savefig("../images/flights-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


flight()
# boston()
# bulldozer()
# rent()
