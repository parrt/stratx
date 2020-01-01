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
                             min_slopes_per_x=1,
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
                             top_features_range=(1, 8),
                             include=['Spearman','PCA','OLS','StratImpact'])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    plot_topk(R, ax, k=8)
    ax.set_ylabel("20% Validation MAE ($)")
    ax.set_title("NYC rent prices")
    plt.tight_layout()
    plt.savefig("../images/rent-topk-spearman.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def boston_pca():
    boston = load_boston()
    data = boston.data
    X = pd.DataFrame(data, columns=boston.feature_names)

    pca_I = pca_importances(X)
    print(pca_I)


def rent_pca():
    n = 30_000  # more and shap gets bus error it seems
    X, y = load_rent(n=n)
    pca_I = pca_importances(X)
    print(pca_I)


def bulldozer_pca():
    n = 20_000  # shap crashes above this; 20k works
    X, y = load_bulldozer()
    X = X.iloc[-n:]
    pca_I = pca_importances(X)
    print(pca_I)

# boston()
bulldozer()
# rent()
