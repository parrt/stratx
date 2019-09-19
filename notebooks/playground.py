from sklearn.datasets import load_boston, load_diabetes
from stratx.partdep import *
from stratx.featimp import *

PLOT = True

if PLOT:
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MEDV'] = boston.target
    # df = df.head(50)

    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    fig, ax = plt.subplots(1,1)
    # WORKS ONLY WITH DATAFRAMES AT MOMENT
    plot_stratpd(X, y, 'AGE', 'MEDV', yrange=(-10,10), ax=ax)
    marginal_xranges, marginal_sizes, marginal_slopes, ignored = \
        discrete_xc_space(X['AGE'], y)

    uniq_x = np.array(sorted(np.unique(X['AGE'])))

    ax.scatter(uniq_x[:-1], marginal_slopes)
    plt.tight_layout()
    plt.show()

boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['MEDV'] = boston.target
df = df.head(50)

X = df.drop('MEDV', axis=1)
y = df['MEDV']

# WORKS ONLY WITH DATAFRAMES AT MOMENT
importances(X, y, 'AGE')

