from sklearn.datasets import load_boston, load_diabetes
from stratx.partdep import *
from stratx.featimp import *

PLOT = True

def synthetic_poly_data(n, p, degree=1):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p)*10 # get p random coefficients
    coeff = np.array([5, 2, 10])
    for i in range(p):
        df[f'x{i+1}'] = np.random.random_sample(size=n)
    # multiply coefficients x each column (var) and sum along columns
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 )
    #TODO add noise
    return df, coeff


df, coeff = synthetic_poly_data(100,3)
X = df.drop('y', axis=1)
y = df['y']
X = standardize(X)

if PLOT:

    fig, ax = plt.subplots(1,1)
    xc = 'x2'
    min_samples_leaf = 10
    plot_stratpd(X, y, 'x1', 'y', ax=ax, min_samples_leaf=min_samples_leaf)
    plot_stratpd(X, y, 'x2', 'y', ax=ax, min_samples_leaf=min_samples_leaf)
    plot_stratpd(X, y, 'x3', 'y', ax=ax, min_samples_leaf=min_samples_leaf)
    # marginal_xranges, marginal_sizes, marginal_slopes, ignored = \
    #     discrete_xc_space(X['AGE'], y)

    # uniq_x = np.array(sorted(np.unique(X[xc])))
    # ax.scatter(uniq_x[:-1], marginal_slopes)
    ax.set_title(f"Coeffs = {coeff}, xc={xc}")
    plt.tight_layout()
    plt.show()

# WORKS ONLY WITH DATAFRAMES AT MOMENT

importances(X, y)

