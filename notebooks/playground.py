from sklearn.datasets import load_boston, load_diabetes
from stratx.partdep import *
from stratx.featimp import *
from rfpimp import plot_importances

PLOT = False

def synthetic_poly_data(n, p, degree=1):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p)*10 # get p random coefficients
    coeff = np.array([5, 2, 10])
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)*10,1)
    #df['x3'] = df['x1']  # copy x1
    # multiply coefficients x each column (var) and sum along columns
    # df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 )
    df['y'] = 5*df['x1'] + 2*df['x2'] + 3*df['x3']**2
    #TODO add noise
    return df, coeff


df, coeff = synthetic_poly_data(500,3)
X = df.drop('y', axis=1)
y = df['y']
# X = standardize(X)

I = importances(X, y)
plot_importances(I, imp_range=(0,1))

if True:
    fig, ax = plt.subplots(1,1)
    xc = 'x2'
    min_samples_leaf = 5
    for colname in X.columns:
        print(colname)
        plot_stratpd(X, y, colname, 'y', ax=ax, min_samples_leaf=min_samples_leaf)
    # ax.scatter(df['x'], df['sum_pd'], s=3, c='k')

    # uniq_x = np.array(sorted(np.unique(X[xc])))
    # ax.scatter(uniq_x[:-1], marginal_slopes)
    ax.set_title(f"Coeffs = {coeff}, xc={xc}")
    plt.tight_layout()
    plt.show()

# WORKS ONLY WITH DATAFRAMES AT MOMENT



