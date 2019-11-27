import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib
# matplotlib.use('TkAgg') # separate window

# from stratx import featimp
from stratx.partdep import *
from rfpimp import plot_importances
import rfpimp

palette = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928"
]

def synthetic_poly_data(n, p):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p)*1 # get p random coefficients
    coeff = np.array([1,1,1,1,1,1,1,1,1])
    # coeff = np.array([5,10])
    for j in range(p):
        df[f'x{j+1}'] = np.round(np.random.random_sample(size=n)*1,1) # shift x_i to right 2
    #df['x3'] = df['x1']*np.round(np.random.random_sample(size=n),1) # copy x1 + noise
    # multiply coefficients x each column (var) and sum along columns
    yintercept = 10
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + '+'.join(terms)
    return df, coeff, eqn


def impact_importances(X: pd.DataFrame,
                       y: pd.Series,
                       k: int=1) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    p = X.shape[1]
    variations = np.zeros(shape=(p,))
    stddevs = np.zeros(shape=(p,))
    total_dy_mass = 0.0
    for j, colname in enumerate(X.columns):
        # Ask stratx package for the partial dependence of y with respect to X[colname]
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
            partial_dependence(X=X, y=y, colname=colname)
        #         print(f"Ignored for {colname} = {ignored}")
        # stddevs[j] = np.std(np.abs(pdpy))
        print(colname, np.max(pdpx), np.min(pdpx))
        variations[j] = np.mean(np.abs(pdpy)) * (np.max(pdpx) - np.min(pdpx))
        total_dy_mass += variations[j]

    normalized_variations = variations / total_dy_mass

    I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': normalized_variations,
                           'AUC PDP':variations})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)

    return I


def plot_partials(X,y,eqn,yrange=(.5,1.5)):
    p = X.shape[1]
    fig, axes = plt.subplots(p, 1, figsize=(7, p*2+2))

    plt.suptitle('$'+eqn+'$', y=1.0)
    for j,colname in enumerate(X.columns):
        xj = X[colname]
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
            partial_dependence(X=X, y=y, colname=colname)
        # Plot dydx
        axes[j].scatter(pdpx, dydx, c='k', s=3)
        axes[j].plot([min(xj),max(xj)], [np.mean(dydx),np.mean(dydx)], c='orange')

        # Plot PD
        axes[j].plot(pdpx, pdpy, c='blue', lw=.5)

        axes[j].set_xlim(min(xj), max(xj))
        if yrange is not None:
            axes[j].set_ylim(*yrange)
        axes[j].set_xlabel(colname)
        axes[j].set_ylabel("y")
        axes[j].set_title(#(min(xj)+max(xj))/2, 1.4,
                     f"$\\sigma(dy)$={np.std(dydx):.3f}, $\\mu(pdpy)$={np.mean(pdpy):.3f}, $\\sigma(pdpy)$={np.std(pdpy):.3f}",
                     horizontalalignment='center')


p=5
df, coeff, eqn = synthetic_poly_data(1000,p)
print(df.describe())
#
X = df.drop('y', axis=1)
#X['noise'] = np.random.random_sample(size=len(X))
y = df['y']

I = impact_importances(X, y)
print(I)
plot_importances(I, imp_range=(0,1.0))
#                  #color='#FEE192', #'#5D9CD2', #'#A99EFF',
#                  # bgcolor='#F1F8FE'
#                  )

plot_partials(X, y, eqn, yrange=None)

#plot_marginal_dy(df)
#
plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)
plt.show()

