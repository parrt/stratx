import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils import resample
import matplotlib
import time
from stratx.partdep import *
from rfpimp import plot_importances
import rfpimp
import shap

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


def shap_importances(rf, X):
    shap_values = shap.TreeExplainer(rf).shap_values(X)
    shapimp = np.mean(np.abs(shap_values), axis=0)
    print(shapimp)
    shapI = pd.DataFrame(data={'Feature': X.columns, 'Importance': shapimp})
    shapI = shapI.set_index('Feature')
    shapI = shapI.sort_values('Importance', ascending=False)
    # plot_importances(shapI)
    return shapI


def ginidrop_importances(rf, X):
    ginidrop_I = rf.feature_importances_
    ginidrop_I = pd.DataFrame(data={'Feature': X.columns, 'Importance': ginidrop_I})
    ginidrop_I = ginidrop_I.set_index('Feature')
    ginidrop_I = ginidrop_I.sort_values('Importance', ascending=False)
    return ginidrop_I


def compare_imp(rf, X, y, eqn):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))

    I = impact_importances(X, y)
    plot_importances(I, imp_range=(0, 1), ax=axes[0])

    shap_I = shap_importances(rf, X)
    plot_importances(shap_I, ax=axes[1])

    gini_I = ginidrop_importances(rf, X)
    plot_importances(gini_I, ax=axes[2])

    perm_I = rfpimp.importances(rf, X, y)
    plot_importances(perm_I, ax=axes[3])
    drop_I = rfpimp.dropcol_importances(rf, X, y)
    plot_importances(drop_I, ax=axes[4])

    axes[0].set_title("Strat Imp")
    axes[1].set_title("SHAP Imp")
    axes[2].set_title("ginidrop Imp")
    axes[3].set_title("Permute column")
    axes[4].set_title("Drop column")
    plt.suptitle(f"${eqn}$")


def synthetic_poly_data(n, p, noise=0):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p)*1 # get p random coefficients
    coeff = np.array([1,1,1,1,1,1,1,1,1])
    # coeff = np.array([5,10])
    for j in range(p):
        df[f'x{j+1}'] = np.round(np.random.random_sample(size=n)*1,1)
    if noise>0:
        for j in range(noise):
            df[f'noise{p+j+1}'] = np.round(np.random.random_sample(size=n)*1,1)

    #df['x3'] = df['x1']*np.round(np.random.random_sample(size=n),1) # copy x1 + noise
    # multiply coefficients x each column (var) and sum along columns
    yintercept = 10
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + '+'.join(terms)
    if noise>0:
        eqn += ' + '
        eqn += '+'.join([f'noise{p+j+1}' for j in range(noise)])
    return df, coeff, eqn


def impact_importances(X: pd.DataFrame,
                       y: pd.Series,
                       n_samples=None,  # use all by default
                       n_trials:int=1) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    n,p = X.shape
    imps = np.zeros(shape=(p, n_trials)) # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        bootstrap_sample_idxs = resample(range(n), n_samples=n_samples, replace=True)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        imps[:,i] = impact_importances_(X_, y_)

    avg_imps = np.mean(imps, axis=1)
    stddev_imps = np.std(imps, axis=1)

    I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': avg_imps,
                           "2Sigma":2*stddev_imps})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)

    return I


def impact_importances_(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    p = X.shape[1]
    auc_pdp = np.zeros(shape=(p,)) # area under PDP curve
    total_pdpy_mass = 0.0
    for j, colname in enumerate(X.columns):
        # Ask stratx package for the partial dependence of y with respect to X[colname]
        start = time.time()
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
            partial_dependence(X=X, y=y, colname=colname)
        #         print(f"Ignored for {colname} = {ignored}")
        stop = time.time()
        print(f"PD time {(stop-start)*1000:.0f}ms")
        auc_pdp[j] = np.mean(np.abs(pdpy)) * (np.max(pdpx) - np.min(pdpx))
        total_pdpy_mass += auc_pdp[j]

    normalized_auc_pdp = auc_pdp / total_pdpy_mass

    return normalized_auc_pdp
    # I = pd.DataFrame(data={'Feature': X.columns,
    #                        'Importance': normalized_auc_pdp,
    #                        'AUC PDP':auc_pdp})
    # I = I.set_index('Feature')
    # I = I.sort_values('Importance', ascending=False)
    #
    # return I


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


p=3
df, coeff, eqn = synthetic_poly_data(1000,p,noise=1)
print(eqn)
X = df.drop('y', axis=1)
#X['noise'] = np.random.random_sample(size=len(X))
y = df['y']

# I = impact_importances(X, y, n_samples=500, n_trials=5)
# print(I)
# plot_importances(I, imp_range=(0,1.0))

#                  #color='#FEE192', #'#5D9CD2', #'#A99EFF',
#                  # bgcolor='#F1F8FE'
#                  )

X = df.drop('y', axis=1)
y = df['y']
#X = featimp.standardize(X)

rf = RandomForestRegressor(n_estimators=10)
rf.fit(X,y)

compare_imp(rf, X, y, eqn)

#plot_partials(X, y, eqn, yrange=None)
plt.tight_layout()
# plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)
plt.show()

