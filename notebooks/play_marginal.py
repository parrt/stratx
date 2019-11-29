from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.utils import resample
import matplotlib
import time
import timeit
from timeit import default_timer as timer
from stratx.partdep import *
#from stratx.cy_partdep import *
from rfpimp import plot_importances
import rfpimp
import shap
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype, is_bool_dtype

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

def df_string_to_cat(df:pd.DataFrame) -> dict:
    catencoders = {}
    for colname in df.columns:
        if is_string_dtype(df[colname]) or is_object_dtype(df[colname]):
            df[colname] = df[colname].astype('category').cat.as_ordered()
            catencoders[colname] = df[colname].cat.categories
    return catencoders


def df_cat_to_catcode(df):
    for col in df.columns:
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.codes + 1


def toy_weather_data():
    df_yr1 = toy_weather_data_()
    df_yr1['year'] = 1980
    df_yr2 = toy_weather_data_()
    df_yr2['year'] = 1981
    df_yr3 = toy_weather_data_()
    df_yr3['year'] = 1982
    df_raw = pd.concat([df_yr1, df_yr2, df_yr3], axis=0)
    df = df_raw.copy()
    return df


def toy_weather_data_():
    def temp(x): return np.sin((x+365/2)*(2*np.pi)/365)
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df = pd.DataFrame()
    df['dayofyear'] = range(1,365+1)
    df['state'] = np.random.choice(['CA','CO','AZ','WA'], len(df))
    df['temperature'] = temp(df['dayofyear'])
    df.loc[df['state']=='CA','temperature'] = 70 + df.loc[df['state']=='CA','temperature'] * noise('CA')
    df.loc[df['state']=='CO','temperature'] = 40 + df.loc[df['state']=='CO','temperature'] * noise('CO')
    df.loc[df['state']=='AZ','temperature'] = 90 + df.loc[df['state']=='AZ','temperature'] * noise('AZ')
    df.loc[df['state']=='WA','temperature'] = 60 + df.loc[df['state']=='WA','temperature'] * noise('WA')
    return df


def toy_weight_data(n):
    df = pd.DataFrame()
    nmen = n // 2
    nwomen = n // 2
    df['sex'] = ['M'] * nmen + ['F'] * nwomen
    df.loc[df['sex'] == 'F', 'pregnant'] = np.random.randint(0, 2, size=(nwomen,))
    df.loc[df['sex'] == 'M', 'pregnant'] = 0
    df.loc[df['sex'] == 'M', 'height'] = 5 * 12 + 8 + np.random.uniform(-7, +8,
                                                                        size=(nmen,))
    df.loc[df['sex'] == 'F', 'height'] = 5 * 12 + 5 + np.random.uniform(-4.5, +5,
                                                                        size=(nwomen,))
    df.loc[df['sex'] == 'M', 'education'] = 10 + np.random.randint(0, 8, size=nmen)
    df.loc[df['sex'] == 'F', 'education'] = 12 + np.random.randint(0, 8, size=nwomen)
    df['weight'] = 120 \
                   + (df['height'] - df['height'].min()) * 10 \
                   + df['pregnant'] * 30 \
                   - df['education'] * 1.5
    df['pregnant'] = df['pregnant'].astype(bool)
    df['education'] = df['education'].astype(int)
    eqn = "y = 120 + 10(x_{height} - min(x_{height})) + 30x_{pregnant} - 1.5x_{education}"
    return df, eqn


def synthetic_poly_data(n, p, noise=0):
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p) # get p random coefficients
    # coeff = np.array([1,1,1,1,1,1,1,1,1])
    coeff = np.array([1,1,1])
    exponents = np.array([1,1,1])
    for j in range(p):
        df[f'x{j+1}'] = np.round(np.random.random_sample(size=n)*5,1)

    if noise>0:
        for j in range(noise):
            df[f'noise{p+j+1}'] = np.round(np.random.random_sample(size=n)*1,1)

    # multiply coefficients x each column (var) and sum along columns
    yintercept = 10
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}']**exponents[i] for i in range(p)], axis=0 ) + yintercept

    # add hidden var
    #df['y'] += np.random.random_sample(size=n)*2-1

    terms = [f"{coeff[i]:.1f}x_{i+1}^{exponents[i]}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms)
    if noise>0:
        eqn += ' + '
        eqn += '+'.join([f'noise{p+j+1}' for j in range(noise)])
    return df, coeff, eqn


def synthetic_poly2dup_data(n, p):
    """
    SHAP seems to make x3, x1 very different despite same coeffs. over several runs,
    it varies a lot. e.g., i see one where x1,x2,x3 are same as they should be.
    very unstable. same with permutation. dropcol shows x1,x3 as 0.

    Adding noise so x3=x1+noise seems to overcome need for RF in stratpd and we
    get roughly equal imps.   SHAP still varies.
    """
    df = pd.DataFrame()
    # Add independent x variables in [0.0, 1.0)
    coeff = np.random.random_sample(size=p)*10 # get p random coefficients
    coeff = np.array([5, 3, 9])
    coeff = np.array([1,1,1])
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)*10,5)
    df['x3'] = df['x1'] + np.random.random_sample(size=n) # copy x1 into x3
    yintercept = 100
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    # df['y'] = 5*df['x1'] + 3*df['x2'] + 9*df['x3']
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms)
    return df, coeff, eqn+" where x_3 = x_1 + noise"



def shap_importances(rf, X):
    shap_values = shap.TreeExplainer(rf).shap_values(X)
    shapimp = np.mean(np.abs(shap_values), axis=0)

    total_imp = np.sum(shapimp)

    normalized_shap = shapimp / total_imp
    # print("SHAP", normalized_shap)
    shapI = pd.DataFrame(data={'Feature': X.columns, 'Importance': normalized_shap})
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


def compare_imp(rf, X, y, catcolnames=set(), eqn="n/a"):
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.2))

    I = impact_importances(X, y, catcolnames=catcolnames)
    plot_importances(I, imp_range=(0, 1), ax=axes[0])

    shap_I = shap_importances(rf, X)
    plot_importances(shap_I, ax=axes[1], imp_range=(0, 1))

    gini_I = ginidrop_importances(rf, X)
    plot_importances(gini_I, ax=axes[2])

    perm_I = rfpimp.importances(rf, X, y)
    plot_importances(perm_I, ax=axes[3])
    drop_I = rfpimp.dropcol_importances(rf, X, y)
    plot_importances(drop_I, ax=axes[4])

    axes[0].set_title("Impact Imp")
    axes[1].set_title("SHAP Imp")
    axes[2].set_title("ginidrop Imp")
    axes[3].set_title("Permute column")
    axes[4].set_title("Drop column")
    plt.suptitle(f"${eqn}$", y=1.02)


def impact_importances(X: pd.DataFrame,
                       y: pd.Series,
                       catcolnames=set(),
                       n_samples=None,  # use all by default
                       bootstrap_sampling=True,
                       n_trials:int=1,
                       n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                       verbose=False) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    if n_trees==1:
        bootstrap_sampling = False

    n,p = X.shape
    imps = np.zeros(shape=(p, n_trials)) # track p var importances for ntrials; cols are trials
    for i in range(n_trials):
        bootstrap_sample_idxs = resample(range(n), n_samples=n_samples, replace=bootstrap_sampling)
        X_, y_ = X.iloc[bootstrap_sample_idxs], y.iloc[bootstrap_sample_idxs]
        imps[:,i] = impact_importances_(X_, y_, catcolnames=catcolnames,
                                        n_trees=n_trees,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=bootstrap,
                                        max_features=max_features,
                                        verbose=verbose)

    avg_imps = np.mean(imps, axis=1)
    stddev_imps = np.std(imps, axis=1)

    I = pd.DataFrame(data={'Feature': X.columns,
                           'Importance': avg_imps,
                           "Sigma":stddev_imps})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)

    return I


def impact_importances_(X: pd.DataFrame, y: pd.Series, catcolnames=set(),
                        n_trees=1, min_samples_leaf=10, bootstrap=False, max_features=1.0,
                        verbose=False) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Can only operate on dataframes at the moment")

    p = X.shape[1]
    avg_pdp = np.zeros(shape=(p,)) # track avg pdp, not magnitude
    avg_abs_pdp = np.zeros(shape=(p,)) # like area under PDP curve but not including width
    total_avg_pdpy = 0.0
    for j, colname in enumerate(X.columns):
        # Ask stratx package for the partial dependence of y with respect to X[colname]
        if colname in catcolnames:
            start = time.time()
            leaf_histos, avg_per_cat, ignored = \
                cat_partial_dependence(X, y, colname=colname,
                                       ntrees=n_trees,
                                       min_samples_leaf=min_samples_leaf,
                                       bootstrap=bootstrap,
                                       max_features=max_features,
                                       verbose=verbose)
            #         print(f"Ignored for {colname} = {ignored}")
            stop = time.time()
            # print(f"PD time {(stop - start) * 1000:.0f}ms")
            min_avg_value = np.nanmin(avg_per_cat)
            avg_per_cat_from_0 = avg_per_cat - min_avg_value # all positive now, relative to 0 for lowest cat
            # some cats have NaN, such as 0th which is for "missing values"
            avg_abs_pdp[j] = np.nanmean(avg_per_cat_from_0)# * (ncats - 1)
            avg_pdp[j] = np.mean(avg_per_cat_from_0)
            total_avg_pdpy += avg_abs_pdp[j]
        else:
            start = time.time()
            leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
                partial_dependence(X=X, y=y, colname=colname,
                                   ntrees=n_trees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features,
                                   verbose=verbose)
        #         print(f"Ignored for {colname} = {ignored}")
            stop = time.time()
            # print(f"PD time {(stop-start)*1000:.0f}ms")
            avg_abs_pdp[j] = np.mean(np.abs(pdpy))# * (np.max(pdpx) - np.min(pdpx))
            avg_pdp[j] = np.mean(pdpy)
            total_avg_pdpy += avg_abs_pdp[j]

    # print("avg_pdp", avg_pdp, "sum", np.sum(avg_pdp), "avg y", np.mean(y), "avg y-min(y)", np.mean(y)-np.min(y))
    normalized_importances = avg_abs_pdp / total_avg_pdpy

    return normalized_importances


def plot_partials(X,y,eqn, yrange=(.5,1.5)):
    p = X.shape[1]
    fig, axes = plt.subplots(p, 1, figsize=(7, p*2+2))

    plt.suptitle('$'+eqn+'$', y=1.02)
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


def weight():
    df, eqn = toy_weight_data(2000)
    X = df.drop('weight', axis=1)
    y = df['weight']
    X['pregnant'] = X['pregnant'].astype(int)
    X['sex'] = X['sex'].map({'M': 0, 'F': 1}).astype(int)
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']

    I = impact_importances(X, y, catcolnames={'sex', 'pregnant'})
    print(X.columns)
    print(I)
    plot_importances(I, imp_range=(0, 1.0))

    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X,y)

    compare_imp(rf, X, y, catcolnames={'sex','pregnant'}, eqn=eqn)


def weather():
    df = toy_weather_data()
    df_string_to_cat(df)
    df_cat_to_catcode(df)

    X = df.drop('temperature', axis=1)
    y = df['temperature']
    I = impact_importances(X, y, catcolnames={'state'})
    print(X.columns)
    print(I)
    plot_importances(I, imp_range=(0,1.0))

    rf = RandomForestRegressor(n_estimators=10)
    rf.fit(X,y)

    compare_imp(rf, X, y, catcolnames={'state'})


def poly():
    p=3
    df, coeff, eqn = synthetic_poly_data(1000,p,noise=1)
    #df, coeff, eqn = synthetic_poly2dup_data(2000,p)
    X = df.drop('y', axis=1)
    y = df['y']
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']

    I = impact_importances(X, y, n_samples=None)
    print(I)
    # I = impact_importances(X, y, n_samples=100, bootstrap=False, n_trials=25)
    # print(I)
    plot_importances(I, imp_range=(0,1.0))
    plt.suptitle(f"${eqn}$", y=1.02)

    #compare_imp(rf, X, y, catcolnames={'sex','pregnant'}, eqn=eqn)

    #plot_partials(X, y, eqn, yrange=None)
    # plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)


def poly_dupcol():
    p=3
    df, coeff, eqn = synthetic_poly2dup_data(1000,p)
    X = df.drop('y', axis=1)
    y = df['y']
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']

    I = impact_importances(X, y)
                           # ntrees=5, min_samples_leaf=10, bootstrap=False, max_features=1)
    print(I)
    # I = impact_importances(X, y, n_samples=100, bootstrap=False, n_trials=25)
    # print(I)
    plot_importances(I, imp_range=(0,1.0))


def unstable_SHAP():
    p=2 # p=2 and we're pretty stable and about even; p=2 implies don't add noisy x3 to y; hmm... shap seems stable but not even
        # damn, tried again and p=2 is the most unstable
        # bumping trees to 20 made shap more stable. both of us get x2 right but x1,x3
        # pair bounces quite a bit in shap, less so in ours but some. x3 is stealing importance
        # from x1 but x1 is more important as it's in the y equation.
    #p=3 # p=3 and x3 not added to y; ours has stairstep but stable, shap same order but can vary alot

    nplots=7
    fig, axes = plt.subplots(nplots, 2, figsize=(6, nplots+3))

    imps = np.zeros(shape=(p+1, nplots))
    shap_imps = np.zeros(shape=(p+1, nplots))

    for i in range(nplots):
        # make new data set each time
        df, coeff, eqn = synthetic_poly2dup_data(1000, p)
        X = df.drop('y', axis=1)
        y = df['y']

        print(eqn)
        # pass copies to ensure we don't alter them, giving fresh stuff to SHAP
        I = impact_importances(X.copy(), y.copy(), n_trials=1)
                               # ntrees=10, min_samples_leaf=10, bootstrap=False, max_features=1)
        imps[:,i] = I['Importance'].values
        plot_importances(I, imp_range=(0,1.0), ax=axes[i][0])
        axes[i][0].set_title("Impact", fontsize=9)

        rf = RandomForestRegressor(n_estimators=20, oob_score=True)
        rf.fit(X,y)
        shap_I = shap_importances(rf, X)
        shap_imps[:,i] = shap_I['Importance'].values
        plot_importances(shap_I, ax=axes[i][1], imp_range=(0, 1))
        axes[i][1].set_title(f"SHAP (OOB $R^2$ {rf.oob_score_:.2f})", fontsize=9)

    plt.suptitle(f"${eqn}$", y=1.0)

    stds = np.std(imps, axis=1)
    shap_stds = np.std(shap_imps, axis=1)
    print(stds)
    print(shap_stds)

    #compare_imp(rf, X, y, catcolnames={'sex','pregnant'}, eqn=eqn)

    #plot_partials(X, y, eqn, yrange=None)
    # plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)


def time_SHAP(n_estimators, n_records):
    df = pd.read_feather("../notebooks/data/bulldozer-train.feather")
    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    # basefeatures = ['ModelID', 'YearMade', 'MachineHours']
    basefeatures = ['SalesID', 'MachineID', 'ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHoursCurrentMeter']

    # Get subsample; it's a (sorted) timeseries so get last records not random
    df = df.iloc[-n_records:]  # take only last records
    X, y = df[basefeatures], df['SalePrice']
    X = X.fillna(0)  # flip missing numeric values to zeros
    start = time.time()
    rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
    rf.fit(X, y)
    stop = time.time()
    print(f"RF fit time for {n_records} = {(stop - start):.1f}s")
    start = timer()
    shap_I = shap_importances(rf, X)
    stop = timer()
    print(f"SHAP OOB $R^2$ {rf.oob_score_:.2f}, time for {n_records} = {(stop - start):.1f}s")
    return shap_I


def our_bulldozer():
    df = pd.read_feather("../notebooks/data/bulldozer-train.feather")
    df['MachineHours'] = df['MachineHoursCurrentMeter']  # shorten name
    # basefeatures = ['ModelID', 'YearMade', 'MachineHours']
    basefeatures = ['SalesID', 'MachineID', 'ModelID',
                    'datasource', 'YearMade',
                    # some missing values but use anyway:
                    'auctioneerID', 'MachineHoursCurrentMeter']

    # Get subsample; it's a (sorted) timeseries so get last records not random
    n_records = 10_000
    df = df.iloc[-n_records:]  # take only last records
    X, y = df[basefeatures], df['SalePrice']
    X = X.fillna(0)  # flip missing numeric values to zeros
    start = timer()
    I = impact_importances(X, y, verbose=True)#, n_samples=3000, bootstrap_sampling=False, n_trials=10)#, catcolnames={'ModelID','SalesID','auctioneerID','MachineID','datasource'})
    print(I)
    stop = timer()
    print(f"Time for {n_records} = {(stop - start):.1f}s")
    plot_importances(I, imp_range=(0, 1))


def bulldozer():
    n_estimators=50
    n_records=2000
    I = time_SHAP(n_estimators, n_records)
    plot_importances(I, imp_range=(0, 1))


def speed_SHAP():
    n_estimators = 50
    print("n_estimators",n_estimators)
    for n_records in [1000, 3000, 5000, 8000]:
        time_SHAP(n_estimators, n_records)


#weather()
#poly()
# unstable_SHAP()
# poly_dupcol()
# speed_SHAP()
# bulldozer()

our_bulldozer()


#plt.tight_layout()
#plt.subplots_adjust(top=0.85) # must be after tight_layout
plt.show()
