import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib
# matplotlib.use('TkAgg') # separate window

from stratx import featimp
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
    coeff = np.array([1,3,5,8])
    # coeff = np.array([5,10])
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)*10+2,1) # shift x_i to right 2
    #df['x3'] = df['x1']+np.random.random_sample(size=n)*2 # copy x1 + noise
    # multiply coefficients x each column (var) and sum along columns
    yintercept = 0
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    #TODO add noise
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + '+'.join(terms)
    return df, coeff, eqn


def partial_derivative(X, y, colname,
                       ntrees=1, min_samples_leaf=10, bootstrap=False,
                       max_features=1.0,
                       supervised=True,
                       verbose=False):
    if supervised:
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features)
        rf.fit(X.drop(colname, axis=1), y)
        if verbose:
            print(f"Strat Partition RF: missing {colname} training R^2 {rf.score(X.drop(colname, axis=1), y)}")

    else:
        """
        Wow. Breiman's trick works in most cases. Falls apart on Boston housing MEDV target vs AGE
        """
        if verbose: print("USING UNSUPERVISED MODE")
        X_synth, y_synth = conjure_twoclass(X)
        rf = RandomForestRegressor(n_estimators=ntrees,
                                   min_samples_leaf=min_samples_leaf,
                                   bootstrap=bootstrap,
                                   max_features=max_features,
                                   oob_score=False)
        rf.fit(X_synth.drop(colname, axis=1), y_synth)

    if verbose:
        leaves = leaf_samples(rf, X.drop(colname, axis=1))
        nnodes = rf.estimators_[0].tree_.node_count
        print(f"Partitioning 'x not {colname}': {nnodes} nodes in (first) tree, "
              f"{len(rf.estimators_)} trees, {len(leaves)} total leaves")

    leaf_xranges, leaf_sizes, leaf_slopes, ignored = \
        collect_discrete_slopes(rf, X, y, colname)

    # print('leaf_xranges', leaf_xranges)
    # print('leaf_slopes', leaf_slopes)

    real_uniq_x = np.array(sorted(np.unique(X[colname])))
    if verbose:
        print(f"discrete StratPD num samples ignored {ignored}/{len(X)} for {colname}")

    slope_at_x = avg_values_at_x(real_uniq_x, leaf_xranges, leaf_slopes)
    # slope_at_x = weighted_avg_values_at_x(real_uniq_x, leaf_xranges, leaf_slopes, leaf_sizes, use_weighted_avg=True)

    # Drop any nan slopes; implies we have no reliable data for that range
    # Make sure to drop uniq_x values too :)
    notnan_idx = ~np.isnan(slope_at_x) # should be same for slope_at_x
    slope_at_x = slope_at_x[notnan_idx]
    pdpx = real_uniq_x[notnan_idx]

    dx = np.diff(pdpx)
    y_deltas = slope_at_x[:-1] * dx  # last slope is nan since no data after last x value

    # print(f"y_deltas: {y_deltas}")
    return leaf_xranges, leaf_slopes, pdpx, dx, y_deltas, ignored


def PD(X, y, colname,
       ntrees=1, min_samples_leaf=10, bootstrap=False,
       max_features=1.0,
       supervised=True,
       verbose=False):
    leaf_xranges, leaf_slopes, pdpx, dx, y_deltas, ignored = \
        partial_derivative(**locals())            # pass all args to other function
    pdpy = np.cumsum(y_deltas)                    # we lose one value here
    pdpy = np.concatenate([np.array([0]), pdpy])  # add back the 0 we lost
    return leaf_xranges, leaf_slopes, pdpx, pdpy, dx, y_deltas, ignored


def plot_marginal_dy(df):
    fig, axes = plt.subplots(p+1,1,figsize=(7,7))


    plt.suptitle('$'+eqn+'$', y=1.0)
    y = df['y']
    axes[0].scatter(range(len(df)), y, s=3)
    axes[0].set_xlabel("$i$")
    axes[0].set_ylabel("$y$")
    dy = np.diff(y)
    dy = np.concatenate([np.array([0]), dy])  # add back the 0 we lost
    axes[0].plot(range(len(df)), dy, lw=.5, c='orange')
    axes[0].text(len(df), 0, f"$\\frac{{\\partial y}}{{i}}$")
    my = np.mean(np.abs(dy))
    axes[0].set_title(f"mean $|\\frac{{\\partial y}}{{i}}|$ = {my:.2f}, $\\overline{{|y|}}$ = {np.mean(np.abs(y)):.2f}, $\\overline{{y-min(y)}}$ = {np.mean(y - min(y)):.1f}")

    variations = np.zeros(shape=(p,))
    total_mean_not_xj = 0.0
    total_dy_mass = 0.0
    for j in range(p):
        # axes[j+1].set_title(f"$\\overline{{\\partial y/x_{j+1}}}$")
        varname = f'x{j+1}'
        axes[j+1].scatter(df[varname], y, s=3) # Plot marginal y
        axes[j+1].set_xlabel(f'$x_{j+1}$')
        df_by_xj = df.groupby(by=varname).mean().reset_index()
        uniq_x = df_by_xj[varname]
        avg_y = df_by_xj['y']
        axes[j+1].plot(uniq_x, avg_y, lw=.5) # Plot avg y vs uniq x
        df_sorted_by_xj = df_by_xj.sort_values(by=varname)

        # Plot dy/dx_j
        xj = df_sorted_by_xj[varname]
        dy = np.diff(df_sorted_by_xj['y'])

        dx = np.diff(xj)
        dydx = dy / dx

        segments = []
        for i,delta in enumerate(dy):# / np.std(y)):
            one_line = [(xj[i],0), (xj[i+1],delta)]
            segments.append( one_line )
        lines = LineCollection(segments, linewidths=.5, color='orange')
        axes[j+1].add_collection(lines)
        #axes[j+1].set_ylim(np.min(dy),np.max(y))
        # axes[j+1].plot(df_sorted_by_xj[varname][:-1], dy, lw=.5, c='orange')
        axes[j+1].text(max(xj), 0, f"$\\frac{{\\partial y}}{{x_{j+1}}}$")

        # Draw partial derivative of partial dependence
        leaf_xranges, leaf_slopes, pdpx, pdpy, dx, y_deltas, ignored = \
            PD(X=df.drop('y', axis=1), y=y, colname=varname)
        axes[j+1].plot(pdpx, pdpy,lw=.5, c='k')  # Plot PD_j
        axes[j+1].text(max(pdpx), pdpy[-1], f"$PD_{j+1}$")
        x_mass = np.mean(np.abs(pdpy))
        dy_mass = np.mean(np.abs(y_deltas)) * (np.max(xj) - np.min(xj))
        variations[j] = dy_mass
        total_dy_mass += dy_mass
        mean_not_xj = np.mean(np.abs(dy))# * (np.max(xj) - np.min(xj))
        total_mean_not_xj += mean_not_xj

        not_xj = set(range(1,p+1)) - {j+1}
        not_xj = ','.join([f"x_{j}" for j in not_xj])
        axes[j+1].set_title(f"mean $|\\frac{{\\partial y}}{{{not_xj}}}|$ = {mean_not_xj:.2f}, pd mass {x_mass:.2f}, partial mass {dy_mass:.3f}")
        axes[j+1].set_ylabel("$y$")

    print(f"Total means not x_j {total_mean_not_xj:.2f}, total dy mass {total_dy_mass:.2f}")
    print("Variations", variations/total_dy_mass)

p=3
df, coeff, eqn = synthetic_poly_data(1000,p)
print(df.head(3))
#
# X = df.drop('y', axis=1)
# y = df['y']

plot_marginal_dy(df)

plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)
plt.show()