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
    coeff = np.array([1,3,5,8])
    # coeff = np.array([5,10])
    for i in range(p):
        df[f'x{i+1}'] = np.round(np.random.random_sample(size=n)+2,1) # shift x_i to right 2
    #df['x3'] = df['x1']+np.random.random_sample(size=n)*2 # copy x1 + noise
    # multiply coefficients x each column (var) and sum along columns
    yintercept = 10
    df['y'] = np.sum( [coeff[i]*df[f'x{i+1}'] for i in range(p)], axis=0 ) + yintercept
    #TODO add noise
    terms = [f"{coeff[i]:.1f}x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + '+'.join(terms)
    return df, coeff, eqn


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
        leaf_xranges, leaf_slopes, dx, dydx, pdpx, pdpy, ignored = \
            partial_dependence(X=df.drop('y', axis=1), y=y, colname=varname)
        axes[j+1].plot(pdpx, pdpy,lw=.5, c='k')  # Plot PD_j
        axes[j+1].text(max(pdpx), pdpy[-1], f"$PD_{j+1}$")
        x_mass = np.mean(np.abs(pdpy))
        dy_mass = np.mean(np.abs(dydx))# * (np.max(xj) - np.min(xj))
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
X = df.drop('y', axis=1)
y = df['y']

I = pd_importances(X, y)
plot_importances(I, imp_range=(0,1.0),
                 #color='#FEE192', #'#5D9CD2', #'#A99EFF',
                 # bgcolor='#F1F8FE'
                 )

plot_marginal_dy(df)
#
# plt.tight_layout()
plt.savefig("/Users/parrt/Desktop/marginal.pdf", bbox_inches=0)
plt.show()
