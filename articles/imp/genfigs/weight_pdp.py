from support import *
import numpy as np
import shap

def weight():
    X, y, df, eqn = toy_weight_data(2000)
    X = df.drop('weight', axis=1)
    y = df['weight']
    print(X.head())

    print(eqn)
    # X = df.drop('y', axis=1)
    # #X['noise'] = np.random.random_sample(size=len(X))
    # y = df['y']
    fig, ax = plt.subplots(1, 1, figsize=(3.9,3.2))

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    plot_stratpd(X, y, colname='height', targetname='weight',
                 show_slope_lines=False,
                 show_x_counts=False,
                 show_impact=False,
                 pdp_marker_size=.5,
                 yrange=(-5,155),
                 xrange=(60,76),
                 ax=ax)

    ax.set_ylabel("weight (lbs)", fontsize=13)
    ax.set_xlabel("height", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # yrange=(-10,100))
    plt.tight_layout()
    plt.savefig("../images/weight-stratpd.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    # catcolnames = {'sex', 'pregnant'},
    # I = importances(X, y, catcolnames={'sex', 'pregnant'})
    # print(I)


def shap_weight():
    n = 2000
    shap_test_size = 2000
    X, y, df, eqn = toy_weight_data(n)
    X = df.drop('weight', axis=1)
    y = df['weight']

    rf = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=-1)
    rf.fit(X,y)
    # print("OOB", rf.oob_score_)

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
    shap_values = explainer.shap_values(X[:shap_test_size], check_additivity=False)

    print(f"n={n}, {eqn}, avg={np.mean(y)}")
    shapimp = np.mean(np.abs(shap_values), axis=0)
    s = np.sum(shapimp)
    print("\nRF SHAP importances", list(shapimp), list(shapimp / s))

    shap.dependence_plot("height", shap_values, X[:shap_test_size], interaction_index=None)


def combined():
    n = 2000
    shap_test_size = 2000
    X, y, df, eqn = toy_weight_data(n)
    X = df.drop('weight', axis=1)
    y = df['weight']

    rf = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=-1)
    rf.fit(X,y)

    explainer = shap.TreeExplainer(rf, data=shap.sample(X, 100), feature_perturbation='interventional')
    shap_values = explainer.shap_values(X[:shap_test_size], check_additivity=False)

    leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
        partial_dependence(X=X, y=y, colname='height')

    GREY = '#444443'
    fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))


    # height_col_i = 2
    # ax.plot(X['height'][:shap_test_size], shap_values[:,height_col_i],
    #         'o',
    #         markersize=5, color=GREY, fillstyle='none',
    #         markeredgewidth=.5, alpha=.5)

    shap.dependence_plot("height", shap_values, X[:shap_test_size],
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=.5)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_ylabel("Impact on weight\n(height SHAP)")
    ax.set_xlabel("height")
    ax.tick_params(axis='both', which='major', labelsize=10)

    # ax.set_ylim(-40,70)

    # rise = max(pdpy) - min(pdpy)
    # run = max(pdpx) - min(pdpx)
    # slope = rise/run
    # ax.plot([min(pdpx),max(pdpyX['height'])], [0,]

    # ax2 = ax.twinx()
    # ax2.plot(pdpx, pdpy-5, # shift y down a bit to make it visible
    #          '-', lw=.5,
    #          markersize=1, c='k')
    # ax2.set_ylim(min(pdpy), max(pdpy))

    # too hard to see, leave out
    # ax.plot(pdpx, pdpy-np.mean(pdpy), # shift y down a bit to make it visible
    #          '-', lw=.5,
    #          markersize=1, c='k')

    plt.tight_layout()
    plt.savefig("../images/weight-shap.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

weight()
combined()
# shap_weight()
