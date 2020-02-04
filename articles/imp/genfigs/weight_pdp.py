from support import *
import numpy as np
import shap

"""
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
                 pdp_marker_size=.01,
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


def play():
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

    # shap.dependence_plot("height", shap_values, X[:shap_test_size],
    #                      interaction_index=None)
    shap.dependence_plot("height", shap_values, X[:shap_test_size],
                         interaction_index=None, dot_size=5,
                         show=False, alpha=.5)
"""

def combined(feature_perturbation, twin=False):
    n = 2000
    shap_test_size = 2000
    X, y, df, eqn = toy_weight_data(n)
    X = df.drop('weight', axis=1)
    y = df['weight']

    rf = RandomForestRegressor(n_estimators=40, oob_score=True, n_jobs=-1)
    rf.fit(X,y)

    if feature_perturbation=='interventional':
        explainer = shap.TreeExplainer(rf, data=shap.sample(X, 500), feature_perturbation='interventional')
        xlabel = "height\n(b)"
    else:
        explainer = shap.TreeExplainer(rf, feature_perturbation='tree_path_dependent')
        xlabel = "height\n(a)"
    shap_sample = X[:shap_test_size]
    shap_values = explainer.shap_values(shap_sample, check_additivity=False)

    GREY = '#444443'
    fig, ax = plt.subplots(1, 1, figsize=(3.8,3.2))

    shap.dependence_plot("height", shap_values, shap_sample,
                         interaction_index=None, ax=ax, dot_size=5,
                         show=False, alpha=1)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['top'].set_linewidth(.5)

    ax.set_ylabel("Impact on weight\n(height SHAP)", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.plot([70,70], [-75,75], '--', lw=.6, color=GREY)
    ax.text(69.8,60, "Max female height", horizontalalignment='right',
            fontsize=9)

    leaf_xranges, leaf_slopes, slope_counts_at_x, dx, slope_at_x, pdpx, pdpy, ignored = \
        partial_dependence(X=X, y=y, colname='height')

    ax.set_ylim(-77,75)
    # ax.set_xlim(min(pdpx), max(pdpx))
    ax.set_xticks([60,65,70,75])
    ax.set_yticks([-75,-60,-40,-20,0,20,40,60,75])

    ax.set_title(f"SHAP {feature_perturbation}", fontsize=12)
    # ax.set_ylim(-40,70)

    print(min(pdpx), max(pdpx))
    print(min(pdpy), max(pdpy))
    rise = max(pdpy) - min(pdpy)
    run = max(pdpx) - min(pdpx)
    slope = rise/run
    print(slope)
    # ax.plot([min(pdpx),max(pdpyX['height'])], [0,]

    if twin:
        ax2 = ax.twinx()
        # ax2.set_xlim(min(pdpx), max(pdpx))
        ax2.set_ylim(min(pdpy)-5, max(pdpy)+5)
        ax2.set_xticks([60,65,70,75])
        ax2.set_yticks([0,20,40,60,80,100,120,140,150])
        ax2.set_ylabel("height", fontsize=12)

        #too hard to see, leave out
        ax2.plot(pdpx, pdpy, # shift y down a bit to make it visible
                 '-', lw=.9,
                 markersize=1, c='k')
        # ax2.text(65,25, f"StratPD slope = {slope:.1f}")
        ax2.annotate(f"StratPD (slope={slope:.1f})", (64.65,39), xytext=(66,18),
                     horizontalalignment='left',
                     arrowprops=dict(facecolor='black', width=.5, headwidth=5, headlength=5),
                     fontsize=9)

    plt.tight_layout()
    plt.savefig(f"../images/weight-shap-{feature_perturbation}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

# weight()
combined('tree_path_dependent', twin=True)
combined('interventional', twin=True)
