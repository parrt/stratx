from support import *
from stratx.partdep import *

np.set_printoptions(precision=2, suppress=True, linewidth=300, threshold=2000)


def toy_weather_data(n = 1000, p=50, n_outliers=None):
    """
    For each state, create a (fictional) ramp of data from day 1 to 365 so mean is not
    0, as we'd get from a sinusoid.
    """
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df_avgs = pd.read_csv("../../../testing/state_avgtemp.csv")
    avgtemp = df_avgs['avgtemp']

    df = pd.DataFrame()
    df['dayofyear'] = np.random.randint(1, 365 + 1, size=n)
    df['state'] = np.random.randint(0, p, size=n) # get only p states
    df['temp'] = .1 * df['dayofyear'] + avgtemp.iloc[df['state']].values

    if n_outliers>0:
        # bump up or down some y to create outliers
        outliers_idx = np.random.randint(0, len(df), size=n_outliers)
        outliers_vals = np.random.normal(loc=0, scale=15, size=n_outliers)
        df.iloc[outliers_idx, 2] += outliers_vals

    avgtemp = df_avgs.iloc[np.unique(df['state'])]['avgtemp']
    print("avg of states' avg temps:", np.mean(avgtemp))
    true_impact = np.mean(avgtemp) - np.min(avgtemp)
    print("avg of states' avg temps minus min:", true_impact)

    X = df.drop('temp', axis=1)
    y = df['temp']
    return X, y, df_avgs['state'].values, df_avgs.iloc[0:p], true_impact


def viz_weather(n, p, min_samples_leaf, n_outliers=0, seed=None, show_truth=True):
    if seed is not None:
        save_state = np.random.get_state()
        np.random.seed(seed)

    X, y, catnames, avgtemps, true_impact = \
        toy_weather_data(n=n, p=p, n_outliers=n_outliers)
    y_bar = np.mean(y)
    print("overall mean(y)", y_bar)

    #title = f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9

    fig,ax = plt.subplots(1,1, figsize=(10,3))
    uniq_catcodes, avg_per_cat, ignored, merge_ignored = \
        plot_catstratpd(X, y, colname='state', targetname="temp",
                        catnames=catnames,
                        n_trials=1,
                        min_samples_leaf=min_samples_leaf,
                        show_x_counts=False,
                        ticklabel_fontsize=6,
                        pdp_marker_size=10,
                        yrange=(-20,50),
                        # min_y_shifted_to_zero=False,
                        min_y_shifted_to_zero=True,
                        ax=ax,
                        verbose=False,
                        # title=
                        )
    for i,(state,t) in enumerate(zip(catnames,avgtemps['avgtemp'].values)):
        ax.text(i,45,f"{t:.1f}", horizontalalignment="center", fontsize=9)
        # catnames[i] += f"\n{t:.1f}"

    print("ignored", ignored, "merge_ignored", merge_ignored)
    if seed is not None:
        np.random.set_state(save_state)

    computed_impact = featimp.avg_all_pairs_abs_delta(avg_per_cat)
    title = f"strat ignored={ignored}, merge ignored = {merge_ignored}\nimpact={computed_impact:.1f}, true_impact={true_impact:.1f}"
    plt.title(title)
    plt.tight_layout()
    plt.show()

# viz_weather(1000, p=30, min_samples_leaf=5, seed=1, n_outliers=0)
# viz_weather(1000, p=30, min_samples_leaf=3, seed=1, n_outliers=0)
# viz_weather(1000, p=30, min_samples_leaf=2, seed=1, n_outliers=0)

viz_weather(1000, p=30, min_samples_leaf=15, seed=1, n_outliers=10)
viz_weather(1000, p=30, min_samples_leaf=15, seed=2, n_outliers=10)
viz_weather(1000, p=30, min_samples_leaf=15, seed=3, n_outliers=10)
viz_weather(1000, p=30, min_samples_leaf=15, seed=4, n_outliers=10)