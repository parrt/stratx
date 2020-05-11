from support import *

from stratx.partdep import *

np.set_printoptions(precision=2, suppress=True, linewidth=300, threshold=2000)


def toy_weather_data_foo(n = 1000, p=50, n_outliers=None):
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


def toy_weather_data_1yr(p):
    # def temp(x): return 10*np.sin((2*x + 365) * (np.pi) / 365)
    def temp(x): return 10*np.sin(((2/365)*np.pi*x + np.pi))

    def noise(state):
        # noise_per_state = {'CA':2, 'CO':4, 'AZ':7, 'WA':2, 'NV':5}
        return np.random.normal(0, 4, sum(df['state'] == state))

    df = pd.DataFrame()
    df['dayofyear'] = range(1, 365 + 1)
    states = ['CA', 'CO', 'AZ', 'WA', 'NV']
    states = states[:p]
    df['state'] = np.random.choice(states, len(df))
    df['temperature'] = temp(df['dayofyear'])
    if p>=1:
        df.loc[df['state'] == 'CA', 'temperature'] = \
            70 + df.loc[df['state'] == 'CA', 'temperature'] + noise('CA')
    if p>=2:
        df.loc[df['state'] == 'CO', 'temperature'] = \
            40 + df.loc[df['state'] == 'CO', 'temperature'] + noise('CO')
    if p>=3:
        df.loc[df['state'] == 'AZ', 'temperature'] = \
            90 + df.loc[df['state'] == 'AZ', 'temperature'] + noise('AZ')
    if p>=4:
        df.loc[df['state'] == 'WA', 'temperature'] = \
            60 + df.loc[df['state'] == 'WA', 'temperature'] + noise('WA')
    if p>=5:
        df.loc[df['state'] == 'NV', 'temperature'] = \
            80 + df.loc[df['state'] == 'NV', 'temperature'] + noise('NV')

    return df


def toy_weather_data(p):
    df_yr1 = toy_weather_data_1yr(p)
    df_yr1['year'] = 1980
    df_yr2 = toy_weather_data_1yr(p)
    df_yr2['year'] = 1981
    df_yr3 = toy_weather_data_1yr(p)
    df_yr3['year'] = 1982
    df_raw = pd.concat([df_yr1, df_yr2, df_yr3], axis=0)
    df = df_raw.copy()
    return df


def viz_weather(n, p, min_samples_leaf, n_outliers=0, seed=None, show_truth=True):
    if seed is not None:
        save_state = np.random.get_state()
        np.random.seed(seed)

    df = toy_weather_data(p)
    df_string_to_cat(df)
    names = np.unique(df['state'])
    catnames = OrderedDict()
    for i,v in enumerate(names):
        catnames[i+1] = v
    df_cat_to_catcode(df)
    X, y = df.drop('temperature', axis=1), df['temperature']

    # X, y, catnames, avgtemps, true_impact = toy_weather_data(n=n, p=p, n_outliers=n_outliers)
    y_bar = np.mean(y)
    print("overall mean(y)", y_bar)

    #title = f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9
    figsize = (2.5, 2)
    fig,ax = plt.subplots(1,1, figsize=figsize)
    uniq_catcodes, avg_per_cat, ignored, merge_ignored = \
        plot_catstratpd(X, y, colname='state', targetname="temperature",
                        catnames=catnames,
                        n_trials=1,
                        min_samples_leaf=min_samples_leaf,
                        show_x_counts=False,
                        ticklabel_fontsize=6,
                        pdp_marker_size=10,
                        # yrange=(-35,35),
                        yrange=(0,35),
                        # min_y_shifted_to_zero=False,
                        min_y_shifted_to_zero=True,
                        ax=ax,
                        verbose=False,
                        # title=
                        )
    abs_avg_per_cat = np.abs(avg_per_cat)
    avg_abs_pdp = np.nanmean(abs_avg_per_cat)
    print("avg_abs_pdp", avg_abs_pdp)

    print("ignored", ignored, "merge_ignored", merge_ignored)

    plot_stratpd(X, y, colname='state', targetname="temperature",
                 show_x_counts=False,
                 pdp_marker_size=20)
    # for i,(state,t) in enumerate(zip(catnames,avgtemps['avgtemp'].values)):
    #     ax.text(i,45,f"{t:.1f}", horizontalalignment="center", fontsize=9)
    #     # catnames[i] += f"\n{t:.1f}"

    if seed is not None:
        np.random.set_state(save_state)

    # computed_impact = np.nanmean(np.abs(featimp.all_pairs_deltas_foo(avg_per_cat)))
    # title = f"strat ignored={ignored}, merge ignored = {merge_ignored}\nimpact={computed_impact:.1f}, true_impact={true_impact:.1f}"
    # plt.title(title)
    plt.tight_layout()
    plt.savefig('/Users/parrt/Desktop/state_vs_temp.png', dpi=200)
    plt.show()

viz_weather(100, p=50, min_samples_leaf=5, seed=1, n_outliers=0)
# viz_weather(1000, p=30, min_samples_leaf=3, seed=1, n_outliers=0)
# viz_weather(1000, p=30, min_samples_leaf=2, seed=1, n_outliers=0)

# viz_weather(1000, p=30, min_samples_leaf=15, seed=1, n_outliers=10)
# viz_weather(1000, p=30, min_samples_leaf=15, seed=2, n_outliers=10)
# viz_weather(1000, p=30, min_samples_leaf=15, seed=3, n_outliers=10)
# viz_weather(1000, p=30, min_samples_leaf=15, seed=4, n_outliers=10)