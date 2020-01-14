from support import *
from stratx.featimp import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25000

def toy_weather_data(n = 1000, p=50):
    """
    For each state, create a (fictional) ramp of data from day 1 to 365 so mean is not
    0, as we'd get from a sinusoid.
    """
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df_avgs = pd.read_csv("data/weather.csv")
    df_avgs.head(3)

    df = pd.DataFrame()
    df['dayofyear'] = np.random.randint(1, 365 + 1, size=n)
    df['state'] = np.random.randint(0, p, size=n) # get only p states
    df['temp'] = .1 * df['dayofyear'] + df_avgs['avgtemp'].iloc[df['state']].values
    return df.drop('temp', axis=1), df['temp'], df_avgs['state'].values, df_avgs.iloc[0:p]

# X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
# X = X.iloc[-50_000:]
# y = y.iloc[-50_000:]
#
# idxs = resample(range(50_000), n_samples=n, replace=False)
# X, y = X.iloc[idxs], y.iloc[idxs]  # get sample from last part of time range

# constrained_years = (X[colname] >= 1995) & (X[colname] <= 2010)
# X = X[constrained_years]
# y = y[constrained_years]
# n = len(X)

p = 50
n = 5000
X, y, catnames, avgtemps = toy_weather_data(n=n, p=p)
y_bar = np.mean(y)
print("overall mean(y)", y_bar)
print("avg temps = ", avgtemps)

#title = f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9

fig,ax = plt.subplots(1,1, figsize=(20,3))
uniq_catcodes, avg_per_cat, ignored = \
    plot_catstratpd(X, y, colname='state', targetname="temp", catnames=catnames,
                    n_trials=1,
                    show_x_counts=False,
                    ticklabel_fontsize=6,
                    pdp_marker_size=10,
                    show_impact=True,
                    ax=ax,
                    title=f"n={n}, mean(y)={y_bar :.1f}, true impact={np.mean(avgtemps['avgtemp']-np.min(avgtemps['avgtemp'])):.1f}")

min_cat_value = np.min(avg_per_cat)
# for i in range(p):
#     ax.text(i, np.max(avg_per_cat)*.90, f"{catnames[i]}={avgtemps.iloc[i]['avgtemp']:.1f}", fontsize=9,
#             horizontalalignment='center')
#     ax.text(i, np.max(avg_per_cat)*.84, f"{catnames[i]}-min={avgtemps.iloc[i]['avgtemp']-np.min(avgtemps['avgtemp']):.1f}",
#             fontsize=9,
#             horizontalalignment='center')
#     ax.text(i, np.max(avg_per_cat)*.78, "$\\tilde{y}$="+f"{avg_per_cat[i]:.1f}", fontsize=9,
#             horizontalalignment='center')

plt.tight_layout()
plt.savefig(f"/Users/parrt/Desktop/weather-mu.pdf", pad_inches=0)
plt.show()
