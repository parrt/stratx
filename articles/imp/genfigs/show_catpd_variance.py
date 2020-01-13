from support import *
from stratx.featimp import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25000

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



X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

idxs = resample(range(50_000), n_samples=n, replace=False)
X, y = X.iloc[idxs], y.iloc[idxs]  # get sample from last part of time range

# constrained_years = (X[colname] >= 1995) & (X[colname] <= 2010)
# X = X[constrained_years]
# y = y[constrained_years]
# n = len(X)

print(f"n={n}")

#title = f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9
plot_catstratpd(X, y, colname='ProductSize', targetname="SalePrice")

plt.savefig(f"/Users/parrt/Desktop/catstrat-mu.pdf", pad_inches=0)
plt.show()
