from support import *
from stratx.featimp import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

forcerun = False
warmedup = False

def get_timing(dataset, X, y, catcolnames=set(), cat_min_samples_leaf=5, max_size = 30_000):
    sizes = np.arange(1000, max_size, 1000)
    times = []
    for n in sizes:
        X_ = X[:n]
        y_ = y[:n]
        start = timer()
        I = importances(X_, y_, catcolnames=catcolnames, cat_min_samples_leaf=cat_min_samples_leaf)
        stop = timer()
        print(f"{dataset} time for {n} records = {(stop - start):.1f}s")
        times.append(stop-start)

    R = pd.DataFrame({"size":sizes, "time":times})
    R.to_csv(f"data/{dataset}-timing.csv", index=False)
    return R


def jit_warmup():
    global warmedup
    if warmedup: return

    X, y = load_rent()
    I = importances(X[:2000], y[:2000])
    X, y = load_bulldozer()
    I = importances(X[:2000], y[:2000])
    X, y, _ = load_flights(n=2000)
    I = importances(X[:2000], y[:2000])
    print("Finished JIT warmup")
    warmedup=True


def rent():
    if not forcerun and os.path.exists(f"data/rent-timing.csv"):
        return pd.read_csv(f"data/rent-timing.csv")

    jit_warmup()
    X, y = load_rent()
    return get_timing("rent", X, y)


def bulldozer():
    if not forcerun and os.path.exists(f"data/bulldozer-timing.csv"):
        return pd.read_csv(f"data/bulldozer-timing.csv")

    jit_warmup()
    X, y = load_bulldozer()

    # Most recent timeseries data is more relevant so get big recent chunk
    # then we can sample from that to get n
    X = X.iloc[-50_000:]
    y = y.iloc[-50_000:]

    idxs = resample(range(50_000), n_samples=50_000, replace=False) # shuffle
    X, y = X.iloc[idxs], y.iloc[idxs]
    return get_timing("bulldozer", X, y, catcolnames={"AC","ModelID"})


def flight(max_size=30_000):
    if not forcerun and os.path.exists(f"data/flight-timing.csv"):
        return pd.read_csv(f"data/flight-timing.csv")

    jit_warmup()
    X, y, _ = load_flights(n=max_size)
    return get_timing("flight", X, y,
                      catcolnames={'AIRLINE',
                                   'ORIGIN_AIRPORT',
                                   'DESTINATION_AIRPORT',
                                   'FLIGHT_NUMBER',
                                   'DAY_OF_WEEK'},
                      cat_min_samples_leaf=2)


R_flight = flight()
R_rent = rent()
R_bulldozer = bulldozer()

R_flight['size'] /= 1000
R_rent['size'] /= 1000
R_bulldozer['size'] /= 1000

figsize = (3.5, 3.0)

fig, ax = plt.subplots(1,1,figsize=figsize)

lw = .8
ax.plot(R_flight['size'], R_flight['time'], '-',
        markersize=5, label="flight", lw=lw*2, c='#FEAE61')
ax.plot(R_rent['size'], R_rent['time'], '-',
        markersize=5, label="rent", lw=lw, c='#A40227')
ax.plot(R_bulldozer['size'], R_bulldozer['time'], '-',
        markersize=5, label="bulldozer", lw=lw, c='#415BA3')

# Fit to quadratic for flight delay data; mildly quadratic
poly = PolynomialFeatures(2)
model = make_pipeline(poly, Ridge())
x = R_flight['size'].values.reshape(-1, 1)
model.fit(x, R_flight['time'])
print(poly.get_feature_names())
ridge = model.named_steps['ridge']
#ax.text(100,40, f"${ridge.coef_[1]:.5f} n + {ridge.coef_[2]:.8f} n^2$")
y = model.predict(x)

ax.plot(x, y, ':', c='k', lw=.7, label=f"${ridge.coef_[1]:.3f} n + {ridge.coef_[2]:.3f} n^2$")

plt.legend(loc="upper left")

ax.set_ylim(0,120)
ax.set_xlabel(f"Sample size $n$ ($10^3$ samples)")
ax.set_ylabel("Time (sec)")

plt.tight_layout()
plt.savefig(f"../images/timing.pdf", pad_inches=0)
plt.show()