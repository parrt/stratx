from support import *
from stratx.featimp import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt


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
    X, y = load_rent()
    I = importances(X[:2000], y[:2000])
    X, y = load_bulldozer()
    I = importances(X[:2000], y[:2000])
    X, y, _ = load_flights(n=2000)
    I = importances(X[:2000], y[:2000])
    print("Finished JIT warmup")


def rent():
    if os.path.exists(f"data/rent-timing.csv"):
        return pd.read_csv(f"data/rent-timing.csv")

    X, y = load_rent()
    return get_timing("rent", X, y)


def bulldozer():
    if os.path.exists(f"data/bulldozer-timing.csv"):
        return pd.read_csv(f"data/bulldozer-timing.csv")

    X, y = load_bulldozer()

    # Most recent timeseries data is more relevant so get big recent chunk
    # then we can sample from that to get n
    X = X.iloc[-50_000:]
    y = y.iloc[-50_000:]

    idxs = resample(range(50_000), n_samples=50_000, replace=False) # shuffle
    X, y = X.iloc[idxs], y.iloc[idxs]
    return get_timing("bulldozer", X, y, catcolnames={"AC","ModelID"})


def flight(max_size=30_000):
    if os.path.exists(f"data/flight-timing.csv"):
        return pd.read_csv(f"data/flight-timing.csv")

    X, y, _ = load_flights(n=max_size)
    return get_timing("flight", X, y,
                      catcolnames={'AIRLINE',
                                   'ORIGIN_AIRPORT',
                                   'DESTINATION_AIRPORT',
                                   'FLIGHT_NUMBER',
                                   'DAY_OF_WEEK'},
                      cat_min_samples_leaf=2)


jit_warmup()

R_flight = flight()
R_rent = rent()
R_bulldozer = bulldozer()

figsize = (3.5, 3.0)

fig, ax = plt.subplots(1,1,figsize=figsize)

ax.plot(R_flight['size'], R_flight['time'], '.-', markersize=5, label="flight")
ax.plot(R_rent['size'], R_rent['time'], '.-', markersize=5, label="rent")
ax.plot(R_bulldozer['size'], R_bulldozer['time'], '.-', markersize=5, label="bulldozer")

plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(f"../images/timing.pdf", pad_inches=0)
plt.show()