from articles.pd.support import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from stratx.partdep import cat_partial_dependence, partial_dependence

forcerun = True
warmedup = False
# warmedup = True

def get_timing(dataset, X, y, catcolnames=set(), max_size = 3_000):
    sizes = np.arange(1000, max_size+1, 1000)
    times = np.zeros(shape=(len(sizes), X.shape[1]+1))
    times[:,0] = sizes
    for j, colname in enumerate(X.columns):
        coltime = np.zeros(shape=(len(sizes),))
        for i, n in enumerate(sizes):
            X_ = X[:n]
            y_ = y[:n]
            start = timer()
            if colname in catcolnames:
                cat_partial_dependence(X_, y_, colname)
            else:
                partial_dependence(X_, y_, colname)
            stop = timer()
            print(f"{colname} {dataset} time for {n} records = {(stop - start):.1f}s")
            coltime[i] = stop - start
        times[:,j+1] = coltime

    R = pd.DataFrame(data=times, columns=['size']+list(X.columns))#{"size":sizes, "time":times})
    R.to_csv(f"data/{dataset}-timing.csv", index=False)
    return R


def jit_warmup():
    global warmedup
    if warmedup: return

    X, y = load_rent(2000)
    for colname in X.columns:
        partial_dependence(X, y, colname)
    X, y = load_bulldozer(2000)
    for colname in X.columns:
        partial_dependence(X, y, colname)
    X, y, _ = load_flights(n=2000)
    for colname in X.columns:
        partial_dependence(X, y, colname)
    print("Finished JIT warmup")
    warmedup=True


def rent(max_size=30_000):
    if not forcerun and os.path.exists(f"data/rent-timing.csv"):
        return pd.read_csv(f"data/rent-timing.csv")

    jit_warmup()
    X, y = load_rent(max_size)
    print("rent shape",X.shape)
    return get_timing("rent", X, y, max_size=max_size)


def bulldozer(max_size=30_000):
    if not forcerun and os.path.exists(f"data/bulldozer-timing.csv"):
        return pd.read_csv(f"data/bulldozer-timing.csv")

    jit_warmup()
    X, y = load_bulldozer(max_size)
    print("bulldozer shape",X.shape)

    return get_timing("bulldozer", X, y, catcolnames={"AC","ModelID"},
                      max_size=max_size)


def flight(max_size=30_000):
    if not forcerun and os.path.exists(f"data/flight-timing.csv"):
        return pd.read_csv(f"data/flight-timing.csv")

    jit_warmup()
    X, y, _ = load_flights(n=max_size)
    print("flight shape",X.shape)
    return get_timing("flight", X, y,
                      catcolnames={'AIRLINE',
                                   'ORIGIN_AIRPORT',
                                   'DESTINATION_AIRPORT',
                                   'FLIGHT_NUMBER',
                                   'DAY_OF_WEEK'},
                      max_size=max_size)


def fitcurve(dataset,x,y,order=2):
    poly = PolynomialFeatures(order)
    model = make_pipeline(poly, Ridge())
    model.fit(x, y)
    # print(poly.get_feature_names())
    ridge = model.named_steps['ridge']
    s = model.score(x,y)
    # y_pred = model.predict(x)
    # ax.plot(x, y_pred, ':', c='k', lw=.7)
    if order==3:
        eqn = f"${ridge.coef_[1]:.3f} n + {ridge.coef_[2]:.3f} n^2 + {ridge.coef_[3]:.4f} n^3$"
    elif order == 2:
        eqn = f"${ridge.coef_[1]:.3f} n + {ridge.coef_[2]:.3f} n^2$"
    else:
        eqn = f"${ridge.coef_[1]:.3f} n$"
    print(f"{dataset} R^2 {s:.5f} {eqn}")
    return s, eqn


max_size = 30000
R_flight = flight(max_size=max_size)
R_rent = rent(max_size=max_size)
R_bulldozer = bulldozer(max_size=max_size)

R_flight['size'] /= 1000
R_rent['size'] /= 1000
R_bulldozer['size'] /= 1000

lw = .8
fig, axes = plt.subplots(1, 2, figsize=(6.5,3))

for colname in R_flight.drop('size', axis=1).columns:
    ax = axes[1] if colname in {'AIRLINE',
                                'ORIGIN_AIRPORT',
                                'DESTINATION_AIRPORT',
                                'FLIGHT_NUMBER',
                                'DAY_OF_WEEK',
                                'TAIL_NUMBER'} else axes[0]
    ax.plot(R_flight['size'], R_flight[colname], '-',
            markersize=5, label=colname, lw=lw * 2)#, c='#FEAE61')

for colname in R_rent.drop('size', axis=1).columns:
    axes[0].plot(R_rent['size'], R_rent[colname], '-',
            markersize=5, label=colname, lw=lw)#, c='#A40227')

for colname in R_bulldozer.drop('size', axis=1).columns:
    ax = axes[1] if colname in {"AC", "ModelID", "auctioneerID",
                                "datasource", "MachineHours_na"} else axes[0]
    ax.plot(R_bulldozer['size'], R_bulldozer[colname], '-',
            markersize=5, label=colname, lw=lw)#, c='#415BA3')

# axes[0,0].legend(loc="upper left")
# axes[0,1].legend(loc="upper left")
# axes[1,0].legend(loc="upper left")
# axes[2,0].legend(loc="upper left")
# axes[2,1].legend(loc="upper left")

axes[0].set_ylim(0,1.5)
axes[0].set_title("Numerical vars, 3 datasets", fontsize=12)
axes[1].set_ylim(0,14)
axes[1].set_title("Categorical vars, 3 datasets", fontsize=12)

# ax.set_ylim(0,5)
axes[0].set_xlabel(f"Sample size $n$ ($10^3$ samples)", fontsize=12)
axes[1].set_xlabel(f"Sample size $n$ ($10^3$ samples)", fontsize=12)
axes[0].set_ylabel("Execution time (secs)", fontsize=12)
# axes[1].set_ylabel("Execution time (secs)", fontsize=12)

axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# axes[1].set_ylabel("Time (sec)")
# axes[2,0].set_ylabel("Time (sec)")

plt.tight_layout()
plt.savefig(f"images/timing.pdf", pad_inches=0)
plt.show()
