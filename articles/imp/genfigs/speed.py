from support import *
from stratx.featimp import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

forcerun = False
warmedup = True

def get_timing(dataset, X, y, catcolnames=set(),
               min_samples_leaf=20, # same across all simulations
               cat_min_samples_leaf=10,
               max_size = 30_000):
    sizes = np.arange(1000, max_size+1, 1000)
    times = []
    for n in sizes:
        X_ = X[:n]
        y_ = y[:n]
        start = timer()
        I = importances(X_, y_, catcolnames=catcolnames,
                        min_samples_leaf=min_samples_leaf,
                        cat_min_samples_leaf=cat_min_samples_leaf)
        stop = timer()
        print(f"{dataset} time for {n} records = {(stop - start):.1f}s")
        times.append(stop-start)

    R = pd.DataFrame({"size":sizes, "time":times})
    R.to_csv(f"data/{dataset}-timing.csv", index=False)
    return R


def jit_warmup():
    global warmedup
    if warmedup: return

    X, y = load_rent(n=2000)
    I = importances(X[:2000], y[:2000])
    X, y = load_bulldozer(n=2000)
    I = importances(X[:2000], y[:2000])
    X, y, _ = load_flights(n=2000)
    I = importances(X[:2000], y[:2000])
    print("Finished JIT warmup")
    warmedup=True


def rent(max_size=30_000):
    if not forcerun and os.path.exists(f"data/rent-timing.csv"):
        return pd.read_csv(f"data/rent-timing.csv")

    jit_warmup()
    X, y = load_rent(max_size)
    return get_timing("rent", X, y, max_size=max_size)


def bulldozer(max_size=30_000):
    if not forcerun and os.path.exists(f"data/bulldozer-timing.csv"):
        return pd.read_csv(f"data/bulldozer-timing.csv")

    jit_warmup()
    X, y = load_bulldozer(max_size)
    print("bulldozer shape",X.shape)

    return get_timing("bulldozer", X, y,
                      catcolnames={"AC", "ModelID", "auctioneerID"},
                      max_size=max_size)


def flight(max_size=30_000):
    if not forcerun and os.path.exists(f"data/flights-timing.csv"):
        return pd.read_csv(f"data/flights-timing.csv")

    jit_warmup()
    X, y, _ = load_flights(n=max_size)
    return get_timing("flights", X, y,
                      catcolnames={'AIRLINE',
                                   'ORIGIN_AIRPORT',
                                   'DESTINATION_AIRPORT',
                                   'FLIGHT_NUMBER',
                                   'DAY_OF_WEEK'},
                      max_size=max_size)


def fitcurve(dataset,x,y,order=2):
    poly = PolynomialFeatures(order, include_bias=False) # don't use x1 * x2 terms
    model = make_pipeline(poly, Ridge())
    model.fit(x, y)
    print("feature names", poly.get_feature_names())
    ridge = model.named_steps['ridge']
    s = model.score(x,y)
    # y_pred = model.predict(x)
    # ax.plot(x, y_pred, ':', c='k', lw=.7)
    if order==3:
        eqn = f"${ridge.coef_[0]:.3f} n + {ridge.coef_[1]:.3f} n^2 + {ridge.coef_[2]:.4f} n^3$"
    elif order == 2:
        eqn = f"${ridge.coef_[0]:.3f} n + {ridge.coef_[1]:.3f} n^2$"
    else:
        eqn = f"${ridge.coef_[1]:.3f} n$"
    print(f"{dataset} R^2 {s:.5f} {eqn}")
    return s, eqn


R_flight = flight(max_size=30_000)
R_rent = rent(max_size=30_000)
R_bulldozer = bulldozer(max_size=30_000)

R_flight['size'] /= 1000
R_rent['size'] /= 1000
R_bulldozer['size'] /= 1000

# Fit to quadratic for FLIGHT delay data; mildly quadratic
x = R_flight['size'].values.reshape(-1, 1)
y = R_flight['time']
fl_s, fl_eqn = fitcurve("flights", x, y, order=2)

x = R_bulldozer['size'].values.reshape(-1, 1)
y = R_bulldozer['time']
bu_s, bu_eqn = fitcurve("bulldozer", x, y, order=2)

x = R_rent['size'].values.reshape(-1, 1)
y = R_rent['time']
re_s, re_eqn = fitcurve("rent", x, y, order=2)

print(r"\begin{tabular}{r r r r r r r r r}")
print(r"{\bf dataset} & $p$ & catvars & {\small $n$=1,000} & {\small 10,000} & {\small 20,000} & {\small 30,000} & time versus $n$~~ & $R^2$\\")
print(r"\hline")
print(r"{\tt\small flight} & 17 & 5", end=' & ')
print(f"{R_flight[R_flight['size']==1.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_flight[R_flight['size']==10.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_flight[R_flight['size']==20.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_flight[R_flight['size']==30.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{{\\small {fl_eqn}}} & {{\\small {fl_s:.4f}}}\\\\")
print(r"{\tt\small bulldozer} & 14 & 2", end=' & ')
print(f"{R_bulldozer[R_bulldozer['size']==1.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_bulldozer[R_bulldozer['size']==10.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_bulldozer[R_bulldozer['size']==20.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_bulldozer[R_bulldozer['size']==30.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{{\\small {bu_eqn}}} & {{\\small {bu_s:.4f}}}\\\\")
print(r"{\tt\small rent} & 20 & 0", end=' & ')
print(f"{R_rent[R_rent['size']==1.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_rent[R_rent['size']==10.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_rent[R_rent['size']==20.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{R_rent[R_rent['size']==30.0]['time'].values[0]:.1f}s", end=' & ')
print(f"{{\\small {re_eqn}}} & {{\\small {re_s:.4f}}}\\\\")
print(r"\end{tabular}")


lw = .8
figsize = (3.5, 3.0)
fig, ax = plt.subplots(1,1,figsize=figsize)

ax.plot(R_flight['size'], R_flight['time'], '-',
        markersize=5, label="flights", lw=lw, c='#FEAE61')
ax.plot(R_rent['size'], R_rent['time'], '-',
        markersize=5, label="rent", lw=lw, c='#A40227')
ax.plot(R_bulldozer['size'], R_bulldozer['time'], '-',
        markersize=5, label="bulldozer", lw=lw, c='#415BA3')

plt.legend(loc="upper left")

ax.set_ylim(0,45)
ax.set_xlabel(f"Sample size $n$ ($10^3$ samples)")
ax.set_ylabel("Time (sec)")

plt.tight_layout()
plt.savefig(f"../images/timing.pdf", pad_inches=0)
plt.show()