import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from PIL import Image

from stratx.partdep import *

SAVE_EXPECTED = True # turn this on to regen expected subdir of images
SAVE_EXPECTED = False

np.set_printoptions(precision=2, suppress=True, linewidth=150)

def savefig(title="", dir="expected"):
    caller_fname = traceback.extract_stack(None, 2)[0][2]
    plt.title(f"{caller_fname} {title}")
    plt.savefig(f"{dir}/{caller_fname}.png", pad_inches=0, bbox_inches=0, dpi=100)

def compare(caller_fname):
    plt.savefig(f"/tmp/{caller_fname}.png", pad_inches=0, bbox_inches=0, dpi=100)
    expected = Image.open(f"expected/{caller_fname}.png")
    found = Image.open(f"/tmp/{caller_fname}.png")
    Image.fromarray(np.hstack((np.array(expected), np.array(found)))).show()

def toy_weather_data(n = 1000, p=50, n_outliers=None):
    """
    For each state, create a (fictional) ramp of data from day 1 to 365 so mean is not
    0, as we'd get from a sinusoid.
    """
    def noise(state): return np.random.normal(-5, 5, sum(df['state'] == state))

    df_avgs = pd.read_csv("../articles/imp/genfigs/data/weather.csv")
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


def synthetic_poly_data(n=1000,max_x=1000,p=2,dtype=float):
    df = pd.DataFrame()
    for i in range(p):
        df[f'x{i + 1}'] = (np.random.random_sample(size=n) * max_x).astype(dtype)
    yintercept = 100
    df['y'] = np.sum(df, axis=1) + yintercept
    terms = [f"x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " where x_i ~ U(0,10)"
    return df, eqn


def synthetic_poly_data_gaussian(n=1000,max_x=1000,p=2,dtype=float):
    df = pd.DataFrame()
    for i in range(p):
        v = np.random.normal(loc=0, scale=1, size=n)
        v -= np.min(v)
        v /= np.max(v) # should be 0..1 now
        df[f'x{i + 1}'] = (v*max_x).astype(dtype)
        df[f'x{i + 1}'] -= np.min(df[f'x{i + 1}']) # shift back so min is 0
    yintercept = 100
    df['y'] = np.sum(df, axis=1) + yintercept
    terms = [f"x_{i+1}" for i in range(p)] + [f"{yintercept:.0f}"]
    eqn = "y = " + ' + '.join(terms) + " where x_i ~ U(0,10)"
    return df, eqn


def viz_clean_synth_uniform(n,p,max_x,min_samples_leaf, seed=None):
    if seed is not None:
        save_state = np.random.get_state()
        np.random.seed(seed)

    df, eqn = synthetic_poly_data(n, p=p, max_x=max_x, dtype=int)
    X = df.drop('y', axis=1)
    y = df['y']
    uniq_catcodes, avg_per_cat, ignored = \
        plot_catstratpd(X, y, colname='x1', targetname='y',
                        n_trials=1,
                        min_samples_leaf=min_samples_leaf,
                        show_x_counts=True,
                        show_xticks=True,
                        show_impact=True,
                        min_y_shifted_to_zero=True,
                        verbose=True,
                        figsize=(10, 4)
                        # yrange=(-1000,1000)
                        )

    if seed is not None:
        np.random.set_state(save_state)

    caller_fname = traceback.extract_stack(None, 2)[0][2]
    plt.title(f"{caller_fname} ignored={ignored}, mean(y)={np.mean(y):.1f}")
    plt.tight_layout()
    if SAVE_EXPECTED:
        plt.savefig(f"expected/{caller_fname}.png", pad_inches=0, bbox_inches=0, dpi=100)
    else:
        compare(caller_fname)

def viz_clean_synth_uniform_n1000_xrange10_minleaf2():
    viz_clean_synth_gauss(1000,2,10,2, seed=222)

def viz_clean_synth_uniform_n1000_xrange100_minleaf2():
    viz_clean_synth_gauss(1000,2,100,2, seed=222)

def viz_clean_synth_gauss(n,p,max_x,min_samples_leaf, seed=None):
    if seed is not None:
        save_state = np.random.get_state()
        np.random.seed(seed)

    df, eqn = synthetic_poly_data_gaussian(n, p=p, max_x=max_x, dtype=int)
    X = df.drop('y', axis=1)
    y = df['y']
    uniq_catcodes, avg_per_cat, ignored, merge_ignored = \
        plot_catstratpd(X, y, colname='x1', targetname='y',
                        n_trials=1,
                        min_samples_leaf=min_samples_leaf,
                        show_x_counts=True,
                        show_xticks=True,
                        show_impact=True,
                        min_y_shifted_to_zero=True,
                        verbose=True,
                        figsize=(10, 4)
                        # yrange=(-1000,1000)
                        )

    if seed is not None:
        np.random.set_state(save_state)

    caller_fname = traceback.extract_stack(None, 2)[0][2]
    plt.title(f"{caller_fname}\nstrat ignored={ignored}, merge ignored = {merge_ignored}\nmean(y)={np.mean(y):.1f}")
    plt.tight_layout()
    if SAVE_EXPECTED:
        plt.savefig(f"expected/{caller_fname}.png", pad_inches=0, bbox_inches=0, dpi=100)
    else:
        compare(caller_fname)


def viz_clean_synth_gauss_n1000_xrange25_minleaf2():
    viz_clean_synth_gauss(1000,2,25,2, seed=222)

def viz_clean_synth_gauss_n20_xrange12_minleaf2():
    viz_clean_synth_gauss(20,2,12,2, seed=222)

def viz_clean_synth_gauss_n20_xrange10_minleaf5():
    viz_clean_synth_gauss(20,2,10,5, seed=222)

def viz_clean_synth_gauss_n1000_xrange100_minleaf10():
    viz_clean_synth_gauss(1000,2,100,10, seed=222)

def viz_clean_synth_gauss_n3000_xrange10_minleaf2():
    viz_clean_synth_gauss(3000,2,10,2, seed=222)


def viz_weather(n, p, min_samples_leaf, n_outliers=0, seed=None, show_truth=True):
    if seed is not None:
        save_state = np.random.get_state()
        np.random.seed(seed)

    X, y, catnames, avgtemps, true_impact = \
        toy_weather_data(n=n, p=p, n_outliers=n_outliers)
    y_bar = np.mean(y)
    print("overall mean(y)", y_bar)
    print("avg temps = ", avgtemps)

    #title = f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9

    fig,ax = plt.subplots(1,1, figsize=(10,3))
    uniq_catcodes, avg_per_cat, ignored, merge_ignored = \
        plot_catstratpd(X, y, colname='state', targetname="temp", catnames=catnames,
                        n_trials=1,
                        min_samples_leaf=min_samples_leaf,
                        show_x_counts=True,
                        ticklabel_fontsize=6,
                        pdp_marker_size=10,
                        show_impact=True,
                        yrange=(0,50),
                        min_y_shifted_to_zero=True,
                        ax=ax,
                        verbose=True,
                        # title=
                        )

    if seed is not None:
        np.random.set_state(save_state)

    if show_truth:
        for i in range(len(avgtemps)):
            rel_avgtemp = avgtemps.iloc[i]['avgtemp'] - np.min(avgtemps['avgtemp'])
            ax.text(i, rel_avgtemp, f"{rel_avgtemp :.1f}")
    title = f"strat ignored={ignored}, merge ignored = {merge_ignored}\nmean(y)={np.mean(y):.1f}"
    caller_fname = traceback.extract_stack(None, 2)[0][2]
    plt.title(f"{caller_fname}\n{title}")
    plt.tight_layout()
    if SAVE_EXPECTED:
        plt.savefig(f"expected/{caller_fname}.png", pad_inches=0, bbox_inches=0, dpi=100)
    else:
        compare(caller_fname)

def viz_clean_weather_n100_p4_minleaf5():
    viz_weather(100, 4, 5, seed=222)

def viz_clean_weather_n100_p10_minleaf5():
    viz_weather(100, 10, 5, seed=222)

def viz_clean_weather_n100_p20_minleaf10():
    viz_weather(100, 20, 10, seed=222)

def viz_outlier8_weather_n100_p10_minleaf5():
    viz_weather(100, 10, 8, n_outliers=5, seed=222)


# viz_clean_synth_uniform_n1000_xrange10_minleaf2()
# viz_clean_synth_gauss_n20_xrange12_minleaf2()
# viz_clean_synth_gauss_n20_xrange10_minleaf5()
# viz_clean_synth_uniform_n1000_xrange100_minleaf2()
# viz_clean_synth_gauss_n1000_xrange25_minleaf2()
# viz_clean_synth_gauss_n3000_xrange10_minleaf2()
# viz_clean_synth_gauss_n1000_xrange100_minleaf10()
# viz_clean_weather_n100_p4_minleaf5()
# viz_clean_weather_n100_p10_minleaf5()
# viz_clean_weather_n100_p20_minleaf10()
viz_outlier8_weather_n100_p10_minleaf5()