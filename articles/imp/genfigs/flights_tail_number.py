from support import *
from stratx.featimp import *
from stratx.partdep import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True, linewidth=300)#, threshold=1e10)

def marginal(df, colname, targetname):
    grouped = df.groupby(colname).mean()
    grouped = grouped.reset_index()
    cats = grouped[colname]
    avg_y = grouped[targetname]
    fig, ax = plt.subplots(1,1,figsize = (10, 4))
    # y = avg_y
    y = avg_y - np.mean(avg_y)
    # y = avg_y - np.mean(df[targetname])
    plt.xlabel(f"{colname} (impact {np.mean(np.abs(y)):.2f}, avg marginal y {np.mean(avg_y):.2f}, avg y {np.mean(df[targetname]):.2f})")
    plt.ylabel(targetname)
    catpos = list(range(len(cats)))
    # print("marginal y", avg_y.values)
    # catpos = catpos[:20]
    # avg_y = avg_y[:20]
    barcontainer = ax.bar(x=catpos,
                          height=y,
                          color='#A5D9B5')
    # Alter appearance of each bar
    for rect in barcontainer.patches:
        rect.set_linewidth(.1)
        rect.set_edgecolor('#444443')
    # plt.bar(cats, avg_y)
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.xlim(0, 6000)
    plt.ylim(-25,100)
    plt.title("Marginal")
    plt.show()

colname = 'TAIL_NUMBER'
targetname = 'ARRIVAL_DELAY'

X, y, X_train, X_test, y_train, y_test = load_dataset("flights", "ARRIVAL_DELAY")
# X = X[0:100]
# y = y[0:100]
uniq_x = np.unique(X[colname])
df = pd.concat([X, y], axis=1)
df = df[df[colname].isin(uniq_x[:30])]
X = df.drop('ARRIVAL_DELAY', axis=1)
y = df['ARRIVAL_DELAY']

"""
min_samples=2 gives about 25 disjoint groups
min_samples=3 and we get only 2 straggler leaves with 3 and 2 cats. impact 21 not 24.
at 5 and 6, we get impact about 17
"""

marginal(df, colname, targetname)

uniq_catcodes, combined_avg_per_cat, ignored = \
    plot_catstratpd(X, y, colname, targetname,
                    min_samples_leaf=15,#30000,
                    yrange=(-25,100),
                    figsize=(10,4),
                    n_trials=1,
                    mean_centered=True,
                    show_x_counts=False,
                    show_unique_cat_xticks=False,
                    show_xticks=True,
                    min_y_shifted_to_zero=False,
                    show_impact=True,
                    # sort_by_y=True,
                    verbose=False)

# plt.xlim(0,6000)
print("IGNORED", ignored)
plt.tight_layout()
# plt.savefig(f"/Users/parrt/Desktop/flight-fnum-cat-most_common.pdf", pad_inches=0)
plt.show()
