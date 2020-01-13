from support import *
from stratx.featimp import *

figsize = (3.5, 3.0)
use_oob=False
metric = mean_absolute_error
n = 25000

X, y = load_bulldozer()

# Most recent timeseries data is more relevant so get big recent chunk
# then we can sample from that to get n
X = X.iloc[-50_000:]
y = y.iloc[-50_000:]

n_trials=5
colname = "YearMade"
colname = 'ModelID'
colname = "ProductSize"
targetname="SalePrice"
min_samples_leaf=5
min_slopes_per_x=5
all_avg_per_cat = []
style='level'
show_xticks=False
show_x_counts=True
barchart_size = 0.20
barchar_alpha = 0.9
label_fontsize = 10
ticklabel_fontsize = 10
pdp_marker_size=3
pdp_marker_alpha = .6
fontname = 'Arial'
yrange = None
show_impact = True
impact_color = '#D73028'
figsize=(5,3)
show_xlabel = True
show_ylabel = True
title=None
title_fontsize=12


idxs = resample(range(50_000), n_samples=n, replace=False)
X, y = X.iloc[idxs], y.iloc[idxs]  # get sample from last part of time range

# constrained_years = (X[colname] >= 1995) & (X[colname] <= 2010)
# X = X[constrained_years]
# y = y[constrained_years]
# n = len(X)

print(f"n={n}")
# catcodes, _, catcode2name = getcats(X, colname, None)

uniq_catcodes = np.unique(X[colname])
max_catcode = max(uniq_catcodes)

X_col = X[colname]

def avg_pd_catvalues(all_avg_per_cat):
    m = np.zeros(shape=(max_catcode+1,))
    c = np.zeros(shape=(max_catcode+1,), dtype=int)

    # For each unique catcode, sum and count avg_per_cat values found among trials
    for i in range(n_trials):
        avg_per_cat = all_avg_per_cat[i]
        catcodes = np.where(~np.isnan(avg_per_cat))[0]
        for code in catcodes:
            m[code] += avg_per_cat[code]
            c[code] += 1
    # Convert to average value per cat
    for code in np.where(m!=0)[0]:
        m[code] /= c[code]
    m = np.where(c==0, np.nan, m) # cats w/o values should be nan, not 0
    return m

for i in range(n_trials):
    print(i)
    # idxs = resample(range(n), n_samples=n, replace=True) # bootstrap
    idxs = resample(range(n), n_samples=int(n*2/3), replace=False) # subset
    X_, y_ = X.iloc[idxs], y.iloc[idxs]

    leaf_histos, avg_per_cat, ignored = \
        cat_partial_dependence(X_, y_,
                               max_catcode=np.max(X_col),
                               colname=colname,
                               n_trees=1,
                               min_samples_leaf=min_samples_leaf,
                               max_features=1.0,
                               bootstrap=False)

    all_avg_per_cat.append( avg_per_cat )


combined_avg_per_cat = avg_pd_catvalues(all_avg_per_cat)

impacts = []
for i in range(n_trials):
    avg_per_cat = all_avg_per_cat[i]
    abs_avg_per_cat = np.abs(avg_per_cat[~np.isnan(avg_per_cat)])
    trial_uniq_cats = np.where(~np.isnan(avg_per_cat))[0]
    cat_counts = [len(np.where(X_col == cat)[0]) for cat in trial_uniq_cats]
    impact = np.sum(np.abs(abs_avg_per_cat * cat_counts)) / np.sum(cat_counts)
    impacts.append(impact)
impact_order = np.argsort(impacts)
print(impacts)
print(impact_order)
avg_impact = np.mean(impacts)
print(avg_impact)

fig, ax = plt.subplots(1, 1, figsize=figsize)

cmap = plt.get_cmap('coolwarm')
colors=cmap(np.linspace(0, 1, num=n_trials))
min_y = 9999999999999
max_y = -min_y
for i in range(n_trials):
    avg_per_cat = all_avg_per_cat[i]
    if np.nanmin(avg_per_cat) < min_y:
        min_y = np.nanmin(avg_per_cat)
    if np.nanmax(avg_per_cat) > max_y:
        max_y = np.nanmax(avg_per_cat)
    trial_catcodes = np.where(~np.isnan(avg_per_cat))[0]
    print("catcodes", trial_catcodes, "range", min(trial_catcodes), max(trial_catcodes))
    # walk each potential catcode but plot with x in 0..maxcode+1; ignore nan avg_per_cat values
    xloc = -1 # go from 0 but must count nan entries
    collect_cats = []
    collect_deltas = []
    for cat in uniq_catcodes:
        cat_delta = avg_per_cat[cat]
        xloc += 1
        if np.isnan(cat_delta): continue
        # ax.plot([xloc - .15, xloc + .15], [cat_delta] * 2, c=colors[impact_order[i]], linewidth=1)
        collect_cats.append(xloc)
        collect_deltas.append(cat_delta)
    print("Got to xloc", xloc, "len(trial_catcodes)", len(trial_catcodes), "len(catcodes)", len(uniq_catcodes))
    # ax.scatter(collect_cats, collect_deltas, c=mpl.colors.rgb2hex(colors[impact_order[i]]),
    #            s=pdp_marker_size, alpha=pdp_marker_alpha)
    ax.plot(collect_cats, collect_deltas, '.', c=mpl.colors.rgb2hex(colors[impact_order[i]]),
            markersize=pdp_marker_size, alpha=pdp_marker_alpha)

# show 0 line
ax.plot([0,len(uniq_catcodes)], [0,0], '--', c='grey', lw=.5)

# Show avg line
xloc = 0
avg_delta = []
for cat in uniq_catcodes:
    cat_delta = combined_avg_per_cat[cat]
    avg_delta.append(cat_delta)
    xloc += 1

ax.plot(range(len(uniq_catcodes)), avg_delta, '.', c='k', markersize=pdp_marker_size + 1)

if show_impact:
    ax.text(0.5, .94, f"Impact {avg_impact:.2f}",
            horizontalalignment='center',
            fontsize=label_fontsize, fontname=fontname,
            transform=ax.transAxes,
            color=impact_color)

if show_x_counts:
    combined_uniq_cats = np.where(~np.isnan(combined_avg_per_cat))[0]
    _, cat_counts = np.unique(X_col[np.isin(X_col, combined_uniq_cats)], return_counts=True)
    # x_width = len(uniq_catcodes)
    # count_bar_width = x_width / len(pdpx)
    # if count_bar_width/x_width < 0.002:
    #     count_bar_width = x_width * 0.002 # don't make them so skinny they're invisible
    count_bar_width=1
    ax2 = ax.twinx()
    # scale y axis so the max count height is 10% of overall chart
    ax2.set_ylim(0, max(cat_counts) * 1/barchart_size)
    # draw just 0 and max count
    ax2.yaxis.set_major_locator(plt.FixedLocator([0, max(cat_counts)]))
    ax2.bar(x=range(len(combined_uniq_cats)), height=cat_counts, width=count_bar_width,
            facecolor='#BABABA', align='edge', alpha=barchar_alpha)
    ax2.set_ylabel(f"$x$ point count", labelpad=-12, fontsize=label_fontsize,
                   fontstretch='extra-condensed',
                   fontname=fontname)
    # shift other y axis down barchart_size to make room
    if yrange is not None:
        ax.set_ylim(yrange[0]-(yrange[1]-yrange[0])*barchart_size, yrange[1])
    else:
        ax.set_ylim(min_y-(max_y-min_y)*barchart_size, max_y)
    # ax2.set_xticks(range(len(uniq_catcodes)))
    # ax2.set_xticklabels([])
    plt.setp(ax2.get_xticklabels(), visible=False)
    # ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
    # for tick in ax2.get_xticklabels():
    #     tick.set_visible(False)
    for tick in ax2.get_yticklabels():
        tick.set_fontname(fontname)
    ax2.spines['top'].set_linewidth(.5)
    ax2.spines['right'].set_linewidth(.5)
    ax2.spines['left'].set_linewidth(.5)
    ax2.spines['bottom'].set_linewidth(.5)

# np.where(combined_avg_per_cat!=0)[0]
if show_xticks:
    ax.set_xticks(range(len(uniq_catcodes)))
    ax.set_xticklabels(uniq_catcodes)
else:
    ax.set_xticks([])
    ax.set_xticklabels([])

if show_xlabel:
    ax.set_xlabel(colname, fontsize=label_fontsize, fontname=fontname)
if show_ylabel:
    ax.set_ylabel(targetname, fontsize=label_fontsize, fontname=fontname)
if title is not None:
    ax.set_title(title, fontsize=title_fontsize, fontname=fontname)

for tick in ax.get_xticklabels():
    tick.set_fontname(fontname)
for tick in ax.get_yticklabels():
    tick.set_fontname(fontname)

# if show_xticks: # sometimes too many
#     ax.set_xticklabels(catcode2name[sorted_catcodes])
#     ax.tick_params(axis='x', which='major', labelsize=ticklabel_fontsize)
# else:
#     ax.set_xticklabels([])
#     ax.tick_params(axis='x', which='major', labelsize=ticklabel_fontsize, bottom=False)
# ax.tick_params(axis='y', which='major', labelsize=ticklabel_fontsize)

ax.spines['top'].set_linewidth(.5)
ax.spines['right'].set_linewidth(.5)
ax.spines['left'].set_linewidth(.5)
ax.spines['bottom'].set_linewidth(.5)

# ax.set_title(f"n={n}\nstd(mean(abs(y)))={std_imp:.3f}\nmin_samples_leaf={min_samples_leaf}\nmin_slopes_per_x={min_slopes_per_x}", fontsize=9)
plt.savefig(f"/Users/parrt/Desktop/catstrat-mu.pdf", pad_inches=0)
plt.show()
