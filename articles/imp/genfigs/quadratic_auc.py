import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def f(x): return x[0]**2

fig, ax = plt.subplots(1,1,figsize=(3,2.5))

GREY = '#D9D9D9'
impact_fill_color = '#FFE091'
n = 200
right = 3

def quad():
    lx = np.linspace(0, right, n)
    y = lx ** 2
    y2 = lx
    ax.plot(lx, y, lw=.1, c='k')
    ax.plot(lx, y2, lw=.1, c='k')
    y_area = np.mean(np.abs(y)) * (right - 0)
    y2_area = np.mean(np.abs(y2)) * (right - 0)
    ax.text(right, max(y) * .11, f"Area = {y2_area :.1f}",
            fontname='Arial', horizontalalignment='right')
    ax.text(right, max(y) * .42, f"Area = {y_area :.1f}",
            fontname='Arial', horizontalalignment='right')
    ax.set_xlabel("$x_1, x_2$", fontname='Arial')
    ax.set_ylabel("$y$", fontname='Arial')
    ax.fill_between(lx, [0] * n, y, color=impact_fill_color)
    ax.fill_between(lx, [0] * n, y2, color=GREY, alpha=.5)
    ax.fill_between(lx[lx<=1], y[lx<=1], y2[lx<=1], color=GREY, alpha=.8)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_yticks([0,2,4,6,8,9])

    plt.tight_layout()
    plt.savefig("../images/quadratic-auc.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def quad_from_mean():
    lx = np.linspace(0, right, n)
    y = lx ** 2
    y2 = lx
    ax.plot(lx, y, lw=.1, c='k')
    ax.plot(lx, y2, lw=.1, c='k')
    m = np.mean(y)
    dev_from_mean = y - m
    avg_dev_from_mean = np.mean(np.abs(dev_from_mean))
    ax.plot([0,right], [m,m], '--', c='k', lw=.5)
    ax.text(0,m*1.05,f"Mean = {m:.1f}", fontname='Arial')
    ax.text(0,m*.75,f"Area = {avg_dev_from_mean:.1f}", fontname='Arial')
    ax.text(right,max(y)*.02,
            f"Area = {np.mean(np.abs(y))*(right-0):.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.set_xlabel("$x_1$", fontname='Arial')
    ax.set_ylabel("$y$", fontname='Arial')
    ax.fill_between(lx, [0] * n, y, color=impact_fill_color)
    ax.fill_between(lx, [m] * n, y, color=GREY, alpha=.8)
    plt.tight_layout()
    plt.savefig("../images/quadratic-from-mean-auc.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

quad()
#quad_from_mean()