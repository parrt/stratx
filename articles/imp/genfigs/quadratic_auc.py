import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def f(x): return x[0]**2

GREY = '#D9D9D9'
impact_fill_color = '#FFE091'
n = 200
right = 3

def quad():
    lx = np.linspace(0, right, n)
    y = lx ** 2
    y2 = lx
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.6))
    ax.plot(lx, y, lw=.1, c='k')
    ax.plot(lx, y2, lw=.1, c='k')
    y_area = np.mean(np.abs(y)) * (right - 0)
    y2_area = np.mean(np.abs(y2)) * (right - 0)
    ax.text(right, max(y) * .11, f"$x_2$ area = {y2_area :.1f}",
            fontname='Arial', horizontalalignment='right')
    ax.text(right, max(y) * .35, f"$x_1^2$ area = {y_area :.1f}",
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
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.6))
    m = np.mean(y)
    avg_dev_from_zero = np.mean(np.abs(y))
    avg_dev_from_mean = np.mean(np.abs(y - m))

    y2 = lx
    print(np.mean(y2))
    print(np.mean(np.abs(y2-np.mean(y2))))

    ax.plot(lx, y, lw=.1, c='k')
    ax.plot(lx, np.abs(y-m), lw=.1, c='k')

    ax.plot([0,right], [m,m], '--', c='k', lw=.5)
    ax.text(1.5,m*1.1,"$\overline{y}$"+f" = {m:.1f}", fontname='Arial',
            horizontalalignment='center')
    ax.text(0,m*.73,'$|\overline{y}-y|$ '+f"area = {avg_dev_from_mean*(right-0):.1f}", fontname='Arial')
    ax.text(right, 0.2,
            f"$y$ area = {avg_dev_from_zero * (right - 0):.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.set_xlabel("$x_1$", fontname='Arial')
    ax.set_ylabel("$y$", fontname='Arial')
    ax.fill_between(lx, [0] * n, y, color=impact_fill_color)
    ax.fill_between(lx, [m] * n, y, color=GREY, alpha=.8)

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
    plt.savefig("../images/quadratic-from-mean-auc.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


def linear_from_mean():
    lx = np.linspace(0, right, n)
    y = lx
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.6))
    m = np.mean(y)
    avg_dev_from_zero = np.mean(np.abs(y))
    avg_dev_from_mean = np.mean(np.abs(y - m))

    y2 = lx
    print(np.mean(y2))
    print(np.mean(np.abs(y2-np.mean(y2))))

    ax.plot(lx, y, lw=.1, c='k')

    ax.plot([0,right], [m,m], '--', c='k', lw=.5)
    ax.text(0.05,m*1.05,"$\overline{y}$"+f" = {m:.1f}", fontname='Arial',
            horizontalalignment='left')
    ax.text(0,m*.73,'$|\overline{y}-y|$ '+f"area = {avg_dev_from_mean*(right-0):.1f}", fontname='Arial')
    ax.text(right, 0.2,
            f"$y$ area = {avg_dev_from_zero * (right - 0):.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.set_xlabel("$x_1$", fontname='Arial')
    ax.set_ylabel("$y$", fontname='Arial')
    ax.fill_between(lx, [0] * n, y, color=impact_fill_color)
    ax.fill_between(lx, [m] * n, y, color=GREY, alpha=.8)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    #ax.set_yticks([0,2,4,6,8,9])

    plt.tight_layout()
    plt.savefig("../images/linear-from-mean-auc.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

linear_from_mean()
quad()
quad_from_mean()