import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def f(x): return x[0]**2

GREY = '#D9D9D9'
impact_fill_color = '#FFE091'
n = 300
right = 3
lx = np.linspace(0, right, n)
y = lx**2 + lx + 100
m = np.mean(y - 100)
print("y bar", m)

def quad(ax):
    y = lx ** 2
    y2 = lx
    # fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.6))
    ax.plot(lx, y, lw=.1, c='k')
    ax.plot(lx, y2, lw=.1, c='k')
    y_area = np.mean(np.abs(y)) * (right - 0)
    y2_area = np.mean(np.abs(y2)) * (right - 0)
    ax.text(right, max(y) * .11, f"$x_2$ area = {y2_area :.1f}",
            fontname='Arial', horizontalalignment='right')
    ax.text(right, max(y) * .35, f"$x_1^2$ area = {y_area :.1f}",
            fontname='Arial', horizontalalignment='right')
    ax.set_xlabel("$x_1, x_2$\n(a)", fontname='Arial')
    ax.set_ylabel("$y$", fontname='Arial')
    ax.fill_between(lx, [0] * n, y, color=impact_fill_color)
    ax.fill_between(lx, [0] * n, y2, color='#B4D2F7', alpha=.8)
    ax.fill_between(lx[lx<=1], y[lx<=1], y2[lx<=1], color='#B4D2F7', alpha=.8)

    ax.spines['top'].set_linewidth(.5)
    ax.spines['right'].set_linewidth(.5)
    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_yticks([0,2,4,6,8,9])
    #
    # plt.tight_layout()
    # plt.savefig("../images/quadratic-auc.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()


def quad_from_mean(ax):
    pdpy1 = lx ** 2
    m = np.mean(pdpy1)
    avg_dev_from_zero = np.mean(np.abs(pdpy1))
    avg_dev_from_mean = np.mean(np.abs(pdpy1 - m))

    y2 = lx
    print(np.mean(y2))
    print(np.mean(np.abs(y2-np.mean(y2))))

    ax.plot(lx, pdpy1, lw=.1, c='k')
    # ax.plot(x1, np.abs(y-m), lw=.1, c='k')

    ax.plot([0,right], [m,m], '--', c='k', lw=.5)
    ax.text(right,m*1.1,"$\overline{y}$"+f" = {m:.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.text(0,m*1.08,'$|\overline{y}-y|$ '+f"area = {avg_dev_from_mean*(right-0):.2f}", fontname='Arial')
    ax.text(right, 0.2,
            f"$y$ area = {avg_dev_from_zero * (right - 0):.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.set_xlabel("$x_1$\n(b)", fontname='Arial')
    ax.set_ylabel("$y$", fontname='Arial')
    ax.fill_between(lx, [0] * n, pdpy1, color=impact_fill_color)
    ax.fill_between(lx, [m] * n, pdpy1, color=GREY, alpha=.8)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_yticks([0,2,4,6,8,9])


def linear_from_mean(ax):
    pdpy2 = lx
    m = np.mean(pdpy2)
    avg_dev_from_zero = np.mean(np.abs(pdpy2))
    avg_dev_from_mean = np.mean(np.abs(pdpy2 - m))

    y2 = lx
    print(np.mean(y2))
    print(np.mean(np.abs(y2-np.mean(y2))))

    ax.plot(lx, pdpy2, lw=.1, c='k')

    ax.plot([0,right], [m,m], '--', c='k', lw=.5)
    ax.text(right,m*1.08,"$\overline{y}$"+f" = {m:.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.text(0,m*1.08,'$|\overline{y}-y|$ '+f"area = {avg_dev_from_mean*(right-0):.2f}", fontname='Arial')
    ax.text(right, 0.2,
            f"$y$ area = {avg_dev_from_zero * (right - 0):.1f}", fontname='Arial',
            horizontalalignment='right')
    ax.set_xlabel("$x_2$\n(c)", fontname='Arial')
    ax.fill_between(lx, [0] * n, pdpy2, color=impact_fill_color)
    ax.fill_between(lx, [m] * n, pdpy2, color=GREY, alpha=.8)

    ax.spines['left'].set_linewidth(.5)
    ax.spines['bottom'].set_linewidth(.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.set_yticks([0,2,4,6,8,9])


fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
quad(axes[0])
quad_from_mean(axes[1])
linear_from_mean(axes[2])
plt.tight_layout()
plt.savefig("../images/quadratic-auc.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
