"""Matplotlib helpers for quick visualization of half-edge graphs."""
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np


def plot_lines(lines, ax=None, **kwargs):
    lc = mc.LineCollection(lines, **kwargs)
    ax = plt.gca() if ax is None else ax
    ax.add_collection(lc)


def plot_polygon(points, **kwargs):
    plot_lines(np.stack([
        points,
        np.concatenate([points[1:], points[:1]])
    ]), **kwargs)


def set_equal_aspect(ax=None):
    ax = plt.gca() if ax is None else ax
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.05)


#def show_lines(points, connections, ax=None):
#    plt.scatter(points[:, 0], points[:, 1])
#    lines = [np.array([points[i], points[j]]) for (i, j) in connections]
#    lc = mc.LineCollection(lines, linewidths=1)
#    ax = plt.gca()
#    ax.add_collection(lc)