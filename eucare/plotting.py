"""Matplotlib helpers for quick visualization of 2D line drawings."""
from __future__ import annotations

import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_lines(lines: NDArray, ax: Axes | None = None, **kwargs) -> None:
    """Add a :class:`LineCollection` of *lines* (shape ``(n, 2, 2)``) to *ax*.

    Extra keyword arguments are forwarded to :class:`matplotlib.collections.LineCollection`.
    """
    lc = mc.LineCollection(lines, **kwargs)
    ax = plt.gca() if ax is None else ax
    ax.add_collection(lc)


def plot_polygon(points: NDArray, **kwargs) -> None:
    """Plot the closed polygon defined by *points* (shape ``(n, 2)``)."""
    plot_lines(np.stack([
        points,
        np.concatenate([points[1:], points[:1]]),
    ]), **kwargs)


def set_equal_aspect(ax: Axes | None = None) -> None:
    """Set equal aspect ratio with a small margin on the current or given axes."""
    ax = plt.gca() if ax is None else ax
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.05)
