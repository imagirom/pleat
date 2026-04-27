"""Smoke tests for plotting helpers (matplotlib backend forced to Agg)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eucare.plotting import plot_lines, plot_polygon, set_equal_aspect  # noqa: E402


def test_plot_lines_adds_collection():
    fig, ax = plt.subplots()
    lines = np.array([[[0.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    before = len(ax.collections)
    plot_lines(lines, ax=ax)
    assert len(ax.collections) == before + 1
    plt.close(fig)


def test_plot_polygon_runs_on_default_axes():
    fig, _ax = plt.subplots()
    plot_polygon(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    plt.close(fig)


def test_set_equal_aspect():
    fig, ax = plt.subplots()
    set_equal_aspect(ax)
    assert ax.get_aspect() == 1.0
    plt.close(fig)
