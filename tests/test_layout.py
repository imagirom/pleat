"""Tests for layout helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import rosette
from pleat.half import EuclideanPositionHEG, RegularNGon
from pleat.layout import (
    angle_to_height,
    min_edge_length,
    optimal_rotation,
    optimize_rotation,
    rotate_graph,
)


def _euclidean(G):
    return EuclideanPositionHEG(other=G) if not isinstance(G, EuclideanPositionHEG) else G


def test_angle_to_height_nonnegative():
    G = _euclidean(rosette(n=6))
    assert angle_to_height(G, 0.0) >= 0
    assert angle_to_height(G, np.pi / 4) >= 0


def test_optimal_rotation_returns_valid_angle():
    G = _euclidean(rosette(n=6))
    a = optimal_rotation(G, steps=200)
    assert 0 <= a < np.pi


def test_optimize_rotation_does_not_increase_height():
    G = _euclidean(rosette(n=8))
    h_before = angle_to_height(G, 0.0)
    a = optimize_rotation(G)
    h_after = angle_to_height(G, 0.0)
    assert h_after <= h_before + 1e-9
    assert isinstance(a, (float, np.floating))


def test_rotate_graph_preserves_edge_lengths():
    G = _euclidean(rosette(n=4))
    lens_before = sorted(np.linalg.norm(h.orig["pos"] - h.dest["pos"]) for h in G.halfedges)
    rotate_graph(G, 0.7)
    lens_after = sorted(np.linalg.norm(h.orig["pos"] - h.dest["pos"]) for h in G.halfedges)
    np.testing.assert_allclose(lens_before, lens_after, atol=1e-9)


def test_min_edge_length_positive():
    G = _euclidean(rosette(n=5))
    assert min_edge_length(G) > 0


def test_min_edge_length_excluding_border():
    G = _euclidean(rosette(n=4))
    full = min_edge_length(G, include_border=True)
    inner = min_edge_length(G, include_border=False)
    # Inner edges are a subset of all edges, so the inner-only minimum is at least as large.
    assert inner == pytest.approx(full) or inner >= full
