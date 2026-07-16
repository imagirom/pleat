"""Tests for NetworkX <-> EuclideanPositionHEG conversion."""

from __future__ import annotations

import networkx as nx
import numpy as np

from pleat.conversions import EHEG_from_nx
from pleat.half import EuclideanPositionHEG


def test_simple_triangle_from_nx():
    G_nx = nx.Graph()
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    G_nx.add_cycle = None  # not used; build edges explicitly
    for p in pts:
        G_nx.add_node(p)
    for p, q in zip(pts, pts[1:] + pts[:1]):
        G_nx.add_edge(p, q)

    G = EHEG_from_nx(G_nx)
    assert isinstance(G, EuclideanPositionHEG)
    G.check_consistency()
    assert len(G.vertices) == 3
    # one inner triangular face (outer face is removed during conversion)
    assert len(G.faces) == 1
    assert next(iter(G.faces)).order() == 3


def test_quadrilateral_with_v_lookup():
    G_nx = nx.Graph()
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for p in pts:
        G_nx.add_node(p)
    for p, q in zip(pts, pts[1:] + pts[:1]):
        G_nx.add_edge(p, q)

    G, v_lookup = EHEG_from_nx(G_nx, return_v_lookup=True)
    assert set(v_lookup.keys()) == set(pts)
    for nx_node, v in v_lookup.items():
        np.testing.assert_allclose(v["pos"], np.array(nx_node))


def test_dangling_edges_pruned():
    G_nx = nx.Graph()
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
    for p in pts:
        G_nx.add_node(p)
    for p, q in zip(pts, pts[1:] + pts[:1]):
        G_nx.add_edge(p, q)
    # add a degree-1 dangle
    dangling = (2.0, 2.0)
    G_nx.add_node(dangling)
    G_nx.add_edge(pts[0], dangling)

    G = EHEG_from_nx(G_nx)
    G.check_consistency()
    # dangling node was pruned; only the triangle remains
    assert len(G.vertices) == 3
