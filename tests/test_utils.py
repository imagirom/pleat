"""Tests for eucare.utils helpers."""
from __future__ import annotations

from eucare.half import RegularNGon
from eucare.utils import invert_mapping, random_directed_set


def test_invert_mapping_basic():
    assert invert_mapping({1: 'a', 2: 'b'}) == {'a': 1, 'b': 2}


def test_invert_mapping_empty():
    assert invert_mapping({}) == {}


def test_random_directed_set_pairs_each_edge_once():
    G = RegularNGon(6)
    directed = random_directed_set(G)
    # Each undirected edge contributes exactly one half-edge.
    assert len(directed) * 2 == len(G.halfedges)
    # No half-edge and its reverse appear together.
    for h in directed:
        assert h.rev not in directed


def test_random_directed_set_accepts_iterable_of_halfedges():
    G = RegularNGon(4)
    directed = random_directed_set(list(G.halfedges))
    assert len(directed) * 2 == len(G.halfedges)
