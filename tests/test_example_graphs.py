"""Topology-level tests for example_graphs constructions."""
from __future__ import annotations

import pytest

from eucare.example_graphs import (
    get_edge_with,
    get_vertex_with,
    rosette,
)
from eucare.half import EuclideanPositionHEG, RegularNGon


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_rosette_topology(n):
    G = rosette(n=n)
    G.check_consistency()
    # A rosette of n rhombi has n tiles meeting at a central vertex; outer ring also exists.
    assert len(G.faces) >= n
    # All faces are quadrilaterals (rhombi).
    for f in G.faces:
        assert f.order() == 4


def test_get_edge_with_finds_border_edge():
    G = RegularNGon(5)
    e = get_edge_with(G, on_border=True)
    assert e.on_border()


def test_get_edge_with_raises_when_missing():
    G = RegularNGon(3)
    with pytest.raises(LookupError):
        get_edge_with(G, func=lambda h: False)


def test_get_vertex_with_predicate():
    G = RegularNGon(4)
    target = next(iter(G.vertices))
    v = get_vertex_with(G, func=lambda x: x is target)
    assert v is target


def test_euclidean_position_heg_from_rosette():
    G = EuclideanPositionHEG(other=rosette(n=5))
    # All vertices should have positions assigned.
    for v in G.vertices:
        assert 'pos' in v.attributes
