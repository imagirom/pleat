"""Topology-level tests for example_graphs constructions."""

from __future__ import annotations

import numpy as np
import pytest

from eucare.example_graphs import (
    RectangleDomain,
    SquareDomain,
    add_vertex_ring,
    complete_closest_vertices,
    complete_vertex,
    fill_domain,
    from_tiles,
    get_edge_with,
    get_vertex_with,
    pgg_2x_tiling,
    rosette,
)
from eucare.example_tilesets import curved_platonic, platonic, t_3_12_12
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


def test_get_vertex_with_raises_when_missing():
    G = RegularNGon(3)
    with pytest.raises(LookupError):
        get_vertex_with(G, func=lambda v: False)


def test_get_vertex_with_on_border():
    G = RegularNGon(3)
    v = get_vertex_with(G, on_border=True)
    assert v.on_border()


def test_euclidean_position_heg_from_rosette():
    G = EuclideanPositionHEG(other=rosette(n=5))
    # All vertices should have positions assigned.
    for v in G.vertices:
        assert "pos" in v.attributes


def test_complete_vertex_completes_a_border_vertex():
    G = from_tiles(t_3_12_12(), rings=1)
    v = next(iter(v for v in G.vertices if v.on_border() and v.angle_sum() > 0))
    complete_vertex(G, v)
    G.check_consistency()
    assert not v.on_border()


def test_add_vertex_ring_grows_graph():
    G = from_tiles(t_3_12_12(), rings=1)
    n_faces_before = len(G.faces)
    add_vertex_ring(G)
    G.check_consistency()
    assert len(G.faces) > n_faces_before


def test_complete_closest_vertices_grows_graph():
    G = from_tiles(curved_platonic(7, 3), rings=1)  # hyperbolic
    n_before = len(G.faces)
    complete_closest_vertices(G)
    G.check_consistency()
    assert len(G.faces) > n_before


def test_pgg_2x_tiling_smoke():
    G = pgg_2x_tiling(rings=2)
    G.check_consistency()
    assert len(G.faces) > 0


def test_kised_soccer_ball_smoke():
    from eucare.example_graphs import kised_soccer_ball

    G = kised_soccer_ball()
    G.check_consistency()
    # Soccer ball with kis applied: all faces should be triangles.
    for f in G.faces:
        assert f.order() == 3


def test_hyperbolic_square_graph_smoke():
    """Exercise hyperbolic_square_graph (Schwarz-Christoffel) on a tiny graph."""
    from eucare.example_graphs import hyperbolic_square_graph, from_tiles
    from eucare.example_tilesets import curved_platonic

    G = from_tiles(curved_platonic(7, 3), rings=1)
    # Use a coarse min_length to avoid slow refinement loop.
    G_square = hyperbolic_square_graph(G, min_length=0.5)
    # check the result is a valid Euclidean graph in the unit square.
    G_square.check_consistency()
    import numpy as np

    for v in G_square.vertices:
        assert np.all(np.abs(v["pos"]) < 2.0)


def test_hyperbolic_square_graph_dual_smoke():
    from eucare.example_graphs import hyperbolic_square_graph, from_tiles
    from eucare.example_tilesets import curved_platonic

    G = from_tiles(curved_platonic(7, 3), rings=1)
    G_square = hyperbolic_square_graph(G, min_length=0.5, dual=True)
    G_square.check_consistency()


def test_square_domain_contains():
    d = SquareDomain(4.0)
    inside = np.array([[0.0, 0.0], [1.0, 1.0], [-1.5, 1.5]])
    outside = np.array([[3.0, 0.0], [0.0, 3.0]])
    assert np.all(d.contains(inside))
    assert not np.any(d.contains(outside))


def test_fill_domain_fills_a_square_box_with_squares():
    """A 5x5 box filled with unit squares contains roughly 25 faces."""
    tiles = platonic(4)
    G = fill_domain(tiles, SquareDomain(5.0))
    # Expect ~25 faces; allow a small slack for boundary growth.
    assert 20 <= len(G.faces) <= 36
    assert all(f.order() == 4 for f in G.faces)


def test_fill_domain_rectangle_with_hexagons():
    """A wide rectangle filled with hexagons contains only hexagons."""
    tiles = platonic(6)
    G = fill_domain(tiles, RectangleDomain(10.0, 4.0))
    assert all(f.order() == 6 for f in G.faces)
    assert len(G.faces) > 3
