"""Smoke tests for example tilesets (instantiation + minimal growth)."""

from __future__ import annotations

import pytest

from pleat import example_tilesets as ts
from pleat.example_graphs import from_tiles
from pleat.geometries import EuclideanGeometry, PoincareDiskModel, SphereModel
from pleat.prototiles import ProtoTile

# ---- Euclidean Archimedean tilesets ----

EUCLIDEAN_TILESETS = [
    ("platonic_3", lambda: ts.platonic(3)),
    ("platonic_4", lambda: ts.platonic(4)),
    ("platonic_6", lambda: ts.platonic(6)),
    ("square_strip", ts.square_strip),
    ("t_3_12_12", ts.t_3_12_12),
    ("t_4_6_12", ts.t_4_6_12),
    ("t_3_3_4_3_4", ts.t_3_3_4_3_4),
    ("t_3_3_3_3_6", ts.t_3_3_3_3_6),
    ("u2_4_6_12__3_4_6_4", ts.u2_4_6_12__3_4_6_4),
    ("pgg_2x", ts.pgg_2x),
]


@pytest.mark.parametrize("name,factory", EUCLIDEAN_TILESETS, ids=lambda x: x if isinstance(x, str) else "")
def test_euclidean_tileset_returns_prototiles(name, factory):
    tiles = factory()
    assert len(tiles) >= 1
    for t in tiles:
        assert isinstance(t, ProtoTile)


def test_platonic_invalid_n_raises():
    with pytest.raises(AssertionError):
        ts.platonic(5)


# ---- Geometry inference ----


def test_archimedean_vertex_to_geometry_euclidean():
    # 4.4.4.4 -> Euclidean
    assert ts.archimedean_vertex_to_geometry([4, 4, 4, 4]) is EuclideanGeometry


def test_archimedean_vertex_to_geometry_spherical():
    # 3.3.3 (tetrahedron) -> spherical
    assert ts.archimedean_vertex_to_geometry([3, 3, 3]) is SphereModel


def test_archimedean_vertex_to_geometry_hyperbolic():
    # 5.5.5.5 -> hyperbolic (>2pi vertex angle in Euclidean)
    assert ts.archimedean_vertex_to_geometry([5, 5, 5, 5]) is PoincareDiskModel


# ---- Curved tile constructors ----

CURVED_FACTORIES = [
    ("curved_platonic_3_3", lambda: ts.curved_platonic(3, 3)),  # tetrahedron (sphere)
    ("curved_platonic_4_3", lambda: ts.curved_platonic(4, 3)),  # cube
    ("curved_platonic_5_4", lambda: ts.curved_platonic(5, 4)),  # hyperbolic
    ("curved_snub_3_3", lambda: ts.curved_snub(3, 3)),
    ("curved_expand_4_3", lambda: ts.curved_expand(4, 3)),
    ("curved_truncate_3_3", lambda: ts.curved_truncate(3, 3)),
    ("curved_ambo_4_3", lambda: ts.curved_ambo(4, 3)),
    ("curved_omnitruncate_3_3", lambda: ts.curved_omnitruncate(3, 3)),
    ("curved_zip_5_3", lambda: ts.curved_zip(5, 3)),
]


@pytest.mark.parametrize("name,factory", CURVED_FACTORIES, ids=lambda x: x if isinstance(x, str) else "")
def test_curved_factory_returns_prototiles(name, factory):
    tiles = factory()
    assert len(tiles) >= 1
    for t in tiles:
        assert isinstance(t, ProtoTile)
        assert t.geometry is not None


# ---- Growth via from_tiles ----


def test_from_tiles_euclidean_growth():
    G = from_tiles(ts.platonic(4), rings=2)
    G.check_consistency()
    # A 2-ring expansion from a single square produces several faces.
    assert len(G.faces) > 1


def test_from_tiles_no_positions_returns_in_angle_heg():
    from pleat.half import GeometricHEG, InAngleHEG

    G = from_tiles(ts.platonic(4), rings=1, add_positions=False)
    G.check_consistency()
    assert isinstance(G, InAngleHEG)
    assert not isinstance(G, GeometricHEG)


def test_from_tiles_edge_based_growth():
    G = from_tiles(ts.platonic(3), rings=1, vertex_based=False)
    G.check_consistency()
    assert len(G.faces) >= 1
