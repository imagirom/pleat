"""Tests for tiling construction in all geometries."""
import numpy as np
import pytest
from eucare.half import IdObject, EuclideanPositionHEG, GeometricHEG
from eucare.example_tilesets import (
    platonic, t_4_6_12, t_3_3_3_3_6, t_3_3_4_3_4,
    t_3_12_12,
    curved_platonic, curved_snub, curved_expand,
)
from eucare.example_graphs import from_tiles
from eucare.prototiles import RegularEuclideanTile
from eucare.geometries import EuclideanGeometry, PoincareDiskModel, SphereModel


class TestEuclideanTilings:
    """Test Euclidean tiling construction and consistency."""

    @pytest.mark.parametrize("tileset_fn,name", [
        (lambda: platonic(3), "triangular"),
        (lambda: platonic(4), "square"),
        (lambda: platonic(6), "hexagonal"),
        (t_4_6_12, "4.6.12"),
        (t_3_3_3_3_6, "3.3.3.3.6"),
        (t_3_3_4_3_4, "3.3.4.3.4"),
        (t_3_12_12, "3.12.12"),
    ])
    def test_tiling_consistency(self, tileset_fn, name):
        tiles = tileset_fn()
        G = from_tiles(tiles, rings=2)
        G.check_consistency()
        assert len(G.vertices) > 0
        assert len(G.faces) > 0

    @pytest.mark.parametrize("tileset_fn", [
        lambda: platonic(3),
        lambda: platonic(4),
        lambda: platonic(6),
        t_4_6_12,
    ])
    def test_tiling_grows_with_rings(self, tileset_fn):
        tiles = tileset_fn()
        G1 = from_tiles(tiles, rings=1)
        G2 = from_tiles(tiles, rings=3)
        assert G2.order > G1.order

    @pytest.mark.parametrize("n", [3, 4, 6])
    def test_platonic_positive_areas(self, n):
        tiles = platonic(n)
        G = from_tiles(tiles, rings=2)
        for f in G.faces:
            assert f.area() > 0

    def test_euclidean_geometry_is_set(self):
        tiles = platonic(4)
        G = from_tiles(tiles, rings=1)
        assert G.geometry is EuclideanGeometry


class TestSphericalTilings:
    """Test spherical (platonic solid) tiling construction."""

    @pytest.mark.parametrize("n,k,name", [
        (3, 3, "tetrahedron"),
        (3, 4, "octahedron"),
        (3, 5, "icosahedron"),
        (4, 3, "cube"),
        (5, 3, "dodecahedron"),
    ])
    def test_platonic_solid(self, n, k, name):
        tiles = curved_platonic(n, k)
        from eucare.example_graphs import add_vertex_ring
        base_tile = tiles[-1]
        G = GeometricHEG(geometry=tiles[0].geometry,
                         other=base_tile.make_graph(add_positions=True)[0])
        # Build incrementally until the tiling closes or we hit max rings
        for _ in range(20):
            border_count = len(list(G.border_edges()))
            if border_count == 0:
                break
            try:
                add_vertex_ring(G)
            except (AssertionError, RuntimeError):
                break
        G.check_consistency()
        border_count = len(list(G.border_edges()))
        assert border_count == 0, f"{name} tiling did not close, {border_count} border edges remain"

    @pytest.mark.parametrize("n,k", [
        (3, 5),
        (5, 3),
    ])
    def test_spherical_geometry_is_set(self, n, k):
        tiles = curved_platonic(n, k)
        G = from_tiles(tiles, rings=2)
        assert G.geometry is SphereModel


class TestHyperbolicTilings:
    """Test hyperbolic tiling construction."""

    @pytest.mark.parametrize("n,k", [
        (3, 7),
        (4, 5),
        (5, 4),
        (7, 3),
    ])
    def test_hyperbolic_platonic(self, n, k):
        tiles = curved_platonic(n, k)
        G = from_tiles(tiles, rings=2)
        G.check_consistency()
        assert len(G.faces) > 0
        assert G.geometry is PoincareDiskModel

    def test_hyperbolic_has_border(self):
        """Hyperbolic tilings can never close, so border must exist."""
        tiles = curved_platonic(4, 5)
        G = from_tiles(tiles, rings=3)
        border_count = len(list(G.border_edges()))
        assert border_count > 0


class TestCurvedOperators:
    """Test curved snub and expand tilesets."""

    @pytest.mark.parametrize("fn,args", [
        (curved_snub, (3, 3)),
        (curved_snub, (4, 3)),
        (curved_expand, (3, 3)),
        (curved_expand, (4, 3)),
    ])
    def test_curved_operator_tiles(self, fn, args):
        tiles = fn(*args)
        G = from_tiles(tiles, rings=2)
        G.check_consistency()
        assert len(G.faces) > 0


class TestProtoTile:
    """Test individual proto-tile construction."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
    def test_regular_tile(self, n):
        tile = RegularEuclideanTile(n)
        G, edges = tile.make_graph(add_positions=True)
        assert len(list(G.vertices)) == n
        assert len(list(G.faces)) == 1
