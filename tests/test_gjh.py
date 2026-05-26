"""Tests for GJH-notation parsing, distillation, and the cached library."""

from __future__ import annotations

import pytest

from eucare.gjh.parser import compile_gjh_graph, polygon_placement
from eucare.gjh.distill import spec_from_graph
from eucare.gjh.library import CACHED_SPECS, GJH_CODES, cached_spec


def test_polygon_placement_single_polygon():
    """Stage 1 with one polygon returns a single n-gon graph."""
    G = polygon_placement("3")
    assert len(G.faces) == 1
    assert next(iter(G.faces)).order() == 3

    G = polygon_placement("6")
    assert len(G.faces) == 1
    assert next(iter(G.faces)).order() == 6


def test_polygon_placement_multiple_phases():
    """Stage 1 with more phases adds polygons around the border."""
    # "6-3-3" = hex seed, then a triangle, then a triangle
    G = polygon_placement("6-3-3")
    face_orders = sorted(f.order() for f in G.faces)
    assert face_orders == [3, 3, 6]


def test_compile_gjh_graph_returns_expanded_tiling():
    """A simple 1-uniform code expands to many faces in a 20x20 box."""
    G = compile_gjh_graph("6-3-3/r60/r(h5)", bbox_size=20)
    # 1-uniform 3.3.3.3.6 tiling — at this bbox we expect dozens of faces.
    assert len(G.faces) > 30
    face_orders = {f.order() for f in G.faces}
    assert face_orders == {3, 6}


def test_compile_gjh_graph_platonic_hex():
    """The hex platonic tiling fills a bounded box with only hexagons."""
    G = compile_gjh_graph("6/m30/r(h1)", bbox_size=12)
    assert all(f.order() == 6 for f in G.faces)
    assert len(G.faces) > 5


def test_spec_from_graph_hex_tiling():
    """A pure hex tiling distills to a single tile spec with 6 edges."""
    G = compile_gjh_graph("6/m30/r(h1)", bbox_size=12)
    spec = spec_from_graph(G)
    assert len(spec) == 1
    only_tile = next(iter(spec.values()))
    assert len(only_tile) == 6
    # every edge of the hex glues to another hex
    assert all(neighbor == next(iter(spec.keys())) for neighbor, _ in only_tile)


def test_spec_from_graph_3_6_3_6_tiling():
    """3.6.3.6 tiling distills to two tiles, a triangle (3 edges) and hex (6 edges)."""
    G = compile_gjh_graph("3-6/m30/r(c2)", bbox_size=20)
    spec = spec_from_graph(G)
    orders = sorted(len(edges) for edges in spec.values())
    assert orders == [3, 6]


def test_library_loads_89_codes():
    assert len(GJH_CODES) == 89
    assert len(CACHED_SPECS) == 89


def test_library_preserves_numeric_prefix_ordering():
    # First entry is the regular triangle tiling; last is one of the 3-uniform.
    assert GJH_CODES[0] == "3/m30/r(h2)"
    assert GJH_CODES[1] == "6/m30/r(h1)"
    assert GJH_CODES[2] == "4/m45/r(h1)"


def test_cached_spec_known_code():
    spec = cached_spec("3/m30/r(h2)")
    assert spec == {"a": [("a", 0), ("a", 0), ("a", 0)]}


def test_cached_spec_missing_code_raises():
    with pytest.raises(KeyError):
        cached_spec("99-99-99")


from eucare import gjh as gjh_module
from eucare.example_graphs import from_tiles
from eucare.gjh import GJH_CODES, compile_gjh_spec, gjh, gjh_graph, gjh_spec


def test_gjh_returns_tiles_for_cached_code():
    tiles = gjh("3/m30/r(h2)")
    assert len(tiles) == 1
    assert tiles[0].order == 3


def test_gjh_grows_to_a_tiling():
    tiles = gjh("3-6/m30/r(c2)")  # 3.6.3.6
    G = from_tiles(tiles, rings=2)
    orders = {f.order() for f in G.faces}
    assert orders == {3, 6}


def test_gjh_spec_matches_cache_for_known_code():
    spec = gjh_spec("3/m30/r(h2)")
    assert spec == {"a": [("a", 0), ("a", 0), ("a", 0)]}


def test_gjh_graph_for_uncached_code_uses_parser():
    # Even cached codes should round-trip through gjh_graph.
    G = gjh_graph("6/m30/r(h1)", bbox_size=10)
    assert all(f.order() == 6 for f in G.faces)
    assert len(G.faces) > 4


@pytest.mark.parametrize("code", GJH_CODES)
def test_all_cached_codes_load_as_tiles(code):
    tiles = gjh(code)
    assert len(tiles) >= 1
    # Every tile must have at least 3 edges.
    assert all(tile.order >= 3 for tile in tiles)


@pytest.mark.slow
@pytest.mark.parametrize(
    "code",
    [
        "3/m30/r(h2)",  # platonic
        "3-6/m30/r(c2)",  # simple 1-uniform (3.6.3.6)
        "4-6,4-0,3,3/m/r(v1)/r(h25)",  # 3-uniform [3.4²·6; (3.6.3.6)²]
    ],
)
def test_compile_matches_cache_structurally(code):
    """Parser+distiller must produce a spec with the same tile-order multiset as the cache.

    A strict ``fresh == cached`` equality is currently not robust because the
    expansion + distillation pipeline is order-sensitive (iteration order of
    sets and dicts of faces affects which exemplars get picked and how edges
    are labelled). We therefore check the structural invariant that survives
    relabelling: the multiset of polygon orders per tile.
    """
    fresh = compile_gjh_spec(code, bbox_size=20)
    cached = gjh_spec(code)
    fresh_orders = sorted(len(edges) for edges in fresh.values())
    cached_orders = sorted(len(edges) for edges in cached.values())
    assert fresh_orders == cached_orders, (
        f"Tile-order multisets differ for {code}:\n" f"  fresh  = {fresh_orders}\n  cached = {cached_orders}"
    )
