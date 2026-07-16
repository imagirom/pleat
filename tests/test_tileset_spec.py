"""Tests for the declarative TilesetSpec format."""

from __future__ import annotations

import pytest

from pleat.tileset_spec import (
    TilesetSpec,
    parse_edge_ref,
    spec_from_yaml,
    spec_to_yaml,
    tileset_from_spec,
)


def test_parse_edge_ref_basic():
    assert parse_edge_ref("a.0") == ("a", 0)
    assert parse_edge_ref("tri.12") == ("tri", 12)


def test_parse_edge_ref_legacy_compact():
    # Backwards-compat: allow the notebook's compact "b1" form on read.
    assert parse_edge_ref("a0") == ("a", 0)
    assert parse_edge_ref("e2") == ("e", 2)


def test_spec_yaml_roundtrip():
    spec: TilesetSpec = {
        "a": [("b", 1), ("c", 2), ("a", 0)],
        "b": [("a", 0), ("a", 0), ("a", 0)],
        "c": [("a", 1)],
    }
    yaml_text = spec_to_yaml(spec)
    parsed = spec_from_yaml(yaml_text)
    assert parsed == spec


def test_spec_from_yaml_legacy_compact_form():
    # The cached YAMLs originally used "b1" form; reader must still accept it.
    yaml_text = "a: [b1, b1, b1]\nb: [a0, a0, a0, a0, a0, a0]\n"
    spec = spec_from_yaml(yaml_text)
    assert spec == {
        "a": [("b", 1), ("b", 1), ("b", 1)],
        "b": [("a", 0)] * 6,
    }


def test_tileset_from_spec_platonic_triangle():
    # 3.6.3.6: hex + triangle, each triangle edge glues to a hex
    spec: TilesetSpec = {
        "a": [("b", 1), ("b", 1), ("b", 1)],
        "b": [("a", 0), ("a", 0), ("a", 0), ("a", 0), ("a", 0), ("a", 0)],
    }
    tiles = tileset_from_spec(spec)
    assert len(tiles) == 2
    # Tiles named in order of decreasing polygon size (hex first, then triangle)
    orders = sorted(tile.order for tile in tiles)
    assert orders == [3, 6]


def test_tileset_from_spec_grows_into_tiling():
    """Round-trip: spec → tiles → grown graph has at least one face per tile type."""
    from pleat.example_graphs import from_tiles

    spec: TilesetSpec = {
        "a": [("b", 1), ("b", 1), ("b", 1)],
        "b": [("a", 0), ("a", 0), ("a", 0), ("a", 0), ("a", 0), ("a", 0)],
    }
    tiles = tileset_from_spec(spec)
    G = from_tiles(tiles, rings=2)
    face_orders = {f.order() for f in G.faces}
    assert 3 in face_orders
    assert 6 in face_orders
