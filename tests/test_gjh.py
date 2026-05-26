"""Tests for GJH-notation parsing, distillation, and the cached library."""

from __future__ import annotations

import pytest

from eucare.gjh.parser import compile_gjh_graph, polygon_placement


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
