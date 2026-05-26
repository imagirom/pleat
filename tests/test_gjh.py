"""Tests for GJH-notation parsing, distillation, and the cached library."""

from __future__ import annotations

import pytest

from eucare.gjh.parser import polygon_placement


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
