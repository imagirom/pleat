"""Tests for the cutting helpers and ``cut_out_poly`` pipeline."""
from __future__ import annotations

import numpy as np

from eucare.cutting import (
    Halfplane,
    cut_out_poly,
    pointinpolygon,
    polygon_line_segment_intersections,
)
from eucare.example_graphs import from_tiles
from eucare.example_tilesets import platonic
from eucare.half import EuclideanPositionHEG


SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_pointinpolygon_inside_outside():
    assert pointinpolygon(0.5, 0.5, SQUARE)
    assert not pointinpolygon(2.0, 0.5, SQUARE)
    assert not pointinpolygon(-0.1, 0.5, SQUARE)


def test_polygon_line_segment_intersections_crossing():
    # A horizontal line at y=0.5 from x=-1 to x=2 crosses the square at x=0 and x=1.
    line = np.array([[-1.0, 0.5], [2.0, 0.5]])
    crossings = polygon_line_segment_intersections(SQUARE, line)
    assert len(crossings) == 2
    xs = sorted(c[0] for c in crossings)
    assert np.allclose(xs, [0.0, 1.0])


def test_polygon_line_segment_intersections_no_cross():
    line = np.array([[-1.0, 2.0], [2.0, 2.0]])
    assert polygon_line_segment_intersections(SQUARE, line) == []


def test_halfplane_signed_distance_and_intersection():
    hp = Halfplane(p=np.array([0.0, 0.0]), v=np.array([1.0, 0.0]))
    pts = np.array([[1.0, 0.0], [-2.0, 0.0]])
    sd = hp.signed_distance(pts)
    assert sd[0] < 0 and sd[1] > 0  # +x is "inside" (signed dist negative)
    seg = np.array([[[-1.0, 0.0], [1.0, 0.0]]])
    inter = hp.intersections(seg)
    assert np.allclose(inter[0], [0.0, 0.0])


def test_cut_out_poly_smoke():
    G = from_tiles(platonic(4), rings=2)
    G = EuclideanPositionHEG(other=G)
    n_faces_before = len(G.faces)
    # Cut a region that crosses the interior; with delete_outside=True the
    # outside region is flood-filled and removed.
    poly = np.array([[-1.2, -1.2], [1.2, -1.2], [1.2, 1.2], [-1.2, 1.2]])
    G2 = cut_out_poly(G, poly, delete_outside=True)
    G2.check_consistency()
    # The cut removed at least some faces.
    assert len(G2.faces) < n_faces_before
    assert len(G2.faces) >= 1


def test_cut_out_poly_no_delete():
    G = from_tiles(platonic(4), rings=1)
    G = EuclideanPositionHEG(other=G)
    poly = np.array([[-0.7, -0.7], [0.7, -0.7], [0.7, 0.7], [-0.7, 0.7]])
    G2 = cut_out_poly(G, poly, delete_outside=False)
    G2.check_consistency()
