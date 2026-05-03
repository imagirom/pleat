"""Tests for low-level overlap helpers."""

from __future__ import annotations

import numpy as np

from eucare.overlap import (
    fast_group_closeby,
    faster_group_closeby_nx,
    intervals_overlapping,
    line_segment_intersections,
)


def test_intervals_overlapping_basic():
    assert intervals_overlapping(np.array([0.0, 1.0]), np.array([0.5, 1.5]))
    assert not intervals_overlapping(np.array([0.0, 1.0]), np.array([2.0, 3.0]))


def test_line_segment_intersections_cross():
    s1 = np.array([[0.0, 0.0], [2.0, 0.0]])
    s2 = np.array([[1.0, -1.0], [1.0, 1.0]])
    out = line_segment_intersections(s1, s2)
    assert len(out) == 1
    np.testing.assert_allclose(out[0], [1.0, 0.0], atol=1e-9)


def test_line_segment_intersections_no_cross():
    s1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    s2 = np.array([[0.0, 1.0], [1.0, 1.0]])
    assert line_segment_intersections(s1, s2) == []


def test_line_segment_intersections_collinear_overlap():
    s1 = np.array([[0.0, 0.0], [2.0, 0.0]])
    s2 = np.array([[1.0, 0.0], [3.0, 0.0]])
    out = line_segment_intersections(s1, s2)
    # collinear overlapping segments produce >=1 endpoint intersection
    assert len(out) >= 1


def test_fast_group_closeby_clusters_nearby_points():
    pts = np.array([[0.0, 0.0], [0.001, 0.0], [10.0, 10.0], [10.0001, 10.0]])
    labels = fast_group_closeby(pts, eps=0.01)
    # Two clusters expected.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_faster_group_closeby_nx_empty():
    out = faster_group_closeby_nx(np.zeros((0, 2)), eps=0.01)
    assert out.shape == (0,)


def test_faster_group_closeby_nx_clusters():
    pts = np.array([[0.0, 0.0], [0.0001, 0.0], [5.0, 5.0]])
    labels = faster_group_closeby_nx(pts, eps=0.01)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]
