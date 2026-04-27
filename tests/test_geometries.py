"""Behavioural tests for geometry backends (no numerical fragility).

These tests check structural invariants that hold across backends:
- distance is symmetric,
- rotation by 0 is the identity,
- ``to_euclidean`` returns 2D points,
- ``from_polar``/``to_polar`` round-trips, and
- ``unit_vector`` lies at unit distance from the origin.
"""
from __future__ import annotations

import numpy as np
import pytest

from eucare.geometries import EuclideanGeometry, PoincareDiskModel, SphereModel

BACKENDS = [EuclideanGeometry, PoincareDiskModel, SphereModel]


def _to_array(p):
    """Coerce a backend point to a numpy array, preserving complex dtype if needed."""
    return np.asarray(p)


@pytest.mark.parametrize("G", BACKENDS)
def test_origin_is_returned(G):
    o = G.origin()
    assert o is not None


@pytest.mark.parametrize("G", BACKENDS)
def test_distance_is_symmetric(G):
    p = G.from_polar(0.3, 0.5)
    q = G.from_polar(0.4, 1.2)
    assert G.distance(p, q) == pytest.approx(G.distance(q, p), abs=1e-9)


@pytest.mark.parametrize("G", BACKENDS)
def test_distance_to_self_is_zero(G):
    p = G.from_polar(0.3, 0.7)
    assert G.distance(p, p) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("G", BACKENDS)
def test_rotation_by_zero_is_identity(G):
    p = G.from_polar(0.5, 0.3)
    rot = G.rotation(G.origin(), 0.0)
    out = _to_array(rot(p))
    assert np.allclose(out, _to_array(p), atol=1e-9)


@pytest.mark.parametrize("G", BACKENDS)
def test_to_euclidean_returns_2d(G):
    pts = np.array([G.from_polar(r, a) for r, a in [(0.1, 0.0), (0.2, 1.0)]])
    out = np.asarray(G.to_euclidean(pts), dtype=float)
    assert out.shape[-1] == 2


@pytest.mark.parametrize("G", BACKENDS)
def test_polar_roundtrip(G):
    p = G.from_polar(0.4, 1.1)
    r, a = G.to_polar(p)
    p2 = G.from_polar(r, a)
    assert np.allclose(_to_array(p), _to_array(p2), atol=1e-9)


@pytest.mark.parametrize("G", BACKENDS)
def test_unit_vector_has_unit_distance_to_origin(G):
    u = G.unit_vector(0.7)
    assert G.distance_to_origin(u) == pytest.approx(1.0, abs=1e-6)
