"""Tests for the local ray caster over crease patterns."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.ray_casting import (
    DegenerateRayError,
    cross2,
    halfedge_direction,
    signed_angle,
    transmit,
)

SQRT_HALF = np.sqrt(0.5)


def test_cross2():
    assert cross2(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cross2(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(-1.0)


def test_signed_angle_is_ccw_and_in_minus_pi_to_pi():
    east, north = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert signed_angle(east, north) == pytest.approx(np.pi / 2)
    assert signed_angle(north, east) == pytest.approx(-np.pi / 2)
    assert signed_angle(east, east) == pytest.approx(0.0)


def test_transmit_through_perpendicular_crease_goes_straight():
    # a rim segment travelling east crosses a vertical (north-south) crease
    d = np.array([1.0, 0.0])
    u = np.array([0.0, 1.0])
    np.testing.assert_allclose(transmit(d, u), d, atol=1e-12)


def test_transmit_through_45_degree_crease_turns_by_90_degrees():
    # the square-preliminary-base check from the spec
    d = np.array([1.0, 0.0])
    u = np.array([SQRT_HALF, SQRT_HALF])
    np.testing.assert_allclose(transmit(d, u), np.array([0.0, -1.0]), atol=1e-12)


def test_transmit_is_not_mirroring_across_the_crease():
    # mirroring would send the ray back into the face it came from
    d = np.array([1.0, 0.0])
    u = np.array([SQRT_HALF, SQRT_HALF])
    mirrored = 2 * np.dot(d, u) * u - d
    assert not np.allclose(transmit(d, u), mirrored)


def test_transmit_preserves_length_and_is_an_involution():
    d = np.array([0.6, -0.8])
    u = np.array([1.0, 2.0])
    out = transmit(d, u)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    np.testing.assert_allclose(transmit(out, u), d, atol=1e-12)


def test_transmit_normalises_the_crease_direction():
    d = np.array([1.0, 0.0])
    np.testing.assert_allclose(transmit(d, np.array([0.0, 5.0])), transmit(d, np.array([0.0, 1.0])), atol=1e-12)


from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.ray_casting import fan_at_vertex


def _grid():
    """Unit square grid; central vertex has degree 4 with four 90-degree sectors."""
    G = from_tiles(platonic(n=4), rings=2)
    G.recompute_lengths_and_angles()
    return G


def _outgoing_towards(v, offset):
    """Return the outgoing half-edge at *v* whose destination is at *v + offset*."""
    target = np.asarray(v["pos"], dtype=float) + np.asarray(offset, dtype=float)
    return next(h for h in v.outgoing_iter() if np.allclose(h.dest["pos"], target, atol=1e-9))


def test_fan_at_degree_4_vertex_transmits_through_exactly_one_crease():
    G = _grid()
    v = G.central_vertex()  # creases pointing W, S, E, N
    west = _outgoing_towards(v, [-1.0, 0.0])
    sw_face = west.face  # the sector spanning 180..270 degrees

    d = np.array([SQRT_HALF, SQRT_HALF])  # arriving north-east, so -d is in the SW face
    crossed, d_out, face_out = fan_at_vertex(v, d, sw_face)

    assert crossed == [west]
    # crossing a horizontal crease flips the horizontal component
    np.testing.assert_allclose(d_out, np.array([-SQRT_HALF, SQRT_HALF]), atol=1e-12)
    assert face_out is west.rev.face


def test_fan_transmits_through_several_creases_at_one_vertex():
    """Hand-computed 3-crossing case from the spec.

    A 135-degree crease is added into the north-west square, so walking
    clockwise from the south-west face the sectors are 45, 45, 90 degrees.
    Arriving along 60 degrees gives theta = 120, 165, 120, 210 -- three
    crossings before theta leaves (0, pi).
    """
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])
    north = _outgoing_towards(v, [0.0, 1.0])
    nw_face = north.face
    corner_pos = np.asarray(v["pos"], dtype=float) + [-1.0, 1.0]
    far_corner = next(w for w in nw_face.vertex_iter() if np.allclose(w["pos"], corner_pos, atol=1e-9))
    G.subdivide_face(nw_face, v, far_corner)  # the 135-degree crease
    G.recompute_lengths_and_angles()
    diagonal = _outgoing_towards(v, [-1.0, 1.0])

    d = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
    crossed, d_out, face_out = fan_at_vertex(v, d, west.face)

    assert crossed == [west, diagonal, north]
    np.testing.assert_allclose(d_out, np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)]), atol=1e-12)
    assert face_out is north.rev.face


def test_fan_grazing_a_corner_transmits_through_nothing():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])

    # theta_1 = 180 - 200 = -20 degrees, outside (0, pi): the offset ray misses
    d = np.array([np.cos(np.deg2rad(200)), np.sin(np.deg2rad(200))])
    crossed, d_out, face_out = fan_at_vertex(v, d, west.face)

    assert crossed == []
    np.testing.assert_allclose(d_out, d, atol=1e-12)
    assert face_out is west.face


def test_fan_side_right_mirrors_side_left():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])
    south = _outgoing_towards(v, [0.0, -1.0])

    d = np.array([SQRT_HALF, SQRT_HALF])
    crossed, d_out, _ = fan_at_vertex(v, d, west.face, side="right")

    assert crossed == [south]
    # crossing a vertical crease flips the vertical component
    np.testing.assert_allclose(d_out, np.array([SQRT_HALF, -SQRT_HALF]), atol=1e-12)


def test_fan_raises_when_the_ray_arrives_along_a_crease():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])

    d = np.array([1.0, 0.0])  # exactly anti-parallel to the west crease
    with pytest.raises(DegenerateRayError):
        fan_at_vertex(v, d, west.face)


def test_fan_raises_when_it_wraps_the_whole_vertex():
    from pleat.example_graphs import rosette

    G = rosette(8)  # eight equal 45-degree sectors: theta oscillates forever
    G.recompute_lengths_and_angles()
    v = G.central_vertex()
    g = next(h for h in v.outgoing_iter() if h.face is not None)
    # aim so that theta_1 is small enough that theta never leaves (0, pi)
    axis = halfedge_direction(g)
    angle = np.arctan2(axis[1], axis[0]) - np.deg2rad(20)
    d = np.array([np.cos(angle), np.sin(angle)])

    with pytest.raises(DegenerateRayError):
        fan_at_vertex(v, d, g.face)
