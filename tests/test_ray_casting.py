"""Tests for the local ray caster over crease patterns."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import platonic
from pleat.ray_casting import (
    DegenerateRayError,
    cross2,
    fan_at_vertex,
    first_crossing,
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


def _grid():
    """Unit square grid; central vertex has degree 4 with four 90-degree sectors."""
    G = from_tiles(platonic(n=4), rings=2)
    G.recompute_lengths_and_angles()
    return G


def _outgoing_towards(v, offset):
    """Return the outgoing half-edge at *v* whose destination is at *v + offset*."""
    target = np.asarray(v["pos"], dtype=float) + np.asarray(offset, dtype=float)
    return next(h for h in v.outgoing_iter() if np.allclose(h.dest["pos"], target, atol=1e-9))


def _subdivide_corner(G, v, face, offset):
    """Crease *face* from *v* to its corner at ``v + offset``; return the new half-edge."""
    corner_pos = np.asarray(v["pos"], dtype=float) + offset
    far_corner = next(w for w in face.vertex_iter() if np.allclose(w["pos"], corner_pos, atol=1e-9))
    G.subdivide_face(face, v, far_corner)
    G.recompute_lengths_and_angles()
    return _outgoing_towards(v, offset)


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
    diagonal = _subdivide_corner(G, v, north.face, [-1.0, 1.0])  # the 135-degree crease

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
    crossed, d_out, face_out = fan_at_vertex(v, d, west.face, side="right")

    assert crossed == [south]
    # crossing a vertical crease flips the vertical component
    np.testing.assert_allclose(d_out, np.array([SQRT_HALF, -SQRT_HALF]), atol=1e-12)
    assert face_out is south.face


def test_fan_side_right_transmits_through_several_creases():
    """The 3-crossing case mirrored about the horizontal axis, walked ccw.

    Pins the ``side="right"`` stepping (``g = g.pre.rev``) and the mirrored
    alternating sign, neither of which a single-crossing case reaches.
    """
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])
    north = _outgoing_towards(v, [0.0, 1.0])
    south = _outgoing_towards(v, [0.0, -1.0])
    diagonal = _subdivide_corner(G, v, west.face, [-1.0, -1.0])  # the 225-degree crease

    d = np.array([np.cos(-np.pi / 3), np.sin(-np.pi / 3)])
    crossed, d_out, face_out = fan_at_vertex(v, d, north.face, side="right")

    assert crossed == [west, diagonal, south]
    np.testing.assert_allclose(d_out, np.array([np.cos(-np.pi / 6), np.sin(-np.pi / 6)]), atol=1e-12)
    assert face_out is south.face


def test_fan_is_stable_when_theta_lands_exactly_on_pi():
    """A 1-ulp nudge must not flip the crossing count.

    In the 3-crossing fixture, arriving dead-centre of the south-west 90-degree
    sector puts the second angle exactly on ``pi``: the crease is met at
    ``t = eps*cot(pi) = -infinity``, i.e. not met, but the float sum lands
    either side of the bound.  ``angle_tol`` makes the answer the same for all
    three directions.
    """
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])
    north = _outgoing_towards(v, [0.0, 1.0])
    _subdivide_corner(G, v, north.face, [-1.0, 1.0])

    for nudge in (0.0, -1e-15, 1e-15):
        angle = np.pi / 4 + nudge
        d = np.array([np.cos(angle), np.sin(angle)])
        crossed, d_out, face_out = fan_at_vertex(v, d, west.face)

        assert crossed == [west], f"nudge {nudge}"
        np.testing.assert_allclose(d_out, np.array([-SQRT_HALF, SQRT_HALF]), atol=1e-12)
        assert face_out is west.rev.face


def test_fan_returns_no_face_when_the_ray_runs_off_the_paper():
    G = _grid()
    # a boundary vertex on the top edge of the 5x5 grid, degree 3
    v = next(w for w in G.vertices if np.allclose(w["pos"], [-1.5, 2.5], atol=1e-9))
    west = _outgoing_towards(v, [-1.0, 0.0])

    d = np.array([SQRT_HALF, SQRT_HALF])
    crossed, d_out, face_out = fan_at_vertex(v, d, west.face)

    assert crossed == [west]
    np.testing.assert_allclose(d_out, np.array([-SQRT_HALF, SQRT_HALF]), atol=1e-12)
    assert face_out is None


def test_fan_rejects_an_unknown_side():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])

    with pytest.raises(ValueError, match="side must be"):
        fan_at_vertex(v, np.array([SQRT_HALF, SQRT_HALF]), west.face, side="up")


def test_fan_raises_when_the_ray_arrives_along_a_crease():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.0, 0.0])

    d = np.array([1.0, 0.0])  # exactly anti-parallel to the west crease
    with pytest.raises(DegenerateRayError):
        fan_at_vertex(v, d, west.face)


def test_fan_raises_when_it_wraps_the_whole_vertex():
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


def _sw_face(v):
    """Return the unit square face whose top-right corner is *v*, and its centre."""
    west = _outgoing_towards(v, [-1.0, 0.0])
    return west.face, np.asarray(v["pos"], dtype=float) + [-0.5, -0.5]


def test_first_crossing_finds_the_forward_edge_of_the_face():
    G = _grid()
    v = G.central_vertex()
    face, p = _sw_face(v)

    h, s = first_crossing(face, p, np.array([1.0, 0.0]), vertex_tol=1e-9)

    # the east edge of the face runs between v and v + (0, -1)
    corner = np.asarray(v["pos"], dtype=float) + [0.0, -1.0]
    assert np.allclose(h.orig["pos"], corner) or np.allclose(h.dest["pos"], corner)
    crossing = h.orig["pos"] + s * halfedge_direction(h)
    np.testing.assert_allclose(crossing, np.asarray(v["pos"], dtype=float) + [0.0, -0.5], atol=1e-12)


def test_first_crossing_ignores_edges_behind_the_ray():
    G = _grid()
    face, p = _sw_face(G.central_vertex())

    forward, _ = first_crossing(face, p, np.array([1.0, 0.0]), vertex_tol=1e-9)
    backward, _ = first_crossing(face, p, np.array([-1.0, 0.0]), vertex_tol=1e-9)
    assert forward is not backward


def test_first_crossing_returns_t_within_the_unit_interval():
    G = _grid()
    face, p = _sw_face(G.central_vertex())

    _, s = first_crossing(face, p, np.array([0.3, 1.0]), vertex_tol=1e-9)
    assert 0.0 <= s <= 1.0


def test_first_crossing_is_invariant_to_the_length_of_the_direction():
    """``vertex_tol`` is a distance, so scaling *d* must not change the answer."""
    G = _grid()
    face, p = _sw_face(G.central_vertex())

    h1, s1 = first_crossing(face, p, np.array([0.3, 1.0]), vertex_tol=1e-6)
    h2, s2 = first_crossing(face, p, np.array([0.3, 1.0]) * 1e-4, vertex_tol=1e-6)
    assert h1 is h2
    assert s1 == pytest.approx(s2)


def test_first_crossing_skips_an_edge_closer_than_vertex_tol():
    """The slack is measured in distance along the ray, not in ray parameter."""
    G = _grid()
    v = G.central_vertex()
    face, _ = _sw_face(v)
    # a hair west of the east edge of the face, aimed east with a tiny |d|
    p = np.asarray(v["pos"], dtype=float) + [-1e-6, -0.5]
    d = np.array([1e-4, 0.0])

    assert first_crossing(face, p, d, vertex_tol=1e-3) is None
    h, _ = first_crossing(face, p, d, vertex_tol=1e-9)
    assert np.allclose(h.orig["pos"] + halfedge_direction(h) / 2, v["pos"] + np.array([0.0, -0.5]))
