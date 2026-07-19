"""Tests for the local ray caster over crease patterns."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import platonic
from pleat.ray_casting import (
    DegenerateRayError,
    RayHit,
    _closes,
    cast_ray,
    cross2,
    default_vertex_tol,
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


def _north_edge(G):
    """Return the central vertex and its outgoing half-edge pointing north."""
    v = G.central_vertex()
    return v, _outgoing_towards(v, [0.0, 1.0])


def test_cast_ray_straight_across_a_grid_reaches_the_border():
    G = _grid()
    _, north = _north_edge(G)

    # start at the midpoint of a vertical edge, heading east: every crease it
    # meets is vertical, so it transmits straight through and never turns
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False)

    assert not path.closed
    assert path.ends[1] == "border"
    ys = [hit.position[1] for hit in path.hits]
    assert np.allclose(ys, ys[0])  # it really did travel in a straight line


def test_cast_ray_records_the_starting_point_as_its_first_hit():
    G = _grid()
    v, north = _north_edge(G)
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False)

    assert path.hits[0].halfedges[0] in (north, north.rev)
    expected = np.asarray(v["pos"], dtype=float) + [0.0, 0.5]
    np.testing.assert_allclose(path.hits[0].position, expected, atol=1e-12)


def test_cast_ray_respects_max_steps():
    G = _grid()
    _, north = _north_edge(G)
    # one step is always short: `central_vertex` is tied four ways on this
    # grid, so the ray has either two or three creases left to cross
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False, max_steps=1)

    assert path.ends[1] == "max_steps"
    assert len(path.hits) == 2  # the start plus one step


def test_cast_ray_does_not_call_a_completed_path_a_max_steps_failure():
    """The border test must not consume one of the permitted steps.

    The smallest cap under which the straight-across ray still reports
    ``"border"`` is exactly the number of crossings it makes; one fewer, and
    only then, is it a genuinely truncated path.
    """
    G = _grid()
    _, north = _north_edge(G)
    # derived, not hardcoded: `central_vertex` is tied four ways on this grid,
    # so the ray has either two or three creases left to cross
    needed = len(cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False).hits) - 1

    at_cap = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False, max_steps=needed)
    assert at_cap.ends[1] == "border"
    assert len(at_cap.hits) == needed + 1

    under_cap = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False, max_steps=needed - 1)
    assert under_cap.ends[1] == "max_steps"


def test_cast_ray_rejects_a_direction_along_its_own_start_edge():
    G = _grid()
    _, north = _north_edge(G)

    with pytest.raises(DegenerateRayError, match="along its own start edge"):
        cast_ray(G, north, 0.5, halfedge_direction(north), both_ways=False)


def _diagonal_loop_grid():
    """Grid with all four diagonals at the central vertex.

    Each diagonal turns a passing ray by 90 degrees, and the four together
    steer a ray around a closed square loop of side 1 centred on the vertex.
    """
    G = _grid()
    v = G.central_vertex()
    for towards, corner in (
        ([0.0, 1.0], [-1.0, 1.0]),  # north-west square
        ([-1.0, 0.0], [-1.0, -1.0]),  # south-west square
        ([0.0, -1.0], [1.0, -1.0]),  # south-east square
        ([1.0, 0.0], [1.0, 1.0]),  # north-east square
    ):
        face = _outgoing_towards(v, towards).face
        _subdivide_corner(G, v, face, np.asarray(corner))
    return G, v


def test_cast_ray_closes_on_a_square_loop():
    G, v = _diagonal_loop_grid()
    north = _outgoing_towards(v, [0.0, 1.0])

    # heading west from the midpoint of the north crease: the ray turns left at
    # each of the four diagonals and comes back to where it started
    path = cast_ray(G, north, 0.5, np.array([-1.0, 0.0]), both_ways=False)

    assert path.closed
    assert path.ends[1] == "closed"
    assert len(path.hits) == 9  # the start, four turns, four straight-throughs
    np.testing.assert_allclose(path.hits[-1].position, path.hits[0].position, atol=1e-12)
    np.testing.assert_allclose(path.hits[-1].direction_in, path.hits[0].direction_out, atol=1e-12)


def test_cast_ray_reports_a_degenerate_stall_rather_than_calling_it_a_border():
    """A ray that cannot leave its face has not reached the paper's edge."""
    G = _grid()
    _, north = _north_edge(G)
    # a tolerance wider than the face swallows every candidate crossing
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False, vertex_tol=10.0)

    assert path.ends[1] == "degenerate"
    assert len(path.hits) == 1


def test_cast_ray_starting_on_the_border_and_aimed_off_the_paper_stops_at_once():
    G = _grid()
    border = next(h for h in G.halfedges if h.face is None)
    # `border` has no face on its left, so its left normal points off the paper
    along = halfedge_direction(border)
    outwards = np.array([-along[1], along[0]])

    path = cast_ray(G, border, 0.5, outwards, both_ways=False)

    assert path.ends[1] == "border"
    assert len(path.hits) == 1
    assert path.hits[0].face is None


#: how far off dead-centre the vertex-hitting ray is aimed; see `_aimed_near_an_interior_vertex`
NEAR_MISS = 1e-12

START_T = 0.5 + NEAR_MISS


def _aimed_near_an_interior_vertex(G):
    """A start half-edge and direction for a ray that snaps to an interior vertex.

    The ray sets off from the west edge of the square south-west of *v*, at
    ``START_T``, so it passes ``NEAR_MISS`` to the side of *v* rather than
    through it: far inside the ``1e-9`` vertex tolerance, so the crossing is
    snapped to *v*, but far enough out that the crossing parameter is nowhere
    near exactly ``0`` or ``1``.  Aiming dead-centre instead leaves that
    parameter to float luck -- it does land on exactly ``1.0`` for some of the
    grids ``from_tiles`` builds -- and the point here is the tolerance.
    """
    v = next(w for w in G.vertices if np.allclose(w["pos"], [-0.5, 0.5], atol=1e-9))
    west = _outgoing_towards(v, [-1.0, 0.0])
    return v, west, _outgoing_towards(west.dest, [0.0, -1.0]), np.array([1.0, 0.5])


def test_cast_ray_landing_on_a_vertex_resolves_it_with_the_fan():
    """The vertex branch of the walk, reached through `cast_ray` rather than directly."""
    G = _grid()
    v, west, start, d = _aimed_near_an_interior_vertex(G)
    crossed, d_out, face_out = fan_at_vertex(v, d / np.linalg.norm(d), west.face)

    path = cast_ray(G, start, START_T, d, both_ways=False, max_steps=1)
    hit = path.hits[1]

    assert hit.vertex is v
    assert hit.halfedges == crossed == [west]
    # the hit reports the vertex's own position, not the interpolated crossing
    np.testing.assert_array_equal(hit.position, v["pos"])
    np.testing.assert_allclose(hit.direction_out, d_out, atol=1e-12)
    assert hit.face is face_out
    # the crossing sits a near-miss away from an end of whichever edge
    # `first_crossing` found -- which is why the vertex test is a distance and
    # not `s == 0.0` / `s == 1.0`
    assert 1e-13 < min(hit.t, 1 - hit.t) < 1e-9


def test_cast_ray_does_not_close_when_it_returns_to_its_start_on_another_heading():
    """Passing back through the start point is a self-intersection, not a loop."""
    G = _grid()
    _, _, start, d = _aimed_near_an_interior_vertex(G)

    path = cast_ray(G, start, START_T, d, both_ways=False, max_steps=8)

    # the position half of the closure test passes -- only the heading half
    # keeps this from being reported as a closed loop
    back = path.hits[4]
    assert np.linalg.norm(back.position - path.hits[0].position) <= default_vertex_tol(G)
    assert np.dot(back.direction_in, path.hits[0].direction_out) < 1 - 1e-9
    assert not path.closed
    assert path.ends[1] == "max_steps"


def test_the_closure_heading_check_is_a_tolerance_and_not_an_equality():
    """Two headings a long path apart are the same heading; 1e-3 radians is not.

    A closed loop on a fixture whose coordinates happen to be exact returns on
    a bitwise-identical heading, so it cannot tell ``> 1 - 1e-9`` from
    ``>= 1.0``.  Drift is what the tolerance is for, so it is tested directly:
    1e-7 radians is far more than a real loop accumulates (measured: ``1 -
    dot`` under 2.2e-16 over the loop fixture at several rotations) and far
    less than the ~4.5e-5 radians the threshold allows.
    """
    origin, d0 = np.zeros(2), np.array([1.0, 0.0])

    def hit_heading(angle):
        d = np.array([np.cos(angle), np.sin(angle)])
        return RayHit(halfedges=[], t=0.0, position=origin, vertex=None, direction_in=d, direction_out=d, face=None)

    assert _closes(hit_heading(1e-7), origin, d0, vertex_tol=1e-9)
    assert not _closes(hit_heading(1e-3), origin, d0, vertex_tol=1e-9)


def test_cast_ray_normalises_the_direction_it_is_given():
    """`_closes` reads a dot product as an angle, so a long `d` must not break it."""
    G, v = _diagonal_loop_grid()
    north = _outgoing_towards(v, [0.0, 1.0])

    path = cast_ray(G, north, 0.5, np.array([-3.0, 0.0]), both_ways=False)

    assert path.closed
    assert len(path.hits) == 9
