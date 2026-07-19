"""Tests for the local ray caster over crease patterns."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import platonic
from pleat.flat_foldable import kawasaki_sum
from pleat.ray_casting import (
    RAY_CREASE,
    DegenerateRayError,
    RayHit,
    _canonical,
    _closes,
    add_ray_creases,
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


def test_cast_ray_closes_when_it_crosses_its_start_edge_obliquely():
    """Closure is about the heading the ray *departs* on, not the one it arrives on.

    Coming back to the start point means arriving from the far side of the start
    edge and transmitting across it, so the ray arrives on
    ``transmit(d0, start_edge)`` and leaves on ``d0``.  Those two coincide only
    when the loop crosses the start edge perpendicularly -- which the square-loop
    fixture above does, and this one deliberately does not.
    """
    G, v = _diagonal_loop_grid()
    north = _outgoing_towards(v, [0.0, 1.0])
    d = np.array([-1.0, 0.35])

    # the test only bites if the crossing really is oblique
    unit = d / np.linalg.norm(d)
    assert not np.allclose(transmit(unit, halfedge_direction(north)), unit, atol=1e-6)

    path = cast_ray(G, north, 0.5, d, both_ways=False)

    assert path.closed
    assert path.ends[1] == "closed"
    assert len(path.hits) == 9
    np.testing.assert_allclose(path.hits[-1].position, path.hits[0].position, atol=1e-12)
    np.testing.assert_allclose(path.hits[-1].direction_out, path.hits[0].direction_out, atol=1e-12)
    # and the arriving heading is genuinely a different one, so testing
    # `direction_in` here would report the loop as still running
    assert np.dot(path.hits[-1].direction_in, path.hits[0].direction_out) < 1 - 1e-9


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


def test_cast_ray_runs_on_past_a_self_intersection_and_closes_only_on_its_own_heading():
    """Passing back through the start point is a self-intersection, not a loop.

    This ray crosses its own start point twice: once mid-path on a different
    heading, which must *not* stop it, and once at the end on the original
    heading, which must.  Stopping at the first would truncate the loop.
    """
    G, v = _diagonal_loop_grid()
    # a diagonal crease at the central vertex, entered off-centre and steeply
    # enough that the trajectory crosses itself before it comes round
    start = _outgoing_towards(v, [-1.0, 1.0])

    path = cast_ray(G, start, 0.7, np.array([2.0, 1.0]), both_ways=False, max_steps=20)

    origin = path.hits[0]
    revisits = [
        i
        for i, hit in enumerate(path.hits[1:], 1)
        if np.linalg.norm(hit.position - origin.position) <= default_vertex_tol(G)
    ]
    assert len(revisits) == 2, "fixture no longer crosses its own start point twice"
    crossing, closure = revisits
    # the position half of the closure test passes at both -- only the heading
    # half tells the self-intersection from the loop
    assert np.dot(path.hits[crossing].direction_out, origin.direction_out) < 1 - 1e-9
    assert np.dot(path.hits[closure].direction_out, origin.direction_out) > 1 - 1e-9
    assert path.closed
    assert closure == len(path.hits) - 1  # it ran on past the self-intersection


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


def _positions(path):
    return np.array([hit.position for hit in path.hits])


def test_both_ways_extends_a_border_to_border_ray():
    G = _grid()
    _, north = _north_edge(G)
    d = np.array([1.0, 0.0])

    one_way = cast_ray(G, north, 0.5, d, both_ways=False)
    backward_only = cast_ray(G, north, 0.5, -d, both_ways=False)
    two_way = cast_ray(G, north, 0.5, d, both_ways=True)

    assert two_way.ends == ("border", "border")
    assert not two_way.closed
    # the two passes spliced, with the shared start point counted once
    assert len(two_way.hits) == len(one_way.hits) + len(backward_only.hits) - 1
    xs = [hit.position[0] for hit in two_way.hits]
    assert xs == sorted(xs)  # ordered backward end -> forward end
    ys = [hit.position[1] for hit in two_way.hits]
    assert np.allclose(ys, ys[0])


def test_both_ways_emits_the_shared_start_point_only_once():
    """Both passes set off from the start, so one of the two copies must be dropped."""
    G = _grid()
    _, north = _north_edge(G)
    two_way = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=True)
    start = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False).hits[0].position

    at_start = [hit for hit in two_way.hits if np.linalg.norm(hit.position - start) <= default_vertex_tol(G)]
    assert len(at_start) == 1
    # and no two consecutive hits coincide anywhere along the path, which is
    # what a surviving duplicate would look like if the splice point moved
    steps = np.linalg.norm(np.diff(_positions(two_way), axis=0), axis=1)
    assert (steps > default_vertex_tol(G)).all()


def test_both_ways_retraces_the_same_line_backwards_by_mirroring_the_side():
    """Casting the same line the other way round must give the same trajectory.

    A ray passing ``+eps`` to the left of a vertex while travelling along ``d``
    is, travelling the other way down that same offset line, passing ``-eps``
    from it -- on its right.  So the two casts below describe one and the same
    physical line and must produce the same hits, in opposite order.  Reusing
    ``side`` unmirrored for the backward pass breaks exactly this.

    The reversed heading is ``-transmit(d, E)`` and not ``-d``: the start point
    lies *on* the start edge ``E``, so the trajectory crosses it there, and the
    heading that retraces the line is the one it arrived on, negated.
    """
    G = _grid()
    _, _, start, d = _aimed_near_an_interior_vertex(G)
    reversed_d = -transmit(d / np.linalg.norm(d), halfedge_direction(start))

    # this ray closes after four crossings, and a closed forward pass suppresses
    # the backward one; the cap keeps both halves in play so there is something
    # to compare
    left = cast_ray(G, start, START_T, d, side="left", max_steps=3)
    right = cast_ray(G, start, START_T, reversed_d, side="right", max_steps=3)

    # the test is only about the side rule if the path actually meets a vertex,
    # and only bites if the two sides really do send the ray different ways
    assert any(hit.vertex is not None for hit in left.hits)
    wrong_side = _positions(cast_ray(G, start, START_T, d, side="right", max_steps=3))
    assert wrong_side.shape != _positions(left).shape or not np.allclose(wrong_side, _positions(left))

    np.testing.assert_allclose(_positions(left), _positions(right)[::-1], atol=1e-9)
    assert left.ends == right.ends[::-1]


def test_both_ways_does_not_cast_backwards_when_the_ray_closes():
    G, v = _diagonal_loop_grid()
    north = _outgoing_towards(v, [0.0, 1.0])
    d = np.array([-1.0, 0.0])

    closed = cast_ray(G, north, 0.5, d, both_ways=True)

    assert closed.closed
    assert closed.ends == ("closed", "closed")
    # a closed loop is its own continuation: nothing is prepended to it
    assert _positions(closed).tolist() == _positions(cast_ray(G, north, 0.5, d, both_ways=False)).tolist()


def test_both_ways_crosses_the_start_edge_rather_than_kinking_at_it():
    """The spliced path is one line through the start point, not two glued at it.

    The start point lies *on* the start edge ``E``, so the trajectory crosses
    ``E`` there: the segment arriving and the segment leaving are related by
    transmission across ``E``, exactly as at every other crossing on the path.
    That is what forces the backward heading to be ``transmit(-d, E)``; sending
    the backward half off on ``-d`` puts a kink here instead, at the one place
    the rim is guaranteed to cross an existing crease.

    Deliberately oblique: with ``d`` perpendicular to ``E`` the transmitted and
    negated headings coincide and there is nothing to see.
    """
    G = _grid()
    v = next(w for w in G.vertices if np.allclose(w["pos"], [-0.5, 0.5], atol=1e-9))
    start = _outgoing_towards(v, [1.0, 0.0])
    d = np.array([1.0, 4.0])

    unit = d / np.linalg.norm(d)
    assert not np.allclose(transmit(unit, halfedge_direction(start)), unit, atol=1e-6)

    path = cast_ray(G, start, 0.25, d, max_steps=60)

    origin = cast_ray(G, start, 0.25, d, both_ways=False).hits[0].position
    (at_start,) = [i for i, h in enumerate(path.hits) if np.linalg.norm(h.position - origin) <= default_vertex_tol(G)]
    assert 0 < at_start < len(path.hits) - 1, "both halves must be present for this to bite"

    def heading(a, b):
        step = path.hits[b].position - path.hits[a].position
        return step / np.linalg.norm(step)

    arriving = heading(at_start - 1, at_start)
    leaving = heading(at_start, at_start + 1)
    np.testing.assert_allclose(leaving, transmit(arriving, halfedge_direction(start)), atol=1e-9)

    # and the whole thing still materialises into a sane graph
    rim, _ = add_ray_creases(G, start, 0.25, d, max_steps=60)
    G.check_consistency()
    _assert_faces_are_sane(G)
    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig


def _on_the_border(hit):
    """True if *hit* crossed a half-edge on the edge of the paper."""
    return any(h.face is None or h.rev.face is None for h in hit.halfedges)


def test_both_ways_reports_the_backward_end_first():
    """`ends` is ordered like `hits`, so each reason must match its own end.

    The ray below runs off the paper going backwards but is still going
    forwards when the cap runs out, so the two reasons are distinct and each is
    checkable against the hit at its end of the path.  The anchor is which
    half-edge each terminal hit crossed -- an orientation-independent field, so
    it survives the re-orientation of the backward half, unlike ``face``.
    """
    G = _grid()
    # a vertical edge one column in from the west border: one crossing takes the
    # backward half off the paper, while the forward half has the width of the
    # sheet still to cross
    west_but_one = sorted({round(float(w["pos"][0]), 6) for w in G.vertices})[1]
    v = next(w for w in G.vertices if np.allclose(w["pos"], [west_but_one, 0.5], atol=1e-9))
    start = _outgoing_towards(v, [0.0, 1.0])

    # the cap is below the crossings the forward half needs, so that end is
    # genuinely still running when it runs out
    path = cast_ray(G, start, 0.5, np.array([1.0, 0.35]), max_steps=2)

    # the backward end left the paper; the forward end was still inside it
    assert path.ends[0] == "border" and _on_the_border(path.hits[0])
    assert path.ends[1] != "border" and not _on_the_border(path.hits[-1])
    assert path.hits[-1].face is not None


def test_both_ways_reorients_the_backward_half_into_the_forward_sense():
    """`hits` must read as one trajectory, not two glued back to back.

    Reversing the *order* of the backward hits is not enough: each one still
    records `direction_in`/`direction_out`/`face` in the sense the backward
    pass was travelling, which is the opposite of the sense `hits` is ordered
    in.
    """
    G = _grid()
    _, north = _north_edge(G)
    d = np.array([1.0, 0.0])

    two_way = cast_ray(G, north, 0.5, d, both_ways=True)

    # this ray meets only vertical creases, so it transmits straight through
    # every one of them: a coherent trajectory heads east at every hit
    assert len(two_way.hits) > 3  # there is a backward half to get wrong
    for hit in two_way.hits:
        assert np.dot(hit.direction_in, d) > 0
        assert np.dot(hit.direction_out, d) > 0

    # and each hit's `face` is the one the next segment runs through, so the
    # off-paper `None` sits at the forward end only
    for here, nxt in zip(two_way.hits, two_way.hits[1:]):
        assert here.face is not None
        assert any(h is nxt.halfedges[0] or h is nxt.halfedges[0].rev for h in here.face.halfedge_iter())
    assert two_way.hits[-1].face is None


def test_both_ways_chains_directions_along_the_segments_between_hits():
    """Each hit's directions must point along the segments that meet at it.

    The straight-line fixture cannot see this: there `direction_in` and
    `direction_out` are equal at every hit, so negating them without also
    swapping them looks identical.  This ray turns at the vertex it hits, in
    both halves, which is what makes the swap observable.
    """
    G = _grid()
    _, _, start, d = _aimed_near_an_interior_vertex(G)

    path = cast_ray(G, start, START_T, d, max_steps=6)

    turns = [h for h in path.hits if not np.allclose(h.direction_in, h.direction_out)]
    assert len(turns) > 1  # otherwise in/out are interchangeable and this is vacuous
    for here, nxt in zip(path.hits, path.hits[1:]):
        segment = nxt.position - here.position
        segment = segment / np.linalg.norm(segment)
        np.testing.assert_allclose(here.direction_out, segment, atol=1e-6)
        np.testing.assert_allclose(nxt.direction_in, segment, atol=1e-6)


def test_both_ways_keeps_the_forward_pass_copy_of_the_start_point():
    """Of the two copies of the shared start, the forward one is the right one.

    Their positions are identical, so nothing geometric distinguishes them --
    but the backward copy points backwards and its `face` is on the other side
    of the start edge.
    """
    G = _grid()
    _, north = _north_edge(G)
    d = np.array([1.0, 0.0])

    two_way = cast_ray(G, north, 0.5, d, both_ways=True)
    start = cast_ray(G, north, 0.5, d, both_ways=False).hits[0]

    (at_start,) = [h for h in two_way.hits if np.linalg.norm(h.position - start.position) <= default_vertex_tol(G)]
    assert np.dot(at_start.direction_in, d) > 0
    assert np.dot(at_start.direction_out, d) > 0
    assert at_start.face is start.face


# ------------------------------------------------ add_ray_creases ------------------------------------------------


def _signed_area(f):
    poly = np.stack([w["pos"] for w in f.vertex_iter()])
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _assert_faces_are_sane(G):
    """Every face must stay simple and positively oriented.

    This is the geometric half of correctness that `check_consistency` cannot
    see: inserting two vertices on one edge in the wrong order leaves the
    topology perfectly consistent but folds the edge back on itself, which
    shows up here as a face of non-positive area.
    """
    for f in G.faces:
        assert _signed_area(f) > 1e-9, f"face {f} has area {_signed_area(f)}"


def _assert_subdivisions_are_ordered(G):
    """Every interior degree-2 vertex must sit *between* its two neighbours.

    Those are exactly the vertices phase 1 inserted along an existing edge.  One
    inserted on the wrong side of an earlier one folds the edge back on itself,
    which leaves the topology consistent and -- while phase 2 has yet to cut any
    face along it -- leaves every face simple too.  The only trace is that the
    two edges at the new vertex run the same way instead of opposite ways.
    """
    for v in G.vertices:
        if v.order() != 2 or v.on_border():
            continue
        a, b = (h.dest["pos"] - v["pos"] for h in v.outgoing_iter())
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cosine < -1 + 1e-9, f"edge folded back at {v['pos']} (cos {cosine:.3f})"


def _crossings_per_edge(path):
    counts: dict[int, int] = {}
    for hit in path.hits:
        if hit.vertex is not None or not hit.halfedges:
            continue
        h = hit.halfedges[0]
        key = min(h["id"], h.rev["id"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def test_add_ray_creases_keeps_the_graph_consistent():
    G = _grid()
    _, north = _north_edge(G)

    before = len(G.vertices)
    rim, path = add_ray_creases(G, north, 0.5, np.array([1.0, 0.0]))

    G.check_consistency()
    _assert_faces_are_sane(G)
    assert len(G.vertices) > before  # vertices were inserted along the way
    assert rim, "expected at least one rim half-edge"
    assert all(h[RAY_CREASE] for h in rim)
    assert all(h.rev[RAY_CREASE] for h in rim)


def test_canonical_agrees_on_both_halves_of_an_edge():
    """Hits reference whichever half the ray crossed, so the grouping key must not.

    Under an identity mapping the two hits on one edge land in two groups, each
    subdividing from its own end, so the second splits a half-edge the first has
    already made stale.
    """
    G = _grid()
    _, north = _north_edge(G)

    assert _canonical(north) is _canonical(north.rev)
    assert _canonical(north) in (north, north.rev)
    assert _canonical(north)["id"] == min(north["id"], north.rev["id"])


def test_add_ray_creases_leaves_a_flat_foldable_vertex_at_an_oblique_start():
    """The backward half departs along ``transmit(-d, E)``, not ``-d``.

    Materializing makes the start point a degree-4 vertex -- the two halves of
    the start edge ``E`` plus the two rim segments -- and the full trajectory
    crosses ``E`` there, arriving on ``transmit(d, E)`` and departing on ``d``.
    Only that pairing satisfies Kawasaki.  Sending the backward half off on
    ``-d`` instead puts a kink in the rim at the one place it is guaranteed to
    cross an existing crease, and the vertex is then not locally flat-foldable.

    A start perpendicular to ``E`` is the degenerate case where
    ``transmit(d, E) == d`` and the difference is invisible, so this ray
    deliberately crosses its start edge obliquely.
    """
    G = _grid()
    v, north = _north_edge(G)
    d = np.array([1.0, 0.35])

    # the test only bites if the crossing really is oblique
    unit = d / np.linalg.norm(d)
    assert not np.allclose(transmit(unit, halfedge_direction(north)), unit, atol=1e-6)

    start_pos = np.asarray(v["pos"], dtype=float) + [0.0, 0.5]
    _, path = add_ray_creases(G, north, 0.5, d)

    # a closed forward pass casts no backward half, which would make this vacuous
    assert not path.closed
    (start,) = [w for w in G.vertices if np.allclose(w["pos"], start_pos, atol=1e-9)]
    assert start.order() == 4  # the two halves of the start edge, and two rim segments
    assert abs(kawasaki_sum(start)) < 1e-9


def test_add_ray_creases_rejects_a_ray_that_crosses_its_own_path():
    """A self-crossing ray cannot be materialized, and must not be half-materialized.

    The earlier chord has already split the face, so a later segment across it
    has one endpoint on each side and no common face.  Skipping the segment
    leaves a gap in the rim, and callers flood-fill bounded by the rim, so a gap
    leaks across the whole sheet.
    """
    G, v = _diagonal_loop_grid()
    corner = next(w for w in G.vertices if np.allclose(w["pos"], v["pos"] + [-1.0, 1.0], atol=1e-9))
    start = _outgoing_towards(corner, [1.0, 0.0])

    # the cast itself is fine; it is materializing it that cannot be done
    path = cast_ray(G, start, 0.5, np.array([-2.0, 1.0]), max_steps=12)
    assert len(path.hits) > 3

    with pytest.raises(DegenerateRayError, match="crosses itself"):
        add_ray_creases(G, start, 0.5, np.array([-2.0, 1.0]), max_steps=12)


def test_add_ray_creases_rim_is_a_connected_path():
    G = _grid()
    _, north = _north_edge(G)
    rim, _ = add_ray_creases(G, north, 0.5, np.array([1.0, 0.0]))

    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig


def test_add_ray_creases_puts_the_rim_where_the_ray_went():
    G = _grid()
    _, north = _north_edge(G)
    rim, path = add_ray_creases(G, north, 0.5, np.array([1.0, 0.0]))

    walked = [rim[0].orig["pos"]] + [h.dest["pos"] for h in rim]
    np.testing.assert_allclose(walked, [hit.position for hit in path.hits], atol=1e-9)


def test_add_ray_creases_tags_only_the_rim():
    G = _grid()
    _, north = _north_edge(G)
    rim, _ = add_ray_creases(G, north, 0.5, np.array([1.0, 0.0]))

    tagged = {h for h in G.halfedges if RAY_CREASE in h}
    assert tagged == {h for h in rim} | {h.rev for h in rim}


def test_add_ray_creases_handles_a_ray_crossing_the_same_edge_twice():
    """The stale-parameter case: a 45-degree ray revisits edges it has split."""
    G = _grid()
    _, north = _north_edge(G)

    rim, path = add_ray_creases(G, north, 0.5, np.array([SQRT_HALF, SQRT_HALF]), max_steps=6)

    counts = _crossings_per_edge(path)
    assert max(counts.values()) >= 2, "fixture no longer re-crosses an edge; test is vacuous"
    G.check_consistency()
    _assert_faces_are_sane(G)
    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig


def test_add_ray_creases_closes_the_rim_on_a_closed_loop():
    G, v = _diagonal_loop_grid()
    start = _outgoing_towards(v, [0.0, 1.0])

    rim, path = add_ray_creases(G, start, 0.5, np.array([1.0, 0.0]))

    assert path.closed
    G.check_consistency()
    _assert_faces_are_sane(G)
    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig
    assert rim[-1].dest is rim[0].orig  # the loop really is a cycle


def test_add_ray_creases_through_a_vertex_reuses_that_vertex():
    G = _grid()
    v, _, start, d = _aimed_near_an_interior_vertex(G)

    before = len(G.vertices)
    rim, path = add_ray_creases(G, start, START_T, d, max_steps=6)

    G.check_consistency()
    _assert_faces_are_sane(G)
    assert any(hit.vertex is v for hit in path.hits)
    # the fan hit lands on the existing vertex rather than inserting a duplicate
    assert any(h.orig is v or h.dest is v for h in rim)
    # every hit got a vertex, and no crossing inserted a second vertex on top of
    # one already there
    assert len(vertices := [w["pos"] for w in G.vertices]) > before
    for j, first in enumerate(vertices):
        for second in vertices[j + 1 :]:
            assert np.linalg.norm(first - second) > default_vertex_tol(G)


@pytest.mark.slow
def test_add_ray_creases_does_not_double_up_on_a_retraced_segment():
    """A trajectory that runs the same chord twice must reuse the crease it laid.

    Materializing the retraced segment a second time would lay a duplicate edge
    over the first and leave a zero-area face behind.  A retrace needs a
    self-overlapping trajectory, which on a pattern this simple only turns up
    once several rays have already subdivided it -- so the case is reached by
    laying rays until it happens rather than by a single hand-picked cast.  With
    the reuse removed, four or five of the fifty seeds below corrupt the graph;
    with it in place, none do.
    """
    for seed in range(50):
        rng = np.random.default_rng(seed)
        G, _ = _diagonal_loop_grid()
        for _ in range(14):
            candidates = [h for h in G.halfedges if h.face is not None and h.rev.face is not None]
            start = candidates[int(rng.integers(len(candidates)))]
            angle = float(rng.uniform(0.0, 2 * np.pi))
            try:
                rim, _ = add_ray_creases(
                    G, start, float(rng.uniform(0.2, 0.8)), np.array([np.cos(angle), np.sin(angle)]), max_steps=60
                )
            except DegenerateRayError:
                continue
            G.check_consistency()
            # a doubled-up segment is two half-edges with the same endpoints, and
            # the zero-area face between them.  Slivers are legitimate on a
            # pattern this heavily subdivided, so the area bound is only `> 0`.
            for w in G.vertices:
                destinations = [h.dest for h in w.outgoing_iter()]
                assert len(destinations) == len(set(destinations)), "duplicate edge laid over an existing one"
            for f in G.faces:
                assert _signed_area(f) > 0.0, f"face {f} has area {_signed_area(f)}"


def _edges_hit_from_both_halves(path):
    """Return the undirected edges the path crossed at distinct points on both halves."""
    groups: dict[int, list[tuple[bool, np.ndarray]]] = {}
    for hit in path.hits:
        if hit.vertex is not None or not hit.halfedges:
            continue
        h = hit.halfedges[0]
        canonical = h["id"] < h.rev["id"]
        groups.setdefault(min(h["id"], h.rev["id"]), []).append((canonical, hit.position))
    return [
        key
        for key, entries in groups.items()
        if len({side for side, _ in entries}) > 1
        and max(np.linalg.norm(a - b) for _, a in entries for _, b in entries) > 1e-6
    ]


def test_add_ray_creases_orders_hits_recorded_on_opposite_halves_of_one_edge():
    """Two hits on one edge may be recorded on opposite half-edges.

    Phase 1 subdivides an edge left to right and walks the tail, so it must
    order the hits along the edge itself, not by each hit's own ``t`` -- ``t``
    is measured on whichever half the hit happened to reference, so the two
    would otherwise be sorted against each other backwards and the second
    vertex would land on the wrong side of the first, folding the edge back on
    itself.

    Every mixed-halves ray found also crosses itself, so phase 2 rejects it
    (0 materialisable mixed-halves casts in 15744 across five fixtures: the
    square grid, the diagonal-loop grid and ``rosette(5..7)``).  Phase 1 has run
    to completion by then, though, and a folded-back edge is visible in the
    graph it left behind -- so the ordering is still pinned here, on the state
    after the raise, rather than being dropped.
    """
    G, v = _diagonal_loop_grid()
    # the north side of the square north-west of the central vertex, running east
    corner = next(w for w in G.vertices if np.allclose(w["pos"], v["pos"] + [-1.0, 1.0], atol=1e-9))
    start = _outgoing_towards(corner, [1.0, 0.0])

    path = cast_ray(G, start, 0.5, np.array([-2.0, 1.0]), max_steps=12)
    assert _edges_hit_from_both_halves(path), "fixture no longer exercises the mixed-orientation case"

    with pytest.raises(DegenerateRayError):
        add_ray_creases(G, start, 0.5, np.array([-2.0, 1.0]), max_steps=12)

    G.check_consistency()
    # a folded-back edge leaves the topology consistent and shows up only here
    _assert_subdivisions_are_ordered(G)
