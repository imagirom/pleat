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
    RayPath,
    _canonical,
    _closes,
    _reject_self_crossing,
    add_ray_creases,
    cast_ray,
    cross2,
    default_vertex_tol,
    fan_at_vertex,
    first_crossing,
    _point_on,
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


#: A deliberately awkward, non-unit scale for the grid fixtures.
#:
#: Every edge of the raw grid has length exactly ``1.0``, which silently turns
#: the six length-scalings in ``first_crossing`` and ``_walk`` -- the parallel
#: guards, the ``s``-range slack, the vertex-snap distance -- into no-ops:
#: dividing by ``1.0`` cannot be observed, so no test on the raw grid can tell
#: the scaled form from the unscaled one.  The geometry tests below are
#: parametrized over :data:`SCALES` so that they run at least once where the
#: division is not the identity.  A round factor is avoided on purpose, so that
#: nothing cancels against a tolerance by luck.
SCALE = 0.37

#: The scales the geometry tests are parametrized over.
SCALES = [1.0, SCALE]


def test_transmit_rejects_a_zero_length_crease():
    """A zero-length crease has no direction; normalising it hands back ``nan``.

    Reachable in principle from the start-vertex snap's failure mode, which
    manufactures exactly such an edge, so it fails loudly rather than poisoning
    the direction and every hit after it.
    """
    with pytest.raises(DegenerateRayError, match="zero-length crease"):
        transmit(np.array([1.0, 0.0]), np.zeros(2))


def _grid(scale: float = 1.0):
    """Square grid of edge length *scale*; the central vertex has four 90-degree sectors."""
    G = from_tiles(platonic(n=4), rings=2)
    if scale != 1.0:
        for w in G.vertices:
            w["pos"] = np.asarray(w["pos"], dtype=float) * scale
    G.recompute_lengths_and_angles()
    return G


def _scale(G):
    """Return the grid's edge length, so the fixtures can be written in grid units."""
    return min(float(np.linalg.norm(halfedge_direction(h))) for h in G.halfedges)


def _at(G, pos):
    """Return the vertex at *pos*, given in units of the grid's edge length."""
    target = np.asarray(pos, dtype=float) * _scale(G)
    return next(w for w in G.vertices if np.allclose(w["pos"], target, atol=1e-9))


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _outgoing_towards(v, offset):
    """Return the outgoing half-edge at *v* running in the direction *offset*.

    Matched by direction rather than by destination position, so the same
    fixture code reads at any grid scale.
    """
    u = _unit(offset)
    return next(h for h in v.outgoing_iter() if np.allclose(_unit(halfedge_direction(h)), u, atol=1e-9))


def _subdivide_corner(G, v, face, offset):
    """Crease *face* from *v* to its corner in direction *offset*; return the new half-edge."""
    u = _unit(offset)
    far_corner = next(
        w for w in face.vertex_iter() if w is not v and np.allclose(_unit(w["pos"] - v["pos"]), u, atol=1e-9)
    )
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
    v = _at(G, [-1.5, 2.5])
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


@pytest.mark.timeout(120)
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
    """Return the square face whose top-right corner is *v*, and its centre."""
    west = _outgoing_towards(v, [-1.0, 0.0])
    return west.face, np.mean(np.stack([w["pos"] for w in west.face.vertex_iter()]), axis=0)


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_finds_the_forward_edge_of_the_face(scale):
    G = _grid(scale)
    v = G.central_vertex()
    face, p = _sw_face(v)

    h, s = first_crossing(face, p, np.array([1.0, 0.0]), vertex_tol=1e-9 * scale)

    # the east edge of the face runs between v and the corner south of it
    corner = _outgoing_towards(v, [0.0, -1.0]).dest["pos"]
    assert np.allclose(h.orig["pos"], corner) or np.allclose(h.dest["pos"], corner)
    crossing = h.orig["pos"] + s * halfedge_direction(h)
    np.testing.assert_allclose(crossing, 0.5 * (np.asarray(v["pos"], dtype=float) + corner), atol=1e-12)


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_ignores_edges_behind_the_ray(scale):
    G = _grid(scale)
    face, p = _sw_face(G.central_vertex())

    forward, _ = first_crossing(face, p, np.array([1.0, 0.0]), vertex_tol=1e-9 * scale)
    backward, _ = first_crossing(face, p, np.array([-1.0, 0.0]), vertex_tol=1e-9 * scale)
    assert forward is not backward


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_returns_t_within_the_unit_interval(scale):
    G = _grid(scale)
    face, p = _sw_face(G.central_vertex())

    _, s = first_crossing(face, p, np.array([0.3, 1.0]), vertex_tol=1e-9 * scale)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_is_invariant_to_the_length_of_the_direction(scale):
    """*d* is normalised, so ``vertex_tol`` is a distance and ``|d|`` cannot matter.

    The scaling is chosen to bite: unnormalised, the ray parameter of this
    crossing would be ``0.5 * scale / |d|``, which for the long ``d`` below is
    far *under* ``vertex_tol`` and would be thrown away as "too close to the
    start".  Normalising is what makes the same crossing come back either way.
    """
    G = _grid(scale)
    face, p = _sw_face(G.central_vertex())
    d = np.array([0.3, 1.0])
    vertex_tol = 1e-3 * scale

    h1, s1 = first_crossing(face, p, d, vertex_tol=vertex_tol)
    h2, s2 = first_crossing(face, p, d * 1e4, vertex_tol=vertex_tol)
    assert np.linalg.norm(d * 1e4) * vertex_tol > 0.5 * scale, "the long direction no longer bites"
    assert h1 is h2
    assert s1 == pytest.approx(s2)


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_skips_an_edge_closer_than_vertex_tol(scale):
    """The slack is measured in distance along the ray, not in ray parameter."""
    G = _grid(scale)
    v = G.central_vertex()
    face, _ = _sw_face(v)
    south = _outgoing_towards(v, [0.0, -1.0]).dest["pos"]
    midpoint = 0.5 * (np.asarray(v["pos"], dtype=float) + south)
    # a hair west of the east edge of the face, aimed east with a tiny |d|
    p = midpoint + [-1e-6 * scale, 0.0]
    d = np.array([1e-4, 0.0])

    assert first_crossing(face, p, d, vertex_tol=1e-3 * scale) is None
    h, _ = first_crossing(face, p, d, vertex_tol=1e-9 * scale)
    assert np.allclose(h.orig["pos"] + halfedge_direction(h) / 2, midpoint)


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_takes_the_nearest_of_two_candidates(scale):
    """Passing within ``vertex_tol`` of a corner puts *both* its edges in range.

    That is the only way a convex face offers two candidates at once -- from an
    inside point the ray's line meets one edge forwards and one backwards, and
    the backward one is thrown out as too close to the start.  So this is the
    one shape in which "keep the nearest" is distinguishable from "keep the last
    one iterated", and the two aims below straddle the corner so that the answer
    comes before the other candidate in iteration order for one of them.

    Which edge is right is hand-derivable: aiming just *above* the corner, the
    ray reaches the horizontal edge (a hair west of the corner) before the
    vertical one; aiming just below, the other way round.
    """
    G = _grid(scale)
    v = _at(G, [-0.5, -0.5])  # not `central_vertex`, which is tied four ways here
    face, centre = _sw_face(v)
    north = _outgoing_towards(v, [-1.0, 0.0])  # the face's north edge, running west
    east = _outgoing_towards(v, [0.0, -1.0]).rev  # its east edge, running north
    assert {north, east} <= set(face.halfedge_iter())
    vertex_tol = 1e-6

    for offset, expected in (([0.0, 0.6 * vertex_tol], north), ([0.0, -0.6 * vertex_tol], east)):
        target = np.asarray(v["pos"], dtype=float) + offset
        h, _ = first_crossing(face, centre, target - centre, vertex_tol=vertex_tol)
        assert h is expected, f"aiming at {offset} took the far candidate"


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_keeps_a_crossing_that_overshoots_an_endpoint_by_less_than_vertex_tol(scale):
    """``vertex_tol`` is a distance, so the ``s``-range slack is ``vertex_tol / |e|``.

    A ray passing just outside a corner leaves the face through no edge at all
    in exact arithmetic; the slack is what turns that into a crossing clamped to
    the endpoint, which the walk then reports as a vertex hit.  Comparing
    ``vertex_tol`` against ``s`` directly -- forgetting the division -- is
    invisible on the raw grid, where ``|e|`` is exactly ``1.0``, and wrong by a
    factor of ``|e|`` anywhere else.

    Only one candidate edge is in play here: the ray starts *on* the other edge
    at the corner, so that crossing sits at ``t = 0`` and is discarded as too
    close to the start.
    """
    G = _grid(scale)
    v = G.central_vertex()
    face, _ = _sw_face(v)
    edge_len = _scale(G)
    vertex_tol = 1e-6
    # from the midpoint of the face's north edge, aimed to pass 0.6 * vertex_tol
    # *above* the corner -- outside the face, and inside the slack
    p = np.asarray(v["pos"], dtype=float) - [0.5 * edge_len, 0.0]
    slope = 1.2 * vertex_tol / edge_len
    d = np.array([1.0, slope])

    found = first_crossing(face, p, d, vertex_tol=vertex_tol)

    assert found is not None, "the overshooting crossing was thrown away"
    h, s = found
    assert s in (0.0, 1.0), "an overshoot must be clamped to the endpoint"
    np.testing.assert_allclose(_point_on(h, s), v["pos"], atol=1e-12)


@pytest.mark.parametrize("scale", SCALES)
def test_first_crossing_finds_an_edge_it_crosses_at_a_shallow_angle(scale):
    """The parallel guard rejects only what is *numerically* parallel.

    ``|denom| = |e| sin(angle)``, so the threshold has to scale with the edge or
    it means a different angle on every edge.  It is deliberately at the noise
    floor rather than at a "shallow enough to ignore" angle: an edge crossed at
    a millidegree is still crossed, and skipping it sends the ray out through
    the wrong side of the face.
    """
    G = _grid(scale)
    v = G.central_vertex()
    face, _ = _sw_face(v)
    edge_len = _scale(G)
    north_west = np.asarray(v["pos"], dtype=float) - [edge_len, 0.0]
    # just under the face's north edge, climbing steeply enough to cross it well
    # before the east edge but at a very shallow angle to it
    p = north_west + [0.05 * edge_len, -1e-5 * edge_len]
    d = np.array([1.0, 1e-4])

    h, s = first_crossing(face, p, d, vertex_tol=1e-9 * edge_len)

    crossing = _point_on(h, s)
    np.testing.assert_allclose(crossing[1], v["pos"][1], atol=1e-9 * edge_len)
    assert abs(cross2(_unit(d), halfedge_direction(h))) < 1e-3, "the crossing is no longer shallow"


def test_first_crossing_rejects_a_zero_length_direction():
    """Normalising would warn and hand back ``nan``, reported as "no way out"."""
    G = _grid()
    face, p = _sw_face(G.central_vertex())

    with pytest.raises(DegenerateRayError, match="no direction"):
        first_crossing(face, p, np.zeros(2), vertex_tol=1e-9)


def _north_edge(G):
    """Return the central vertex and its outgoing half-edge pointing north."""
    v = G.central_vertex()
    return v, _outgoing_towards(v, [0.0, 1.0])


@pytest.mark.parametrize("scale", SCALES)
def test_cast_ray_straight_across_a_grid_reaches_the_border(scale):
    G = _grid(scale)
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


@pytest.mark.timeout(120)
def test_cast_ray_respects_max_steps():
    G = _grid()
    _, north = _north_edge(G)
    # one step is always short: `central_vertex` is tied four ways on this
    # grid, so the ray has either two or three creases left to cross
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False, max_steps=1)

    assert path.ends[1] == "max_steps"
    assert len(path.hits) == 2  # the start plus one step


@pytest.mark.timeout(120)
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


#: ``(t, direction)`` pairs putting the start of a ray at an endpoint of its
#: start edge.  ``t`` is a parameter, so the near-miss values are converted to a
#: distance against the edge length by the fixture below.  The two directions
#: differ because the ray sets off into ``north.face`` either way, and that face
#: lies west of the edge: from ``north.orig`` its sector is the second quadrant,
#: from ``north.dest`` the third.
ENDPOINT_STARTS = [
    ("orig", 0.0, [-1.0, 0.4]),
    ("dest", 1.0, [-1.0, -0.4]),
    ("near orig", 0.4, [-1.0, 0.4]),  # 0.4 * vertex_tol away from `orig`, as a distance
    ("near dest", -0.4, [-1.0, -0.4]),  # 0.4 * vertex_tol away from `dest`
]
ENDPOINT_IDS = [case[0] for case in ENDPOINT_STARTS]


def _endpoint_start(G, h, raw_t):
    """Return ``(t, vertex)`` for an :data:`ENDPOINT_STARTS` entry on half-edge *h*.

    The near-miss entries are given as a fraction of ``vertex_tol``, which is a
    *distance*, so they are converted against the edge length -- the same
    conversion the snap itself makes, and the reason the fixture reads the same
    at every grid scale.
    """
    if raw_t in (0.0, 1.0):
        t = raw_t
    else:
        slack = abs(raw_t) * default_vertex_tol(G) / float(np.linalg.norm(halfedge_direction(h)))
        t = slack if raw_t > 0 else 1.0 - slack
    return t, (h.orig if t < 0.5 else h.dest)


def _coincident_vertices(G, tol):
    positions = [np.asarray(w["pos"], dtype=float) for w in G.vertices]
    return [(a, b) for i, a in enumerate(positions) for b in positions[i + 1 :] if np.linalg.norm(a - b) <= tol]


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("_which, raw_t, direction", ENDPOINT_STARTS, ids=ENDPOINT_IDS)
def test_cast_ray_starting_at_an_edge_endpoint_snaps_to_that_vertex(scale, _which, raw_t, direction):
    """``t = 0`` and ``t = 1`` mean the endpoints, so the start hit *is* a vertex hit.

    The start is the one hit that never passes through ``first_crossing``: its
    parameter is whatever the caller passed, so it is the one hit that can land
    on an endpoint without having been snapped.  Left unsnapped it reports
    ``vertex=None`` while sitting exactly on a vertex.
    """
    G = _grid(scale)
    _, north = _north_edge(G)
    t, endpoint = _endpoint_start(G, north, raw_t)

    path = cast_ray(G, north, t, np.array(direction), both_ways=False)

    assert path.hits[0].vertex is endpoint
    np.testing.assert_array_equal(path.hits[0].position, endpoint["pos"])


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("_which, raw_t, direction", ENDPOINT_STARTS, ids=ENDPOINT_IDS)
def test_add_ray_creases_at_an_edge_endpoint_adds_no_duplicate_vertex(scale, _which, raw_t, direction):
    """The graph-corrupting half of the same bug.

    Phase 1 snaps only against the *origin* end of each edge, on the argument
    that ``_walk`` has already reported anything nearer an endpoint than
    ``vertex_tol`` as a vertex hit.  That was true of every hit except the
    start, whose parameter is the caller's: so a start at the other end
    subdivided the edge at a point already occupied, leaving a duplicate vertex,
    a zero-length edge and two zero-area faces on a graph that still passes
    ``check_consistency``.  Which of ``t = 0`` / ``t = 1`` broke depended on the
    edge's id order, so it was not even predictable from the caller's side.
    """
    G = _grid(scale)
    _, north = _north_edge(G)
    vertex_tol = default_vertex_tol(G)
    t, endpoint = _endpoint_start(G, north, raw_t)

    rim, _ = add_ray_creases(G, north, t, np.array(direction))

    G.check_consistency()
    _assert_faces_are_sane(G)
    assert not _coincident_vertices(G, vertex_tol), "a second vertex was inserted on top of an existing one"
    shortest = min(float(np.linalg.norm(halfedge_direction(h))) for h in G.halfedges)
    assert shortest > vertex_tol, f"zero-length edge of length {shortest}"
    assert endpoint in {w for h in rim for w in (h.orig, h.dest)}, "the rim does not run through the start vertex"
    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig


def test_cast_ray_rejects_a_zero_length_direction():
    G = _grid()
    _, north = _north_edge(G)

    with pytest.raises(DegenerateRayError, match="no direction"):
        cast_ray(G, north, 0.5, np.zeros(2), both_ways=False)


def test_cast_ray_threads_angle_tol_through_to_the_fan():
    """``angle_tol`` is one of the module's two tolerances; both must be reachable.

    Widened far enough, the fan's near-collinear guard swallows an arrival that
    is nowhere near a crease, which is a change no other argument can produce --
    so this pins that the value really does travel from here to
    ``fan_at_vertex`` rather than sitting at its default.
    """
    G = _grid()
    _, _, start, d = _aimed_near_an_interior_vertex(G)

    assert cast_ray(G, start, START_T, d, both_ways=False, max_steps=1).hits[1].vertex is not None

    with pytest.raises(DegenerateRayError, match="along a crease"):
        cast_ray(G, start, START_T, d, both_ways=False, max_steps=1, angle_tol=1.0)


def test_add_ray_creases_threads_angle_tol_through_to_the_fan():
    G = _grid()
    _, _, start, d = _aimed_near_an_interior_vertex(G)

    with pytest.raises(DegenerateRayError, match="along a crease"):
        add_ray_creases(G, start, START_T, d, max_steps=1, angle_tol=1.0)


def test_cast_ray_rejects_a_direction_along_its_own_start_edge():
    G = _grid()
    _, north = _north_edge(G)

    with pytest.raises(DegenerateRayError, match="along its own start edge"):
        cast_ray(G, north, 0.5, halfedge_direction(north), both_ways=False)


def test_the_along_the_start_edge_test_is_an_angle_and_not_a_distance():
    """``|start_edge x d| = |E| sin(angle)``, so the threshold has to scale with ``|E|``.

    Dropping the ``|E|`` turns one angular rule into a different rule per edge:
    lax on a long edge, strict on a short one.  This grid is a hundred times the
    usual size, so the two forms disagree by a factor of a hundred, and a
    direction whose *angle* is well inside the bound is only caught by the
    scaled form.
    """
    G = _grid(100.0)
    _, north = _north_edge(G)
    along = _unit(halfedge_direction(north))
    normal = np.array([-along[1], along[0]])

    with pytest.raises(DegenerateRayError, match="along its own start edge"):
        cast_ray(G, north, 0.5, along + 5e-13 * normal, both_ways=False)

    # and it really is a threshold: a hundredfold wider angle is let through
    cast_ray(G, north, 0.5, along + 5e-11 * normal, both_ways=False, max_steps=1)


#: Directions at a degree-4 node of the square grid, in degrees: the diagonals
#: point strictly into one of the four sectors, the axes run along a crease.
SECTOR_HEADINGS = [45, 135, 225, 315]
CREASE_HEADINGS = [0, 90, 180, 270]


def _heading(degrees):
    return np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])


def _node_starts(G):
    """Return an interior degree-4 node and every ``(halfedge, t)`` naming it.

    Both senses of all four incident edges, so a cast can be made from the node
    through any of the eight primitives that resolve to it.  The vertex is
    picked by position: ``central_vertex`` is tied four ways on this grid.
    """
    v = _at(G, [0.5, 0.5])
    incident = [_outgoing_towards(v, off) for off in ([0, 1], [1, 0], [0, -1], [-1, 0])]
    return v, [(h, 0.0) for h in incident] + [(h.rev, 1.0) for h in incident]


@pytest.mark.parametrize("degrees", SECTOR_HEADINGS)
def test_cast_ray_from_a_node_sets_off_into_the_sector_that_holds_the_direction(degrees):
    """A node start must serve every direction, not just the two sectors touching one edge.

    Choosing the start face by the *side* of the start edge only distinguishes
    the two sectors adjacent to it, so at a degree-4 node two of the four
    diagonals used to leave the ray in a face it was pointing out of, with no
    forward crossing to find: reported ``"stalled"`` after a single hit.
    """
    G = _grid()
    v, starts = _node_starts(G)
    north, t = starts[0]

    path = cast_ray(G, north, t, _heading(degrees), both_ways=False)

    assert path.hits[0].vertex is v
    assert path.ends[1] == "closed", "the diagonal loop around an interior node closes"
    np.testing.assert_allclose(path.hits[1].position - v["pos"], np.sqrt(2.0) * _heading(degrees), atol=1e-9)


@pytest.mark.parametrize("degrees", SECTOR_HEADINGS)
def test_cast_ray_from_a_node_does_not_depend_on_which_incident_halfedge_names_it(degrees):
    """The node and the direction determine the ray; the primitive naming the node does not.

    A caller holding a node has four edges to choose from and nothing tells them
    which one works, so any of them has to give the same trace.  (One-way: the
    backward heading is ``-transmit(d, E)``, which is a genuine function of the
    start edge, so only the forward half is comparable.)
    """
    G = _grid()
    _, starts = _node_starts(G)
    traces = [
        np.stack([hit.position for hit in cast_ray(G, h, t, _heading(degrees), both_ways=False).hits])
        for h, t in starts
    ]

    for other in traces[1:]:
        np.testing.assert_allclose(other, traces[0], atol=1e-9)


@pytest.mark.parametrize("degrees", [10, 45, 80])
def test_cast_ray_from_a_node_reads_the_sector_from_its_bounds_and_not_from_the_nearest_crease(degrees):
    """Every direction strictly inside a sector is in it, including near its far bound.

    A rule that takes the *nearest* incident crease instead of the nearest one
    clockwise agrees on a direction bisecting its sector and disagrees on one
    that leans: at 80 degrees the nearest crease is the sector's
    counter-clockwise bound, and that crease's face is the next sector round.
    """
    G = _grid()
    v, starts = _node_starts(G)
    north_east = _outgoing_towards(v, [1, 0]).face  # `h.face` is left of `h`: the 0..90 sector

    for h, t in starts:
        path = cast_ray(G, h, t, _heading(degrees), both_ways=False)

        assert path.hits[0].face is north_east
        assert path.ends[1] in ("closed", "border")


@pytest.mark.parametrize("degrees", CREASE_HEADINGS)
def test_cast_ray_from_a_node_along_a_crease_is_degenerate_whichever_halfedge_names_it(degrees):
    """Setting off along a crease is degenerate wherever that crease sits in the fan.

    It used to be reported as one thing when the crease happened to be the start
    edge ("along its own start edge") and as another, a step later and at a
    different vertex, when it did not ("arrives at ... along a crease") -- for
    the same geometry seen from a different primitive.
    """
    G = _grid()
    _, starts = _node_starts(G)

    for h, t in starts:
        with pytest.raises(DegenerateRayError, match="sets off from .* along a crease"):
            cast_ray(G, h, t, _heading(degrees), both_ways=False)


def _diagonal_loop_grid(scale: float = 1.0):
    """Grid with all four diagonals at the central vertex.

    Each diagonal turns a passing ray by 90 degrees, and the four together
    steer a ray around a closed square loop of side *scale* centred on the
    vertex.
    """
    G = _grid(scale)
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


@pytest.mark.parametrize("scale", SCALES)
def test_cast_ray_closes_on_a_square_loop(scale):
    G, v = _diagonal_loop_grid(scale)
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


def test_cast_ray_reports_a_stall_rather_than_calling_it_a_border():
    """A ray that cannot leave its face has not reached the paper's edge.

    The end value is ``"stalled"`` and not ``"degenerate"``: ``ends`` reports
    how a cast that *returned* stopped, while ``DegenerateRayError`` is what a
    cast that never returns raises, over a disjoint set of causes.
    """
    G = _grid()
    _, north = _north_edge(G)
    # A tolerance wider than the face swallows every candidate crossing.  It
    # also snaps the start itself to `north.orig`, so the heading is taken off
    # the axes: due east from a node runs along that node's east crease, which
    # is a degenerate *start*, and the cast would raise before it could stall.
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.35]), both_ways=False, vertex_tol=10.0)

    assert path.ends[1] == "stalled"
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
    v = _at(G, [-0.5, 0.5])
    west = _outgoing_towards(v, [-1.0, 0.0])
    return v, west, _outgoing_towards(west.dest, [0.0, -1.0]), np.array([1.0, 0.5])


@pytest.mark.parametrize("scale", SCALES)
def test_cast_ray_landing_on_a_vertex_resolves_it_with_the_fan(scale):
    """The vertex branch of the walk, reached through `cast_ray` rather than directly."""
    G = _grid(scale)
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


@pytest.mark.timeout(120)
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
    checked = 0
    for here, nxt in zip(two_way.hits, two_way.hits[1:]):
        assert here.face is not None
        if not nxt.halfedges:
            continue  # a fan grazing a corner crosses nothing, so there is no edge to find
        crossed = nxt.halfedges[0]
        assert any(h is crossed or h is crossed.rev for h in here.face.halfedge_iter())
        checked += 1
    assert checked > 2, "every hit grazed; the face check above never ran"
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


@pytest.mark.parametrize("scale", SCALES)
def test_add_ray_creases_keeps_the_graph_consistent(scale):
    G = _grid(scale)
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


def _polyline(*points):
    """A ``RayPath`` that is nothing but a polyline.

    ``_reject_self_crossing`` reads only ``hit.position``, so the geometry can
    be stated directly instead of being coaxed out of a fixture -- which is what
    makes the boundary cases below reachable at all: a ray that produces a
    T-junction with itself is hard to aim and easy to lose to an unrelated
    fixture change.
    """
    hits = [
        RayHit(
            halfedges=[],
            t=0.0,
            position=np.asarray(p, dtype=float),
            vertex=None,
            direction_in=np.zeros(2),
            direction_out=np.zeros(2),
            face=None,
        )
        for p in points
    ]
    return RayPath(hits=hits, closed=False, ends=("border", "border"))


#: ``(name, points, rejected)``.  The rule is that an intersection is ignored
#: only when it lands on an endpoint of **both** segments; every case below
#: turns on that "both".  Weakening the ``and`` to an ``or`` admits the
#: T-junction and the collinear overlap, which is the gapped-rim failure the
#: guard exists to stop; dropping the exclusion altogether rejects the closed
#: loop, which is a rim every sink of a closed ray produces.
SELF_CROSSINGS = [
    # a proper X: the meeting point is interior to both segments
    ("proper crossing", [(-1, 0), (1, 0), (1, 2), (0, -1)], True),
    # the last segment *ends* mid-way along the first: an endpoint of one only
    ("T junction", [(-1, 0), (1, 0), (0, 2), (0, 0)], True),
    # doubling back along a segment already laid, overlapping it in part
    ("collinear overlap", [(0, 0), (2, 0), (3, 0), (1, 0)], True),
    # an ordinary zigzag meets nothing at all
    ("zigzag", [(0, 0), (1, 0), (1, 1), (2, 1)], False),
    # the seam of a closed rim: first and last segment share a corner, and that
    # corner is an endpoint of both
    ("closed loop seam", [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], False),
]


@pytest.mark.parametrize("name, points, rejected", SELF_CROSSINGS, ids=[case[0] for case in SELF_CROSSINGS])
def test_reject_self_crossing_ignores_shared_endpoints_and_nothing_else(name, points, rejected):
    path = _polyline(*points)

    if rejected:
        with pytest.raises(DegenerateRayError, match="crosses itself"):
            _reject_self_crossing(path, vertex_tol=1e-9)
    else:
        _reject_self_crossing(path, vertex_tol=1e-9)


@pytest.mark.timeout(120)
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


def test_add_ray_creases_leaves_the_graph_untouched_when_the_ray_crosses_itself():
    """A rejected ray must not leave *G* part-materialized.

    The crossing is a property of the cast trajectory alone, so it is decided
    before phase 1 runs.  Detecting it mid-phase-2 instead -- where the missing
    common face shows up -- left the graph subdivided and part-creased, with
    nothing to identify and undo, and with the topology still consistent so that
    no check downstream noticed.
    """
    G, v = _diagonal_loop_grid()
    corner = next(w for w in G.vertices if np.allclose(w["pos"], v["pos"] + [-1.0, 1.0], atol=1e-9))
    start = _outgoing_towards(corner, [1.0, 0.0])
    before = (len(G.vertices), len(G.halfedges), len(G.faces))

    with pytest.raises(DegenerateRayError, match="crosses itself"):
        add_ray_creases(G, start, 0.5, np.array([-2.0, 1.0]), max_steps=12)

    assert (len(G.vertices), len(G.halfedges), len(G.faces)) == before
    assert not [h for h in G.halfedges if RAY_CREASE in h]
    G.check_consistency()


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


@pytest.mark.parametrize("scale", SCALES)
def test_add_ray_creases_through_a_vertex_reuses_that_vertex(scale):
    G = _grid(scale)
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
@pytest.mark.timeout(120)
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

    Every mixed-halves ray found also crosses itself (0 materialisable
    mixed-halves casts in 15744 across five fixtures: the square grid, the
    diagonal-loop grid and ``rosette(5..7)``), and the self-crossing pre-flight
    now rejects those on the trajectory, before phase 1 runs -- so there is no
    graph state left behind to inspect and this test can only pin the rejection.
    Nothing else pins the ordering either: mutating it to sort by ``hit.t`` now
    leaves the whole suite green, because every ray that would expose it is
    rejected first.  The ordering stays because it is correct by construction
    and free; it is, for now, unobservable.
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
    _assert_subdivisions_are_ordered(G)
