"""Tests for the open sink fold."""

from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import pytest

from pleat.cutting import pointinpolygon
from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.flat_foldable import is_locally_flat_foldable, local_assignment_valid
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from pleat.ray_casting import RAY_CREASE, DegenerateRayError, add_ray_creases
from pleat.shrink_rotate import assign_this_way_by_bfs, shrink_rotate_pattern
from pleat.sink import InvalidSinkError, _interior_faces, _rim_is_ccw, open_sink

from .test_ray_casting import (
    START_T,
    _aimed_near_an_interior_vertex,
    _diagonal_loop_grid,
    _grid,
    _outgoing_towards,
)

DIAGONALS = ([-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0])
AXES = ([0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0])


def _interior_vertex(G):
    """The interior grid vertex at ``(-0.5, 0.5)``.

    Deliberately not ``G.central_vertex()``: that is an ``argmin`` over a set
    with a four-way tie on this grid, so which of ``(+-0.5, +-0.5)`` it returns
    varies between runs -- and the open-rim tests below need to know which side
    of the sheet the rim cuts off.
    """
    return next(w for w in G.vertices if np.allclose(w["pos"], [-0.5, 0.5], atol=1e-9))


def _assign_all(G, value=MOUNTAIN):
    for h in G.halfedges:
        h[CREASE_ASSIGNMENT] = value


def _assign_radial(v, offsets, value):
    for offset in offsets:
        h = _outgoing_towards(v, offset)
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = value


def _sink_fixture(diagonals=MOUNTAIN):
    """The square-loop sink: a closed rim of side 1 around a degree-8 vertex.

    The ray leaves the midpoint of the north crease heading west and turns left
    at each of the four diagonals, so the rim crosses all eight radial creases:
    four head-on (90-degree sectors, which constrain nothing) and four at the
    corners, where the sectors are 135/45/45/135 and big-little-big forces the
    rim to match the outer half of the diagonal.
    """
    G, v = _diagonal_loop_grid()
    _assign_all(G, MOUNTAIN)
    _assign_radial(v, DIAGONALS, diagonals)
    start = _outgoing_towards(v, [0.0, 1.0])
    return G, v, start


def _radial_halves(v):
    """Both half-edges of every edge at *v* -- exactly the creases inside the rim."""
    outgoing = list(v.outgoing_iter())
    return {h for h in outgoing} | {h.rev for h in outgoing}


def _both_halves_agree(G):
    return all(h.get(CREASE_ASSIGNMENT) == h.rev.get(CREASE_ASSIGNMENT) for h in G.halfedges)


# ------------------------------------------------ closed rim ------------------------------------------------


def test_open_sink_keeps_the_graph_consistent_and_closes_the_rim():
    G, v, start = _sink_fixture()

    rim = open_sink(G, start, 0.5, np.array([-1.0, 0.0]))

    G.check_consistency()
    assert len(rim) == 8
    assert all(h[RAY_CREASE] for h in rim)
    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig
    assert rim[-1].dest is rim[0].orig  # a closed sink rim really is a cycle


def test_open_sink_inverts_exactly_the_creases_inside_the_rim():
    """The eight radial half-edges inside the rim, and nothing else.

    Seeding the flood fill from the wrong side of the rim inverts the whole rest
    of the sheet instead, and skipping the inversion inverts nothing; both make
    this set comparison fail.
    """
    G, v, start = _sink_fixture()

    open_sink(G, start, 0.5, np.array([-1.0, 0.0]))

    inverted = {h for h in G.halfedges if h[CREASE_ASSIGNMENT] == VALLEY}
    assert inverted == _radial_halves(v)
    assert _both_halves_agree(G)


def test_open_sink_rim_is_uniformly_mountain_when_the_diagonals_are_mountain():
    G, _, start = _sink_fixture(diagonals=MOUNTAIN)

    rim = open_sink(G, start, 0.5, np.array([-1.0, 0.0]))

    assert {h[CREASE_ASSIGNMENT] for h in rim} == {MOUNTAIN}
    assert {h.rev[CREASE_ASSIGNMENT] for h in rim} == {MOUNTAIN}


def test_open_sink_flips_the_rim_to_valley_when_the_diagonals_are_valley():
    """The two-candidate test, with the answer forced by big-little-big.

    At a corner node the crease assignment is ``[c_out, rim, c_in, rim]`` over
    sectors ``135, 45, 45, 135``; the inversion makes ``c_in = -c_out``, and the
    only weakly minimal sectors are the two 45s, both of which big-little-big
    admits only when ``rim == c_out``.  So valley diagonals force a valley rim,
    and a ``MOUNTAIN``-always implementation fails here.
    """
    G, _, start = _sink_fixture(diagonals=VALLEY)

    rim = open_sink(G, start, 0.5, np.array([-1.0, 0.0]))

    assert {h[CREASE_ASSIGNMENT] for h in rim} == {VALLEY}


def test_open_sink_leaves_every_rim_node_locally_flat_foldable():
    G, _, start = _sink_fixture(diagonals=VALLEY)

    rim = open_sink(G, start, 0.5, np.array([-1.0, 0.0]))

    nodes = {v for h in rim for v in (h.orig, h.dest)}
    assert len(nodes) == 8
    for v in nodes:
        valid, _ = local_assignment_valid(v)
        assert valid, f"rim node at {v['pos']} does not fold flat"


def test_open_sink_raises_when_no_uniform_rim_assignment_works():
    """Two diagonals mountain and two valley: the corners disagree on the rim."""
    G, v, start = _sink_fixture()
    _assign_radial(v, DIAGONALS[:2], VALLEY)

    with pytest.raises(InvalidSinkError, match="uniform"):
        open_sink(G, start, 0.5, np.array([-1.0, 0.0]))

    # the rim stays in the graph, so it must be left in the same state the
    # strict=False path leaves it in, not in whichever candidate failed last
    rim = [h for h in G.halfedges if h.get(RAY_CREASE)]
    assert rim and {h[CREASE_ASSIGNMENT] for h in rim} == {MOUNTAIN}


def test_open_sink_warns_instead_of_raising_when_not_strict(caplog):
    G, v, start = _sink_fixture()
    _assign_radial(v, DIAGONALS[:2], VALLEY)

    with caplog.at_level(logging.WARNING, logger="pleat.sink"):
        rim = open_sink(G, start, 0.5, np.array([-1.0, 0.0]), strict=False)

    assert "uniform" in caplog.text
    G.check_consistency()
    assert {h[CREASE_ASSIGNMENT] for h in rim} == {MOUNTAIN}


# ------------------------------------------------ assignments ------------------------------------------------


def test_open_sink_raises_when_the_creases_are_unassigned_and_strict():
    G, v = _diagonal_loop_grid()
    start = _outgoing_towards(v, [0.0, 1.0])

    with pytest.raises(InvalidSinkError, match="assignment"):
        open_sink(G, start, 0.5, np.array([-1.0, 0.0]))


def test_open_sink_raises_when_a_crease_outside_a_rim_node_is_unassigned_and_strict():
    """Every crease *inside* the rim is assigned; one outside, at a rim node, is not.

    The ray passes through an existing interior vertex, so that vertex ends up a
    degree-6 rim node with creases on both sides of the rim -- the only way to
    leave a rim node partially assigned, since a node the rim creates by
    splitting an edge carries a copy of that edge's attributes on both halves.
    Skipping such a node would let ``strict=True`` return a rim whose validity
    was never actually tested.
    """
    G = _grid()
    v, _, start, d = _aimed_near_an_interior_vertex(G)
    _assign_all(G, MOUNTAIN)
    del _outgoing_towards(v, [1.0, 0.0]).attributes[CREASE_ASSIGNMENT]

    with pytest.raises(InvalidSinkError, match="assignment"):
        open_sink(G, start, START_T, d)


def test_open_sink_builds_the_same_geometry_with_and_without_an_assignment():
    """``strict=False`` is a crease-assignment switch; it never changes the geometry."""
    assigned, v, start = _sink_fixture()
    bare, bare_v = _diagonal_loop_grid()
    bare_start = _outgoing_towards(bare_v, [0.0, 1.0])

    rim_a = open_sink(assigned, start, 0.5, np.array([-1.0, 0.0]))
    rim_b = open_sink(bare, bare_start, 0.5, np.array([-1.0, 0.0]), strict=False)

    bare.check_consistency()
    assert len(rim_a) == len(rim_b)
    assert len(assigned.vertices) == len(bare.vertices)
    assert len(assigned.faces) == len(bare.faces)


# ------------------------------------------------ open rim ------------------------------------------------


def test_open_sink_on_a_border_to_border_ray_leaves_an_open_rim():
    G = _grid()
    v = _interior_vertex(G)
    _assign_all(G, MOUNTAIN)
    start = _outgoing_towards(v, [0.0, 1.0])

    rim = open_sink(G, start, 0.5, np.array([1.0, 0.0]))

    G.check_consistency()
    assert rim[-1].dest is not rim[0].orig  # a path, not a cycle
    # Every node on this rim is a symmetric degree-4 crossing, so *both* uniform
    # assignments fold flat and only the order of the two candidates decides the
    # answer.  MOUNTAIN is tried first, deliberately: it is the tie-break.
    assert {h[CREASE_ASSIGNMENT] for h in rim} == {MOUNTAIN}
    assert _both_halves_agree(G)
    nodes = [w for h in rim for w in (h.orig, h.dest) if not w.on_border()]
    for w in nodes:
        valid, _ = local_assignment_valid(w)
        assert valid
    # the tie is real, not an artefact: VALLEY would have done just as well
    for h in rim:
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = VALLEY
    for w in nodes:
        valid, _ = local_assignment_valid(w)
        assert valid, "rim is not symmetric after all; this no longer pins the tie-break"


def test_open_sink_on_an_open_rim_inverts_only_the_smaller_side():
    """The rim cuts the sheet above its middle, so the strip above it is the inside.

    An open rim has no canonical inside, so the smaller of the two sides is
    taken; racing the fills from the wrong seed would invert the larger strip
    below the rim instead.
    """
    G = _grid()
    v = _interior_vertex(G)
    _assign_all(G, MOUNTAIN)
    start = _outgoing_towards(v, [0.0, 1.0])
    cut = 0.5 * float(start.orig["pos"][1] + start.dest["pos"][1])
    ys = [float(w["pos"][1]) for w in G.vertices]
    assert max(ys) - cut < cut - min(ys), "fixture is symmetric; the race has no smaller side"

    rim = open_sink(G, start, 0.5, np.array([1.0, 0.0]))
    rim_edges = {h for h in rim} | {h.rev for h in rim}

    inverted = [h for h in G.halfedges if h[CREASE_ASSIGNMENT] == VALLEY and h not in rim_edges]
    assert inverted, "nothing was inverted"
    for h in inverted:
        assert min(h.orig["pos"][1], h.dest["pos"][1]) >= cut - 1e-9


def test_open_sink_never_inverts_a_paper_border_edge():
    G = _grid()
    _assign_all(G, MOUNTAIN)
    start = _outgoing_towards(_interior_vertex(G), [0.0, 1.0])

    open_sink(G, start, 0.5, np.array([1.0, 0.0]))

    for h in G.halfedges:
        if h.on_border() or h.rev.on_border():
            assert h[CREASE_ASSIGNMENT] == MOUNTAIN


def _square_cycle(G, half_side):
    """The counter-clockwise cycle of existing grid half-edges around a square."""
    s = float(half_side)
    corners = [(-s, -s), (s, -s), (s, s), (-s, s)]
    at = {tuple(np.round(w["pos"], 6)): w for w in G.vertices}
    rim = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        step = np.sign(np.subtract(end, start))
        p = np.array(start, dtype=float)
        while not np.allclose(p, end):
            q = p + step
            a, b = at[tuple(np.round(p, 6))], at[tuple(np.round(q, 6))]
            rim.append(next(h for h in a.outgoing_iter() if h.dest is b))
            p = q
    return rim


def test_interior_of_a_closed_rim_is_exact_even_when_the_sink_swallows_the_sheet():
    """A closed rim is resolved by orientation, never by which side is smaller.

    Here the sink encloses 49 of the 81 faces, so racing the two flood fills --
    which is the documented convention for an *open* rim, whose inside is not
    otherwise defined -- would pick the 32 faces outside it.
    """
    G = from_tiles(platonic(n=4), rings=4)
    G.recompute_lengths_and_angles()
    rim = _square_cycle(G, 3.5)

    interior = _interior_faces(rim, closed=True)

    assert len(interior) == 49
    assert len(interior) > len(G.faces) - len(interior), "fixture no longer has the bigger side inside"
    assert len(_interior_faces(rim, closed=False)) == 32, "the race would take the other side"


# ------------------------------------------------ end to end ------------------------------------------------


def _shrink_rotate_base():
    """A pattern that is genuinely locally flat-foldable to start with.

    Every other fixture here is all-MOUNTAIN, which satisfies Maekawa nowhere,
    so sinking into one can only ever be checked node by node.  This is the
    same graph as ``tests/test_flat_foldable.py::_srg``.
    """
    G = from_tiles(platonic(n=6), rings=2)
    assign_this_way_by_bfs(G, G.central_face())
    return shrink_rotate_pattern(G, simplify_boundary=True, alpha=np.pi / 5, factor=0.5)


def _central_start(G):
    """A deterministic interior half-edge near the middle of the pattern.

    The vertex is picked by proximity to a point that only one vertex is
    anywhere near (the next is ten times further off), so this is not the
    tie-prone argmin that ``central_vertex`` would be; the outgoing half-edge
    is then picked by direction, which is unique.
    """
    v = min(G.vertices, key=lambda w: float(np.linalg.norm(np.asarray(w["pos"]) - [0.5, 0.05])))
    return min(v.outgoing_iter(), key=lambda h: float(np.linalg.norm(np.asarray(h.dest["pos"]) - [0.3, -0.4])))


def _rim_polygon(rim):
    return np.stack([h.orig["pos"] for h in rim]).astype(float)


def _faces_inside(G, polygon):
    """The faces whose centroid lies inside *polygon*, by point-in-polygon.

    An independent ground truth for ``_interior_faces``: a global containment
    test, which is exactly what the flood fill exists to avoid, so the two
    agreeing is evidence and not a tautology.  Every face here is convex, so its
    centroid is inside it and containment of the centroid decides the face.
    """
    inside = set()
    for f in G.faces:
        centroid = np.mean(np.stack([w["pos"] for w in f.vertex_iter()]), axis=0)
        if pointinpolygon(float(centroid[0]), float(centroid[1]), polygon):
            inside.add(f)
    return inside


@pytest.mark.parametrize("angle, ccw", [(225.0, True), (255.0, False)])
def test_interior_of_a_closed_rim_is_the_inside_whichever_way_it_runs(angle, ccw):
    """``_rim_is_ccw`` decides which side of the rim the fill is seeded from.

    Both of the existing structural checks use a counter-clockwise rim, so the
    clockwise arm of that sign was defended by nothing but
    ``is_locally_flat_foldable`` -- and the design records that the test is
    *provably* blind to a global M/V flip, which is precisely what taking the
    wrong side produces.  Mutating ``_rim_is_ccw`` to ``return True`` left the
    whole suite green while inverting 308 of 360 half-edges instead of 4.

    So both orientations are pinned here, against a containment test rather than
    against a validity one.
    """
    G = _shrink_rotate_base()
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    rim, path = add_ray_creases(G, _central_start(G), 0.5, d)

    assert path.closed
    assert _rim_is_ccw(rim) is ccw, "fixture no longer runs this way round; the parametrization is stale"

    interior = _interior_faces(rim, closed=True)

    assert interior == _faces_inside(G, _rim_polygon(rim))
    assert 0 < len(interior) < len(G.faces), "the fill took everything or nothing"


@pytest.mark.parametrize("angle", [225.0, 240.0, 255.0])
def test_open_sink_leaves_a_flat_foldable_pattern_flat_foldable(angle):
    """The end-to-end origami claim: a successful sink yields a valid crease pattern.

    Checking the rim nodes is not enough -- the inversion touches creases at
    vertices the rim never meets, and a wrongly seeded or skipped inversion
    breaks those instead.
    """
    G = _shrink_rotate_base()
    assert is_locally_flat_foldable(G)[0], "fixture is not flat-foldable before the sink"
    d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    rim = open_sink(G, _central_start(G), 0.5, d)

    G.check_consistency()
    assert len(rim) == 4 and rim[-1].dest is rim[0].orig
    ok, violations = is_locally_flat_foldable(G)
    assert ok, {tuple(map(float, v["pos"])): msg for v, msg in violations.items()}


# ------------------------------------------------ bad rays ------------------------------------------------


def test_open_sink_refuses_to_cast_one_way():
    """``both_ways=False`` would leave the rim with a loose end in mid-sheet.

    Such a rim does not separate the paper: the interior fill leaks around the
    dangling end, every face comes out "inside", and the sink inverts the whole
    model -- which no local flat-foldability check can see, because a global
    M/V flip preserves every local condition.  So the argument is refused
    rather than honoured.
    """
    G = _grid()
    _assign_all(G, MOUNTAIN)
    start = _outgoing_towards(_interior_vertex(G), [0.0, 1.0])

    with pytest.raises(TypeError, match="both ways"):
        open_sink(G, start, 0.5, np.array([1.0, 0.0]), both_ways=False)

    assert all(h[CREASE_ASSIGNMENT] == MOUNTAIN for h in G.halfedges), "G was touched before the refusal"


def test_open_sink_rejects_an_untraced_rim_end(monkeypatch):
    """``ends[0] == "start"`` is a failure, not a clean end.

    It is unreachable through ``open_sink`` now that the cast is always
    two-way, so the end check is exercised directly: only ``"closed"`` and
    ``"border"`` mean an end was traced to completion.
    """

    def one_way(*args, **kwargs):
        rim, path = add_ray_creases(*args, **kwargs)
        path.ends = ("start", path.ends[1])
        return rim, path

    monkeypatch.setattr("pleat.sink.add_ray_creases", one_way)
    G = _grid()
    _assign_all(G, MOUNTAIN)
    start = _outgoing_towards(_interior_vertex(G), [0.0, 1.0])

    with pytest.raises(InvalidSinkError, match="terminate"):
        open_sink(G, start, 0.5, np.array([1.0, 0.0]))


def test_open_sink_forwards_angle_tol_to_the_cast():
    """Both of the caster's tolerances have to be reachable from every entry point.

    ``vertex_tol`` was already forwarded; ``angle_tol`` reached no public
    signature at all, so its default was the only value any caller could ever
    get.  Widened absurdly, the fan reads an ordinary arrival as being along a
    crease, which nothing else can provoke.
    """
    G = _grid()
    _assign_all(G, MOUNTAIN)
    v, _, start, d = _aimed_near_an_interior_vertex(G)

    with pytest.raises(DegenerateRayError, match="along a crease"):
        open_sink(G, start, START_T, d, angle_tol=1.0)


def test_open_sink_rejects_a_ray_that_hits_the_step_cap():
    G = _grid()
    _assign_all(G, MOUNTAIN)
    start = _outgoing_towards(_interior_vertex(G), [0.0, 1.0])

    with pytest.raises(InvalidSinkError, match="terminate"):
        open_sink(G, start, 0.5, np.array([np.sqrt(0.5), np.sqrt(0.5)]), max_steps=3)


def test_modules_are_importable_from_the_package():
    # In-process this is vacuous -- importing pleat.sink above already set the
    # attributes -- so ask a fresh interpreter whether ``import pleat`` alone does.
    subprocess.run(
        [sys.executable, "-c", "import pleat; pleat.ray_casting; pleat.sink"],
        check=True,
    )
