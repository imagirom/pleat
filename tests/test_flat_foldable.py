"""Tests for local flat-foldability checks (Kawasaki, Maekawa, crimp recursion)."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import platonic
from pleat.flat_foldable import (
    _crimp,
    _crimp_ok,
    folded_crease_angles,
    is_locally_flat_foldable,
    kawasaki_sum,
    local_assignment_valid,
    maekawa_check,
)
from pleat.half import EuclideanPositionHEG
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from pleat.shrink_rotate import assign_this_way_by_bfs, shrink_rotate_pattern


def _srg():
    G = from_tiles(platonic(n=6), rings=2)
    assign_this_way_by_bfs(G, G.central_face())
    return shrink_rotate_pattern(G, simplify_boundary=True, alpha=np.pi / 5, factor=0.5)


def test_shrink_rotate_pattern_is_locally_flat_foldable():
    ok, violations = is_locally_flat_foldable(_srg())
    assert ok, violations


def test_plain_tiling_is_not_flat_foldable():
    G = from_tiles(platonic(n=6), rings=2)
    G.recompute_lengths_and_angles()
    ok, violations = is_locally_flat_foldable(G)
    assert not ok
    # interior vertices of the hexagonal tiling have odd degree 3
    assert any("odd degree" in msg for msg in violations.values())


def test_maekawa_check():
    G = EuclideanPositionHEG(other=rosette(n=4))
    center = next(v for v in G.vertices if not v.on_border())
    creases = list(center.outgoing_iter())
    assert len(creases) == 4

    for h in creases:
        h[CREASE_ASSIGNMENT] = MOUNTAIN
    creases[0][CREASE_ASSIGNMENT] = VALLEY
    assert maekawa_check(center)  # 3 mountains, 1 valley

    creases[1][CREASE_ASSIGNMENT] = VALLEY
    assert not maekawa_check(center)  # 2 mountains, 2 valleys


def _rosette_vertex(n):
    G = EuclideanPositionHEG(other=rosette(n=n))
    G.recompute_lengths_and_angles()
    return G, next(v for v in G.vertices if not v.on_border() and v.order() == n)


def _assign(v, values):
    for h, value in zip(v.outgoing_iter(), values):
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = value


def test_folded_crease_angles_final_entry_is_the_kawasaki_sum():
    G, v = _rosette_vertex(4)
    psi = folded_crease_angles(v)
    assert len(psi) == v.order()
    # psi is the running alternating sum, so its last entry is exactly the
    # Kawasaki sum -- Kawasaki is "the cycle closes", not a separate condition
    assert psi[-1] == pytest.approx(kawasaki_sum(v), abs=1e-9)


def test_symmetric_degree_4_vertex_accepts_every_3_to_1_assignment():
    G, v = _rosette_vertex(4)
    assert v.order() == 4
    for odd in range(4):
        _assign(v, [VALLEY if i == odd else MOUNTAIN for i in range(4)])
        valid, margin = local_assignment_valid(v)
        assert valid, f"odd crease {odd} should be valid"
        assert margin >= 0.0


def test_symmetric_degree_4_vertex_rejects_a_2_to_2_assignment():
    G, v = _rosette_vertex(4)
    _assign(v, [VALLEY, VALLEY, MOUNTAIN, MOUNTAIN])
    valid, _ = local_assignment_valid(v)
    assert not valid


def test_symmetric_vertex_margin_is_the_gap_between_the_two_folded_clusters():
    G, v = _rosette_vertex(4)
    _assign(v, [VALLEY, MOUNTAIN, MOUNTAIN, MOUNTAIN])
    _, margin = local_assignment_valid(v)
    # all four sectors are pi/2, so psi = [pi/2, 0, pi/2, 0]: the four folded
    # creases collapse into exactly two clusters, pi/2 apart
    assert folded_crease_angles(v) == pytest.approx([np.pi / 2, 0.0, np.pi / 2, 0.0], abs=1e-12)
    assert margin == pytest.approx(np.pi / 2, abs=1e-12)


def test_only_a_weakly_minimal_sector_may_be_crimped():
    """Crimping a non-minimal sector accepts assignments that do not fold.

    Every vertex reachable from ``rosette`` has equal sectors, so "crimp the
    *smallest*" is never exercised end-to-end.  Pin it on the helper directly.
    These angles satisfy Kawasaki (2 - 2 + 3 - 3 + 1 - 1 == 0).  The only
    minimal sectors are the two of size 1, and both are blocked by
    big-little-big, so the assignment is invalid -- but dropping the minimality
    filter finds a crimp among the larger sectors and wrongly accepts it.
    """
    angles = [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    assert not _crimp_ok(angles, [-1, 1, 1, -1, -1, -1], 1e-8)
    # a control on the same sectors, so the test cannot pass by rejecting everything
    assert _crimp_ok(angles, [-1, -1, -1, -1, 1, 1], 1e-8)


def test_odd_degree_vertex_is_rejected_before_the_crimp_recursion():
    """``local_assignment_valid`` is called directly by callers that never
    screened for parity, and the crimp recursion cannot bottom out on an odd
    number of creases.  The ``(False, 0.0)`` is the gate's own answer: without
    it this vertex reaches the Kawasaki check and reports a nonzero margin.
    """
    G, v = _rosette_vertex(5)
    assert v.order() == 5
    _assign(v, [MOUNTAIN] * 5)
    assert local_assignment_valid(v) == (False, 0.0)


def _skewed_rosette_vertex():
    """A degree-4 vertex with one rim corner pushed off the symmetric position.

    Kawasaki is broken, and the four folded crease positions fall into three
    distinct clusters rather than the two a symmetric rosette gives.
    """
    G, v = _rosette_vertex(4)
    rim = next(iter(v.outgoing_iter())).dest
    rim["pos"] = rim["pos"] + np.array([0.4, 0.3])
    G.recompute_lengths_and_angles()
    _assign(v, [VALLEY, MOUNTAIN, MOUNTAIN, MOUNTAIN])
    return G, v


def test_kawasaki_violating_vertex_is_rejected():
    """The Kawasaki gate is load-bearing for direct callers: the crimp recursion
    alone accepts this assignment, because a crimp only preserves the alternating
    sum -- it never checks that the sum was zero to begin with.
    """
    G, v = _skewed_rosette_vertex()
    assert abs(kawasaki_sum(v)) > 0.1
    valid, margin = local_assignment_valid(v)
    assert not valid
    # three clusters, so margin is the *smallest* of two different gaps
    psi = np.sort(folded_crease_angles(v))
    assert np.diff(psi) == pytest.approx([0.0, 1.1377723167, 0.4330240101], abs=1e-9)
    assert margin == pytest.approx(0.4330240101, abs=1e-9)


def test_is_locally_flat_foldable_threads_tol_into_the_assignment_check():
    """A tol loose enough to forgive this vertex's Kawasaki error must forgive it
    in the assignment check too, or the graph-level verdict contradicts its own
    tolerance.
    """
    G, v = _skewed_rosette_vertex()
    assert is_locally_flat_foldable(G, tol=1e-8)[1], "sanity: strict tol still rejects"
    ok, violations = is_locally_flat_foldable(G, tol=1.0)
    assert ok, violations


def test_tied_sectors_skip_crimps_blocked_by_big_little_big():
    """A valid assignment whose only workable crimps are late in index order.

    All six sectors of ``rosette(6)`` are equal, so every one of them is weakly
    minimal.  For ``MMMMVV`` the crimps at sectors 0, 1 and 2 are all blocked by
    big-little-big (equal bounding creases); only sector 3 onwards works.  An
    implementation that commits to the *first* weakly-minimal sector -- or that
    demands a *strict* minimum, which does not exist here -- rejects this vertex.

    This does not pin the *retry after a failed recursive call*: committing to
    the first big-little-big-admissible candidate passes every test here, and
    exhaustive enumeration over degree-6 and degree-8 vertices found no
    counterexample.  The retry is kept as cheap insurance, not as tested behaviour.
    """
    G, v = _rosette_vertex(6)
    assert v.order() == 6
    _assign(v, [MOUNTAIN, MOUNTAIN, MOUNTAIN, MOUNTAIN, VALLEY, VALLEY])
    valid, _ = local_assignment_valid(v)
    assert valid


def test_all_mountain_vertex_is_rejected():
    """No crimp is ever available when every crease has the same assignment."""
    G, v = _rosette_vertex(4)
    _assign(v, [MOUNTAIN, MOUNTAIN, MOUNTAIN, MOUNTAIN])
    valid, _ = local_assignment_valid(v)
    assert not valid


def test_crimp_merges_three_sectors_alternately_and_keeps_alignment():
    """``_crimp`` drops the two creases bounding sector *i* and merges ``a-b+c``.

    No real crease pattern here has an interior vertex of degree > 4, so the
    merge formula is only reachable end-to-end on symmetric vertices where a
    wrong sign happens not to matter.  Pin it directly instead.
    """
    angles = [10.0, 3.0, 20.0, 5.0, 8.0, 4.0]
    mv = [1, 1, -1, 1, -1, 1]

    new_angles, new_mv = _crimp(angles, mv, 2)
    # sectors 1, 2, 3 merge into 3 - 20 + 5; creases 2 and 3 disappear
    assert new_angles == pytest.approx([3.0 - 20.0 + 5.0, 8.0, 4.0, 10.0])
    assert new_mv == [mv[1], mv[4], mv[5], mv[0]]

    # the same, wrapping around the end of the list
    new_angles, new_mv = _crimp(angles, mv, 0)
    assert new_angles == pytest.approx([4.0 - 10.0 + 3.0, 20.0, 5.0, 8.0])
    assert new_mv == [mv[5], mv[2], mv[3], mv[4]]
