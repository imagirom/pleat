"""Tests for local flat-foldability checks (Kawasaki, Maekawa, crimp recursion)."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import platonic
from pleat.flat_foldable import (
    _crimp,
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


def test_symmetric_vertex_reports_a_small_margin():
    G, v = _rosette_vertex(4)
    _assign(v, [VALLEY, MOUNTAIN, MOUNTAIN, MOUNTAIN])
    _, margin = local_assignment_valid(v)
    # all four sectors are equal, so the folded creases collapse into two
    # clusters -- the vertex is symmetric, and the margin says so
    assert margin < np.pi


def test_tied_sectors_require_backtracking_over_weakly_minimal_choices():
    """A valid assignment whose only workable crimps are late in index order.

    All six sectors of ``rosette(6)`` are equal, so every one of them is weakly
    minimal.  For ``MMMMVV`` the crimps at sectors 0, 1 and 2 are all blocked by
    big-little-big (equal bounding creases); only sector 3 onwards works.  An
    implementation that commits to one weakly-minimal sector -- or that demands
    a *strict* minimum, which does not exist here -- rejects this vertex.
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
