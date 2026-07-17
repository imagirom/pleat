"""Tests for local flat-foldability checks (Kawasaki, Maekawa)."""

from __future__ import annotations

import numpy as np

from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import platonic
from pleat.flat_foldable import is_locally_flat_foldable, maekawa_check
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
