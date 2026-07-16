"""Tests for the shrink-rotate origami pipeline.

This tests the full workflow:
  graph -> reciprocal figure -> shrink-rotate -> crease assignment -> folding -> overlap
"""

import numpy as np
import pytest

from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic, t_4_6_12
from pleat.half import EuclideanPositionHEG, IdObject
from pleat.overlap import (
    CREASE_ASSIGNMENT,
    MOUNTAIN,
    VALLEY,
    color_creases,
    fold_complete,
    fold_wireframe,
    overlap_graph,
)
from pleat.shrink_rotate import (
    assign_shrink_rotate_creases,
    assign_this_way_by_face_z_order,
    reciprocal_figure,
    shrink_rotate_pattern,
)
from pleat.flat_foldable import kawasaki_sum, max_kawasaki_sum
from pleat.search_trees import face_bfs_tree


def _make_graph(tileset_fn=None, rings=3):
    if tileset_fn is None:

        def tileset_fn():
            return platonic(4)

    tiles = tileset_fn()
    return from_tiles(tiles, rings=rings)


def _assign_face_z_order(G):
    """Assign z_order to faces and THIS_WAY to halfedges via BFS from central face."""
    fs = list(G.faces)
    central = min(fs, key=lambda f: np.linalg.norm(f.midpoint()))
    central["z_order"] = 0
    for orig, dest in face_bfs_tree(central):
        dest["z_order"] = orig["z_order"] + 1
    assign_this_way_by_face_z_order(G)


class TestReciprocalFigure:
    """Test reciprocal figure computation."""

    def test_basic_reciprocal(self):
        G = _make_graph(rings=3)
        D = reciprocal_figure(G)
        D.check_consistency()
        assert len(D.vertices) > 0
        assert len(D.faces) > 0

    def test_reciprocal_stores_pos(self):
        G = _make_graph(rings=3)
        _ = reciprocal_figure(G)
        for f in G.faces:
            assert "reciprocal_pos" in f
            assert len(f["reciprocal_pos"]) == 2

    @pytest.mark.parametrize(
        "tileset_fn",
        [
            lambda: platonic(3),
            lambda: platonic(4),
            lambda: platonic(6),
        ],
    )
    def test_reciprocal_different_tilings(self, tileset_fn):
        G = _make_graph(tileset_fn, rings=3)
        D = reciprocal_figure(G)
        D.check_consistency()


class TestShrinkRotate:
    """Test shrink-rotate graph construction."""

    def test_basic_srg(self):
        G = _make_graph(rings=3)
        SRG = shrink_rotate_pattern(G)
        SRG.check_consistency()
        assert len(SRG.faces) > len(G.faces)  # SRG has more faces (twist-rotate subdivides)

    def test_srg_parameters(self):
        G = _make_graph(rings=3)
        SRG1 = shrink_rotate_pattern(G, alpha=np.pi / 6, factor=0.3)
        SRG1.check_consistency()
        G = _make_graph(rings=3)
        SRG2 = shrink_rotate_pattern(G, alpha=np.pi / 4, factor=0.7)
        SRG2.check_consistency()

    def test_kawasaki_sum_near_zero(self):
        """For a valid flat-foldable CP, Kawasaki sums should be near zero."""
        G = _make_graph(rings=3)
        SRG = shrink_rotate_pattern(G)
        mks = max_kawasaki_sum(SRG)
        assert mks < 1e-6, f"Max Kawasaki sum too large: {mks}"


class TestCreaseAssignment:
    """Test crease assignment on shrink-rotate graphs."""

    def test_assign_creases(self):
        G = _make_graph(rings=3)
        _assign_face_z_order(G)
        SRG = shrink_rotate_pattern(G)
        # Check that some edges have crease assignments
        assigned = [
            e for e in SRG.halfedges if CREASE_ASSIGNMENT in e.attributes and not (e.on_border() or e.rev.on_border())
        ]
        assert len(assigned) > 0

    def test_crease_consistency(self):
        """Crease assignment should be consistent between e and e.rev."""
        G = _make_graph(rings=3)
        _assign_face_z_order(G)
        SRG = shrink_rotate_pattern(G)
        for e in SRG.halfedges:
            if CREASE_ASSIGNMENT in e.attributes and CREASE_ASSIGNMENT in e.rev.attributes:
                assert e[CREASE_ASSIGNMENT] == e.rev[CREASE_ASSIGNMENT]


class TestFolding:
    """Test the folding pipeline."""

    def test_fold_wireframe(self):
        G = _make_graph(rings=3)
        _assign_face_z_order(G)
        SRG = shrink_rotate_pattern(G)
        fold_wireframe(SRG)
        SRG.check_consistency()

    def test_two_coloring(self):
        G = _make_graph(rings=3)
        G.twocolor_faces()
        colors = {f["color_key"] for f in G.faces}
        assert len(colors) == 2


class TestOverlapGraph:
    """Test overlap graph computation (integration test)."""

    @pytest.mark.slow
    def test_overlap_graph_basic(self):
        G = _make_graph(lambda: platonic(4), rings=2)
        _assign_face_z_order(G)
        SRG = shrink_rotate_pattern(G)
        fold_wireframe(SRG)
        OG = overlap_graph(SRG, eps=1e-8)
        OG.check_consistency()
        assert len(OG.faces) > 0
        # Each face in overlap graph should have 'original_faces' attribute
        for f in OG.faces:
            assert "original_faces" in f


class TestFoldComplete:
    """Test the full fold_complete pipeline (integration test, requires solver)."""

    @pytest.mark.slow
    def test_fold_complete_small(self):
        """Full pipeline on a small tiling."""
        G = _make_graph(lambda: platonic(4), rings=2)
        _assign_face_z_order(G)
        SRG = shrink_rotate_pattern(G)
        results = fold_complete(SRG, overlap_eps=1e-8)
        assert "CP" in results
        assert "folded_state" in results
        assert "folded_view_top" in results
        assert "folded_view_bottom" in results
        results["folded_state"].check_consistency()

    @pytest.mark.slow
    def test_fold_complete_triangular(self):
        G = _make_graph(lambda: platonic(3), rings=2)
        _assign_face_z_order(G)
        SRG = shrink_rotate_pattern(G)
        results = fold_complete(SRG, overlap_eps=1e-8)
        assert "CP" in results
        results["folded_state"].check_consistency()
