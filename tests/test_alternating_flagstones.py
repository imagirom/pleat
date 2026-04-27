"""Tests for alternating flagstone and related Conway operators."""
import numpy as np
import pytest
from eucare.half import IdObject, EuclideanPositionHEG
from eucare.example_tilesets import platonic, t_4_6_12
from eucare.example_graphs import from_tiles
from eucare.conway import (
    alternating_flagstone_graph, flagstone_pvitelli_graph,
    twist_rotate_graph, loft_graph, lace_graph, expand_graph,
    chamfer_graph, dual_graph,
)
from eucare.classifiers import congruency_classifier
from eucare.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY


def _make_tiling(tileset_fn=None, rings=3):
    if tileset_fn is None:
        tileset_fn = lambda: platonic(4)
    tiles = tileset_fn()
    return from_tiles(tiles, rings=rings)


class TestAlternatingFlagstone:
    """Test the alternating flagstone Conway operator."""

    def test_basic_consistency(self):
        G = _make_tiling(rings=3)
        op = alternating_flagstone_graph()
        G2 = op(G, delete_on_border=True)
        G2.check_consistency()
        assert len(G2.faces) > 0
        assert len(G2.vertices) > 0

    @pytest.mark.parametrize("t", [0.2, 1/3, 0.5])
    def test_different_parameters(self, t):
        G = _make_tiling(rings=2)
        op = alternating_flagstone_graph(t=t)
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()

    @pytest.mark.parametrize("tileset_fn", [
        lambda: platonic(3),
        lambda: platonic(4),
        lambda: platonic(6),
    ])
    def test_on_different_tilings(self, tileset_fn):
        G = _make_tiling(tileset_fn, rings=2)
        op = alternating_flagstone_graph()
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()

    def test_increases_face_count(self):
        G = _make_tiling(rings=2)
        n_before = len(G.faces)
        op = alternating_flagstone_graph()
        G2 = op(G, delete_on_border=True)
        assert len(G2.faces) > n_before


class TestFlagstoneVariants:
    """Test other flagstone-related Conway operators."""

    @pytest.mark.parametrize("op_fn,name", [
        (flagstone_pvitelli_graph, "pvitelli"),
        (twist_rotate_graph, "twist_rotate"),
        (loft_graph, "loft"),
        (lace_graph, "lace"),
        (expand_graph, "expand"),
        (chamfer_graph, "chamfer"),
    ])
    def test_operator_consistency(self, op_fn, name):
        G = _make_tiling(rings=2)
        op = op_fn()
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()
        assert len(G2.faces) > 0

    @pytest.mark.parametrize("op_fn", [
        twist_rotate_graph, loft_graph, expand_graph,
    ])
    def test_on_triangular_tiling(self, op_fn):
        G = _make_tiling(lambda: platonic(3), rings=2)
        op = op_fn()
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()


class TestFlagstoneComposition:
    """Test composing flagstone operators with other operators."""

    def test_dual_then_flagstone(self):
        G = _make_tiling(rings=3)
        G = dual_graph()(G, delete_on_border=True)
        G.recompute_lengths_and_angles()
        G = alternating_flagstone_graph()(G, delete_on_border=True)
        G.recompute_lengths_and_angles()
        G.check_consistency()

    def test_flagstone_congruency_classes(self):
        """After alternating flagstone, faces should be classifiable."""
        G = _make_tiling(rings=3)
        G = alternating_flagstone_graph()(G, delete_on_border=True)
        G.recompute_lengths_and_angles()
        classifier = congruency_classifier()
        for f in G.faces:
            key = classifier.classify(f)
            assert key is not None


class TestFlagstoneProperties:
    """Test structural properties of alternating flagstone graphs."""

    def test_positive_areas(self):
        G = _make_tiling(rings=2)
        CP = alternating_flagstone_graph(t=0.5)(G, delete_on_border=True)
        CP.recompute_lengths_and_angles()
        for f in CP.faces:
            assert f.area() > 0

    def test_all_faces_at_least_triangles(self):
        G = _make_tiling(rings=2)
        CP = alternating_flagstone_graph(t=0.5)(G, delete_on_border=True)
        for f in CP.faces:
            assert f.order() >= 3
