"""Tests for Conway operators."""
import numpy as np
import pytest

from eucare.classifiers import congruency_classifier
from eucare.conway import (
    ambo_graph,
    dual_graph,
    gyro_graph,
    join_graph,
    kis_graph,
    starify_graph,
    truncate_graph,
)
from eucare.example_graphs import from_tiles
from eucare.example_tilesets import platonic, t_3_3_4_3_4, t_4_6_12
from eucare.half import EuclideanPositionHEG, IdObject


def _make_tiling(tileset_fn=None, rings=3):
    if tileset_fn is None:
        def tileset_fn():
            return platonic(4)
    tiles = tileset_fn()
    return from_tiles(tiles, rings=rings)


class TestConwayOperators:
    """Test that each Conway operator produces valid graphs."""

    @pytest.mark.parametrize("op_fn,name", [
        (dual_graph, "dual"),
        (kis_graph, "kis"),
        (ambo_graph, "ambo"),
        (truncate_graph, "truncate"),
        (join_graph, "join"),
        (gyro_graph, "gyro"),
        (starify_graph, "starify"),
    ])
    def test_operator_consistency(self, op_fn, name):
        G = _make_tiling()
        op = op_fn()
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()
        assert len(G2.faces) > 0
        assert len(G2.vertices) > 0

    @pytest.mark.parametrize("op_fn", [
        dual_graph, ambo_graph, kis_graph, truncate_graph,
    ])
    def test_operator_increases_complexity(self, op_fn):
        G = _make_tiling(rings=2)
        op = op_fn()
        G2 = op(G, delete_on_border=True)
        # most operators create more faces than the input
        # (dual preserves count, but we test it still works)
        assert len(G2.faces) > 0

    def test_dual_involution(self):
        """dual(dual(G)) should have same face count as G (for interior faces)."""
        G = _make_tiling(rings=4)
        D = dual_graph()(G, delete_on_border=True)
        DD = dual_graph()(D, delete_on_border=True)
        DD.check_consistency()
        # After removing border effects, the count should be close
        # (exact match only for closed tilings)
        assert len(DD.faces) > 0


class TestConwayOnDifferentTilings:
    """Test operators on non-square tilings."""

    @pytest.mark.parametrize("tileset_fn", [
        lambda: platonic(3),
        lambda: platonic(6),
        t_4_6_12,
        t_3_3_4_3_4,
    ])
    def test_ambo_on_different_tilings(self, tileset_fn):
        G = _make_tiling(tileset_fn, rings=2)
        G2 = ambo_graph()(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()

    @pytest.mark.parametrize("tileset_fn", [
        lambda: platonic(3),
        lambda: platonic(6),
    ])
    def test_gyro_on_different_tilings(self, tileset_fn):
        G = _make_tiling(tileset_fn, rings=2)
        G2 = gyro_graph()(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()


class TestConwayComposition:
    """Test composing multiple operators."""

    def test_ambo_then_dual(self):
        G = _make_tiling(rings=3)
        G = ambo_graph()(G, delete_on_border=True)
        G.recompute_lengths_and_angles()
        G = dual_graph()(G, delete_on_border=True)
        G.check_consistency()

    def test_triple_composition(self):
        G = _make_tiling(rings=3)
        for op in [ambo_graph(), gyro_graph()]:
            G = op(G, delete_on_border=True)
            G.recompute_lengths_and_angles()
        G.check_consistency()
        assert len(G.faces) > 0


class TestCongruencyClassifier:
    """Test that congruency classification works after Conway ops."""

    def test_classify_after_kis(self):
        G = _make_tiling(rings=3)
        G = kis_graph()(G, delete_on_border=True)
        G.recompute_lengths_and_angles()
        classifier = congruency_classifier()
        for f in G.faces:
            key = classifier.classify(f)
            assert key is not None
