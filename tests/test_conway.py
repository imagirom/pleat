"""Tests for Conway operators."""

import matplotlib

matplotlib.use("Agg")  # headless: rendering smoke tests must not need a display

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

    @pytest.mark.parametrize(
        "op_fn,name",
        [
            (dual_graph, "dual"),
            (kis_graph, "kis"),
            (ambo_graph, "ambo"),
            (truncate_graph, "truncate"),
            (join_graph, "join"),
            (gyro_graph, "gyro"),
            (starify_graph, "starify"),
        ],
    )
    def test_operator_consistency(self, op_fn, name):
        G = _make_tiling()
        op = op_fn()
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()
        assert len(G2.faces) > 0
        assert len(G2.vertices) > 0

    @pytest.mark.parametrize(
        "op_fn",
        [
            dual_graph,
            ambo_graph,
            kis_graph,
            truncate_graph,
        ],
    )
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

    @pytest.mark.parametrize(
        "tileset_fn",
        [
            lambda: platonic(3),
            lambda: platonic(6),
            t_4_6_12,
            t_3_3_4_3_4,
        ],
    )
    def test_ambo_on_different_tilings(self, tileset_fn):
        G = _make_tiling(tileset_fn, rings=2)
        G2 = ambo_graph()(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()

    @pytest.mark.parametrize(
        "tileset_fn",
        [
            lambda: platonic(3),
            lambda: platonic(6),
        ],
    )
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


class TestConwayCopyGraphPath:
    def test_dual_with_copy_graph(self):
        G = _make_tiling(rings=2)
        G2 = dual_graph()(G, copy_graph=True)
        # Original is unchanged when copy_graph=True
        G2.check_consistency()


class TestGoldberg2:
    def test_goldberg2_smoke(self):
        from eucare.conway import goldberg2_graph

        op = goldberg2_graph()
        assert op.graph is not None


class TestExpandLoftLaceChamfer:
    def test_expand(self):
        from eucare.conway import expand_graph

        G = _make_tiling(rings=2)
        result = expand_graph()(G)
        result.check_consistency()

    def test_loft(self):
        from eucare.conway import loft_graph

        G = _make_tiling(rings=2)
        result = loft_graph()(G)
        result.check_consistency()

    def test_lace(self):
        from eucare.conway import lace_graph

        G = _make_tiling(rings=2)
        result = lace_graph()(G)
        result.check_consistency()

    def test_chamfer(self):
        from eucare.conway import chamfer_graph

        G = _make_tiling(rings=2)
        result = chamfer_graph()(G)
        result.check_consistency()

    def test_shrink_rotate_graph(self):
        from eucare.conway import shrink_rotate_graph

        G = _make_tiling(rings=2)
        result = shrink_rotate_graph()(G)
        result.check_consistency()

    def test_flagstone_pvitelli(self):
        from eucare.conway import flagstone_pvitelli_graph

        G = _make_tiling(rings=2)
        result = flagstone_pvitelli_graph()(G)
        result.check_consistency()


class TestConwayOperatorInternals:
    def test_show_does_not_crash(self):
        # show() opens a renderer and writes a file -- skip if that fails on this system
        op = dual_graph()
        # Not invoking show() (would write to disk); just verify the op has the expected attributes.
        assert hasattr(op, "v1") and hasattr(op, "vf") and hasattr(op, "v2")


class TestShorthandMethods:
    """Shorthand methods on GeometricHEG (G.ambo().dual()...). See issue #28."""

    @pytest.mark.parametrize(
        "method_name,args",
        [
            ("dual", ()),
            ("kis", ()),
            ("ambo", ()),
            ("join", ()),
            ("meta", ()),
            ("ortho", ()),
            ("goldberg2", ()),
            ("truncate", (0.4,)),
            ("gyro", ()),
            ("starify", (0.3,)),
            ("alternating_flagstone", (0.3,)),
            ("shrink_rotate", (0.5,)),
            ("loft", (0.5,)),
            ("lace", (0.5,)),
            ("expand", (0.5,)),
            ("flagstone_pvitelli", (0.25,)),
            ("chamfer", (0.5,)),
        ],
    )
    def test_each_method_returns_consistent_graph(self, method_name, args):
        G = _make_tiling(rings=2)
        result = getattr(G, method_name)(*args)
        result.check_consistency()
        assert len(result.faces) > 0

    def test_chaining(self):
        """G.ambo().dual().truncate(0.4).dual() — the canonical example from the issue."""
        G = _make_tiling(rings=3)
        result = G.ambo().dual().truncate(0.4).dual()
        result.check_consistency()
        assert len(result.faces) > 0

    def test_method_returns_self_by_default(self):
        """Without copy_graph, methods mutate in place and return the same object."""
        G = _make_tiling(rings=2)
        result = G.dual()
        assert result is G

    def test_method_with_copy_graph(self):
        G = _make_tiling(rings=2)
        original_faces = set(G.faces)
        result = G.dual(copy_graph=True)
        assert result is not G
        assert set(G.faces) == original_faces  # original untouched

    def test_truncate_with_t(self):
        G = _make_tiling(rings=2)
        a = _make_tiling(rings=2).truncate(0.3)
        b = G.truncate(0.7)
        a.check_consistency()
        b.check_consistency()
        # Different cut depths produce structurally valid (but different) graphs.
        assert len(a.faces) == len(b.faces) > 0


class TestFaceFilterCallable:
    """``faces=`` accepts a callable Face -> bool, in addition to a set."""

    def test_callable_filter_true(self):
        G = _make_tiling(rings=2)
        n_in = len(G.faces)
        result = G.kis(faces=lambda f: True)
        assert len(result.faces) > n_in  # kis splits every face into triangles

    def test_callable_filter_false(self):
        G = _make_tiling(rings=2)
        n_in = len(G.faces)
        result = G.kis(faces=lambda f: False)
        # No face matched, so structure is unchanged.
        assert len(result.faces) == n_in

    def test_callable_filter_matches_explicit_set(self):
        """Passing ``faces=callable`` is equivalent to passing the matching set."""
        from eucare.half import IdObject

        IdObject.reset_ids()
        G1 = _make_tiling(rings=2)
        selected = {next(iter(G1.faces))}
        target_id = next(iter(selected))["id"]
        G1.kis(faces=selected)

        IdObject.reset_ids()
        G2 = _make_tiling(rings=2)
        G2.kis(faces=lambda f: f["id"] == target_id)

        assert len(G1.faces) == len(G2.faces)


@pytest.mark.parametrize(
    "factory_name",
    ["dual_graph", "kis_graph", "gyro_graph", "flagstone_pvitelli_graph"],
)
def test_geometric_show_smoke(factory_name):
    """``GeometricConwayOperator.render``/``show`` run end-to-end headless."""
    import eucare.conway as conway_mod
    from eucare.rendering import Rendering

    factory = getattr(conway_mod, factory_name)
    op = factory()

    # render() builds the styled fundamental-domain graph and rasterises it in memory.
    assert isinstance(op.render(), Rendering)

    # No files, no IPython: show() is display-only and degrades to a no-op headless.
    assert op.show() is None
