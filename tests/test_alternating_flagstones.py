"""Tests for alternating flagstone and related Conway operators."""

import numpy as np
import pytest

from pleat.alternating_flagstones import (
    build_structure,
    connection_length_metric,
    cut_twist_centres,
    extend_border,
    optimize_alternating_flagstone,
    subdivide_ridges_for_curved_fold,
)
from pleat.classifiers import congruency_classifier

try:  # optional dep
    import torch  # noqa: F401

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False
from pleat.conway import (
    alternating_flagstone_graph,
    chamfer_graph,
    dual_graph,
    expand_graph,
    flagstone_pvitelli_graph,
    lace_graph,
    loft_graph,
    shrink_rotate_graph,
)
from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic, t_4_6_12
from pleat.half import EuclideanPositionHEG, IdObject
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY


def _make_tiling(tileset_fn=None, rings=3):
    if tileset_fn is None:

        def tileset_fn():
            return platonic(4)

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

    @pytest.mark.parametrize("t", [0.2, 1 / 3, 0.5])
    def test_different_parameters(self, t):
        G = _make_tiling(rings=2)
        op = alternating_flagstone_graph(t=t)
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()

    @pytest.mark.parametrize(
        "tileset_fn",
        [
            lambda: platonic(3),
            lambda: platonic(4),
            lambda: platonic(6),
        ],
    )
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

    @pytest.mark.parametrize(
        "op_fn,name",
        [
            (flagstone_pvitelli_graph, "pvitelli"),
            (shrink_rotate_graph, "shrink_rotate"),
            (loft_graph, "loft"),
            (lace_graph, "lace"),
            (expand_graph, "expand"),
            (chamfer_graph, "chamfer"),
        ],
    )
    def test_operator_consistency(self, op_fn, name):
        G = _make_tiling(rings=2)
        op = op_fn()
        G2 = op(G, delete_on_border=True)
        G2.recompute_lengths_and_angles()
        G2.check_consistency()
        assert len(G2.faces) > 0

    @pytest.mark.parametrize(
        "op_fn",
        [
            shrink_rotate_graph,
            loft_graph,
            expand_graph,
        ],
    )
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


# ---------------------------------------------------------------------------
# High-level pipeline (pleat.alternating_flagstones)


class TestAlternatingFlagstonePipeline:
    """Tests for the high-level alternating-flagstone pipeline."""

    def test_build_structure_lookups(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        # one flagstone per face of original
        assert len(s.flagstone_faces) == len(s.original_faces) == len(list(s.original.faces))
        # one star per vertex of original
        assert len(s.star_vertices) == len(list(s.original.vertices))
        # every (corner, flagstone) <-> (orig_vertex, orig_face) is a bijection
        assert len(s.corner_to_original) == len(s.original_to_corner)
        # each flagstone has as many corners as the original face
        for f0, f in s.f_to_flagstone.items():
            corners_in_f = [k for k in s.corner_to_original if k[1] is f]
            assert len(corners_in_f) == f0.order()
        # sanity: no duplicate corner vertex (every flagstone-corner in CP
        # belongs to exactly one (corner, flagstone) tuple)
        unique_corners = {c for c, _ in s.corner_to_original}
        assert len(unique_corners) == len(s.flagstone_corners)

    def test_initial_crease_assignments(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        # The "ridge" crease (h.rev.nex for h in flagstone perimeter) is MOUNTAIN.
        # Flagstone perimeters and the radial creases that touch a star are VALLEY.
        n_mountain = 0
        for f in s.flagstone_faces:
            for h in f.halfedge_iter():
                # flagstone perimeter is valley (or border)
                assert h[CREASE_ASSIGNMENT] in (VALLEY, MOUNTAIN)  # nothing else
                ridge = h.rev.nex
                if not ridge.rev.on_border():
                    assert ridge[CREASE_ASSIGNMENT] == MOUNTAIN
                    n_mountain += 1
        assert n_mountain > 0

    def test_metric_returns_finite_values(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        m = connection_length_metric(s)
        assert np.isfinite(m["max_relative_length_error"])
        assert np.isfinite(m["max_angle_error_deg"])
        assert isinstance(m["summary"], str)

    def test_extend_border_increases_edges(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        ext = extend_border(s.CP)
        ext.check_consistency()
        assert len(ext.faces) >= len(s.CP.faces)
        assert len(ext.halfedges) > len(s.CP.halfedges)

    def test_cut_twist_centres_removes_star_neighbourhood(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        n_stars_before = sum(
            1 for v in s.CP.vertices if v.attributes.get("pre_conway") is not None and not v.on_border()
        )
        assert n_stars_before > 0
        cut = cut_twist_centres(s)
        cut.check_consistency()
        # interior star vertices have been cut away
        n_stars_after = sum(1 for v in cut.vertices if v.attributes.get("pre_conway") is not None and not v.on_border())
        assert n_stars_after < n_stars_before

    def test_subdivide_ridges_increases_edge_count(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        n_before = len(s.CP.halfedges)
        sub = subdivide_ridges_for_curved_fold(s.CP, n_subdivisions=4)
        sub.check_consistency()
        assert len(sub.halfedges) > n_before

    def test_subdivide_ridges_validates_n(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        with pytest.raises(ValueError):
            subdivide_ridges_for_curved_fold(s.CP, n_subdivisions=1)
        with pytest.raises(ValueError):
            subdivide_ridges_for_curved_fold(s.CP, n_subdivisions=2.5)

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_TORCH, reason="torch is an optional dependency")
    def test_optimize_smoke(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        m_before = connection_length_metric(s)
        curve = optimize_alternating_flagstone(s, n_steps=400)
        # mostly-monotonically decreasing
        assert curve[-1] < curve[0]
        m_after = connection_length_metric(s)
        assert m_after["max_relative_length_error"] <= m_before["max_relative_length_error"]

    @pytest.mark.skipif(HAS_TORCH, reason="exercises the no-torch error path")
    def test_optimize_raises_without_torch(self):
        G = _make_tiling(rings=2)
        s = build_structure(G, t=0.5)
        with pytest.raises(ModuleNotFoundError, match="PyTorch"):
            optimize_alternating_flagstone(s, n_steps=1)
