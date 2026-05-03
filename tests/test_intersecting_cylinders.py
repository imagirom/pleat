"""Tests for the intersecting-cylinders subpackage."""

import numpy as np
import pytest

from eucare.example_graphs import from_tiles
from eucare.example_tilesets import platonic
from eucare.half import IdObject
from eucare.intersecting_cylinders import (
    Profile,
    circular_profile,
    convert_all_to_triangle_twists,
    convert_to_triangle_twist,
    make_intersecting_cylinders,
    show_3d,
    to_3d_mesh,
    top_view,
)


def _make_graph(p=4, rings=2):
    IdObject.reset_ids()
    return from_tiles(platonic(p), rings=rings)


class TestProfile:
    def test_circular_profile_shape(self):
        p = circular_profile()
        assert p.t.ndim == 1 and p.l.ndim == 1
        assert p.t.shape == p.l.shape
        assert p.t.shape[0] >= 2

    def test_circular_profile_normalized(self):
        p = circular_profile()
        assert p.t[0] == pytest.approx(0.0)
        assert p.l[0] == pytest.approx(0.0)
        # Arc length normalised to 1; t is scaled by shrink_factor < 1.
        assert p.l[-1] == pytest.approx(1.0)
        assert p.t[-1] == pytest.approx(p.shrink_factor)
        assert 0.0 < p.shrink_factor <= 1.0

    def test_arc_length_monotonic(self):
        p = circular_profile()
        assert np.all(np.diff(p.l) >= 0)

    def test_y_height_array(self):
        # y array must have same length as t and l, start at 0, and be monotonic
        # for the circular profile.
        p = circular_profile(scale=1.0)
        assert p.y.shape == p.t.shape == p.l.shape
        assert p.y[0] == pytest.approx(0.0)
        # circular_profile(scale=1.0) ranges in y_axis ∈ [0, 1] (i.e. y/sf ∈ [0, 1]).
        assert (p.y[-1] / p.shrink_factor) == pytest.approx(1.0, abs=1e-3)
        assert np.all(np.diff(p.y) >= -1e-12)

    def test_custom_profile(self):
        # Triangular bump: linear up, linear down.
        def fn(x):
            return np.where(x < 0.5, x, 1.0 - x)

        p = Profile.from_function(fn, n_samples=200)
        assert p.t[0] == 0.0
        assert p.shrink_factor > 0.0
        assert p.shrink_factor < 1.0  # arc length > 1 due to bump


class TestMakeIntersectingCylinders:
    @pytest.mark.parametrize("p", [3, 4, 6])
    def test_runs_for_platonic(self, p):
        G = _make_graph(p=p, rings=2)
        profile = circular_profile()
        CP = make_intersecting_cylinders(G, profile, r=1.0)
        CP.check_consistency()
        assert len(CP.vertices) > 0
        assert len(CP.halfedges) > 0
        assert len(CP.faces) > 0

    def test_color_keys_present(self):
        G = _make_graph(p=4, rings=2)
        CP = make_intersecting_cylinders(G, circular_profile(), r=1.0)
        for h in CP.halfedges:
            assert "color_key" in h
        # Reverses share color.
        for h in CP.halfedges:
            assert h["color_key"] == h.rev["color_key"]

    def test_curve_pos_on_red_edges(self):
        G = _make_graph(p=4, rings=2)
        CP = make_intersecting_cylinders(G, circular_profile(), r=1.0)
        red = [h for h in CP.halfedges if h["color_key"] == (1.0, 0.0, 0.0)]
        assert len(red) > 0
        for h in red:
            assert "curve_pos" in h
            curve = h["curve_pos"]
            assert curve.ndim == 2 and curve.shape[1] == 2
            # Reverse halfedge stores reversed polyline.
            np.testing.assert_allclose(h["curve_pos"], h.rev["curve_pos"][::-1])

    def test_r_less_than_one(self):
        G = _make_graph(p=4, rings=2)
        CP = make_intersecting_cylinders(G, circular_profile(), r=0.7)
        CP.check_consistency()
        assert len(CP.halfedges) > 0

    def test_invalid_r(self):
        G = _make_graph(p=4, rings=1)
        with pytest.raises(ValueError):
            make_intersecting_cylinders(G, circular_profile(), r=0.0)
        with pytest.raises(ValueError):
            make_intersecting_cylinders(G, circular_profile(), r=1.5)

    def test_scaled_circular_pipeline(self):
        G = _make_graph(p=4, rings=2)
        CP = make_intersecting_cylinders(G, circular_profile(scale=1.0), r=1.0)
        CP.check_consistency()
        assert len(CP.halfedges) > 0


class TestTopView:
    @pytest.mark.parametrize("r", [1.0, 0.7])
    def test_top_view_consistent(self, r):
        G = _make_graph(p=4, rings=2)
        tv = top_view(G, r=r)
        tv.check_consistency()
        assert len(tv.faces) > 0

    def test_top_view_does_not_mutate_input(self):
        G = _make_graph(p=4, rings=2)
        n_faces_before = len(G.faces)
        _ = top_view(G, r=1.0)
        assert len(G.faces) == n_faces_before


class TestMesh3d:
    @pytest.mark.parametrize("r", [1.0, 0.7])
    def test_to_3d_mesh_shape(self, r):
        G = _make_graph(p=4, rings=2)
        verts, tris = to_3d_mesh(G, circular_profile(scale=1.0), r=r, n_along_edge=4)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert tris.ndim == 2 and tris.shape[1] == 3
        # All triangle indices reference valid vertices.
        assert tris.min() >= 0
        assert tris.max() < len(verts)

    def test_to_3d_mesh_z_nonpositive(self):
        G = _make_graph(p=4, rings=2)
        verts, _ = to_3d_mesh(G, circular_profile(scale=1.0), r=1.0, n_along_edge=4)
        # Incenters and edge tangent points sit at z=0; vertex spikes go down.
        assert verts[:, 2].max() == pytest.approx(0.0)
        assert verts[:, 2].min() < 0.0

    def test_half_cylinder_apex_height(self):
        # For platonic 4 with circular_profile(scale=1.0) and r=1, every face
        # folds into a half-cylinder of radius R = inradius. The vertex-spike
        # depth equals the blue-circle radius r_v, which for the unit square
        # equals the inradius 0.5.
        G = _make_graph(p=4, rings=2)
        verts, _ = to_3d_mesh(G, circular_profile(scale=1.0), r=1.0, n_along_edge=4)
        assert verts[:, 2].min() == pytest.approx(-0.5, rel=1e-3)
        assert verts[:, 2].max() == pytest.approx(0.0)

    def test_r_less_than_one_inner_face_at_zero(self):
        # With the spike geometry, the incenters and the lifted shrunken inner
        # faces sit at z=0 while vertex spikes go down. Spike depth is
        # r_v * scale * (1 - apex_perp). For platonic 4 r_v = 0.5; the spike
        # depth (and therefore the most-negative z) shrinks with apex_perp.
        profile = circular_profile(scale=1.0)
        sf = profile.shrink_factor
        r = 0.7
        apex_perp = (1 - r) * sf / (1 - (1 - r) * (1 - sf))
        expected_min = -0.5 * 1.0 * (1.0 - apex_perp)

        G = _make_graph(p=4, rings=2)
        verts, _ = to_3d_mesh(G, profile, r=r, n_along_edge=4)
        assert verts[:, 2].max() == pytest.approx(0.0)
        assert verts[:, 2].min() == pytest.approx(expected_min, rel=2e-3)

    def test_to_3d_mesh_invalid_r(self):
        G = _make_graph(p=4, rings=2)
        with pytest.raises(ValueError):
            to_3d_mesh(G, circular_profile(), r=0.0)

    def test_show_3d_returns_figure(self):
        plotly = pytest.importorskip("plotly")
        G = _make_graph(p=4, rings=2)
        fig = show_3d(G, circular_profile(scale=1.0), r=1.0, n_along_edge=4)
        assert isinstance(fig, plotly.graph_objects.Figure)


class TestTriangleTwist:
    @staticmethod
    def _hub_cp():
        import eucare as ec

        G = ec.io.load_graph("graphs/irregular2.heg")
        G = ec.conway.kis_graph()(G, delete_on_border=True)
        return make_intersecting_cylinders(G, circular_profile(), r=1.0)

    def test_convert_all(self):
        CP = self._hub_cp()
        hubs_before = [v for v in CP.vertices if v.order() == 6 and not v.on_border()]
        assert len(hubs_before) > 0
        convert_all_to_triangle_twists(CP)
        CP.check_consistency()
        hubs_after = [v for v in CP.vertices if v.order() == 6 and not v.on_border()]
        assert len(hubs_after) == 0

    def test_convert_single(self):
        CP = self._hub_cp()
        hubs = [v for v in CP.vertices if v.order() == 6 and not v.on_border()]
        assert len(hubs) > 0
        convert_to_triangle_twist(CP, hubs[0])
        CP.check_consistency()
