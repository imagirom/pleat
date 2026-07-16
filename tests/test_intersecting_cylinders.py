"""Tests for the intersecting-cylinders subpackage."""

import numpy as np
import pytest

from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.half import IdObject
from pleat.intersecting_cylinders import (
    Profile,
    circular_profile,
    convert_all_to_triangle_twists,
    convert_to_triangle_twist,
    build_dual_circle_packings,
    make_intersecting_cylinders,
    show_3d,
    show_dual_circle_packings,
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
        verts, tris = to_3d_mesh(G, circular_profile(scale=1.0), r=r, n_across_edge=4)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert tris.ndim == 2 and tris.shape[1] == 3
        # All triangle indices reference valid vertices.
        assert tris.min() >= 0
        assert tris.max() < len(verts)

    def test_to_3d_mesh_z_nonpositive(self):
        G = _make_graph(p=4, rings=2)
        verts, _ = to_3d_mesh(G, circular_profile(scale=1.0), r=1.0, n_across_edge=4)
        # Incenters and edge tangent points sit at z=0; vertex spikes go down.
        assert verts[:, 2].max() == pytest.approx(0.0)
        assert verts[:, 2].min() < 0.0

    def test_half_cylinder_apex_height(self):
        # For platonic 4 with circular_profile(scale=1.0) and r=1, every face
        # folds into a half-cylinder of radius R = inradius. The vertex-spike
        # depth equals the blue-circle radius r_v, which for the unit square
        # equals the inradius 0.5.
        G = _make_graph(p=4, rings=2)
        verts, _ = to_3d_mesh(G, circular_profile(scale=1.0), r=1.0, n_across_edge=4)
        assert verts[:, 2].min() == pytest.approx(-0.5, rel=1e-3)
        assert verts[:, 2].max() == pytest.approx(0.0)

    def test_r_less_than_one_flat_tip_at_vertex(self):
        # For r<1 the cylinder is rescaled proportionally and capped by a flat
        # polygon centred at each original tiling vertex. The flat tip sits at
        # the rescaled cylinder's full depth: for a circular profile of scale 1
        # and platonic-4 vertices (r_v = 0.5) this is z = -r_v * curved_extent.
        profile = circular_profile(scale=1.0)
        sf = profile.shrink_factor
        r = 0.7
        apex_inset = (1 - r) * sf / (1 - (1 - r) * (1 - sf))
        curved_extent = 1.0 - apex_inset
        expected_tip = -0.5 * curved_extent

        G = _make_graph(p=4, rings=2)
        verts, _ = to_3d_mesh(G, profile, r=r, n_across_edge=4)
        assert verts[:, 2].max() == pytest.approx(0.0)
        assert verts[:, 2].min() == pytest.approx(expected_tip, rel=2e-3)

        # The flat tip should sit at each original tiling vertex, not between
        # edges. Mesh vertices exactly at the flat-tip depth (the corners of
        # the flat-tip triangles) should be very close to original tiling
        # vertices in the (x, y) plane: each is either an original vertex v
        # itself or one of c_near_v / t_near_v at distance apex_inset * |c-v|
        # from v.
        actual_tip = float(verts[:, 2].min())
        tip_mask = np.isclose(verts[:, 2], actual_tip, atol=1e-9)
        tip_xy = verts[tip_mask, :2]
        assert len(tip_xy) > 0
        orig_xy = np.array([v["pos"] for v in G.vertices], dtype=float)
        from scipy.spatial.distance import cdist  # type: ignore

        dists = cdist(tip_xy, orig_xy).min(axis=1)
        # The longest |c - v| in a platonic-4 tiling is sqrt(2)/2; allow a
        # small slack for the apex_inset interpolation factor.
        assert dists.max() < apex_inset * 0.71 + 1e-6

    @pytest.mark.parametrize("r", [1.0, 0.7])
    def test_to_3d_mesh_normals_consistently_oriented(self, r):
        # The folded surface is a single-valued height field (z=0 at the
        # incenters/face centres, dipping down to the vertex spikes / flat
        # tips), so every triangle's normal must point the same way. Plotly's
        # Mesh3d shades from the i/j/k winding order, so mixed winding
        # renders half the surface dark/inverted.
        G = _make_graph(p=4, rings=2)
        verts, tris = to_3d_mesh(G, circular_profile(scale=1.0), r=r, n_across_edge=4)
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        nz = np.cross(b - a, c - a)[:, 2]
        nz = nz[np.abs(nz) > 1e-9]  # ignore degenerate / vertical triangles
        assert len(nz) > 0
        assert np.all(nz > 0), f"mesh winding inconsistent: {(nz < 0).sum()} of {len(nz)} " "triangles face downward"

    def test_to_3d_mesh_invalid_r(self):
        G = _make_graph(p=4, rings=2)
        with pytest.raises(ValueError):
            to_3d_mesh(G, circular_profile(), r=0.0)

    def test_profile_orientation_flat_base_pointy_apex(self):
        # The 3D surface should be a "spike" surface: nearly flat in a broad
        # region near the c-t base of each half-triangle (where neighbouring
        # patches meet smoothly), and concentrate the depth into pointy spikes
        # at the original tiling vertices. A monotone profile of the
        # complementary shape (steep near base, flat near apex) would invert
        # the distribution -- most of the surface area would be deep.
        G = _make_graph(p=4, rings=1)
        verts, tris = to_3d_mesh(G, circular_profile(scale=1.0), r=1.0, n_across_edge=20)
        zs = verts[:, 2]
        assert zs.min() == pytest.approx(-0.5, rel=1e-3)
        assert zs.max() == pytest.approx(0.0)
        # Measure by triangle area (not vertex count): the profile-aware
        # sampling deliberately concentrates samples near the steep apex,
        # so vertex-count metrics don't reflect surface area.
        a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        tri_z = (a[:, 2] + b[:, 2] + c[:, 2]) / 3.0
        total = areas.sum()
        near_base = areas[tri_z > -0.05].sum() / total
        deep = areas[tri_z < -0.4].sum() / total
        # With the correct (flat-base, pointy-apex) orientation, most of the
        # surface area sits near the flat base. With the inverted profile
        # these ratios would flip.
        assert near_base > 0.5
        assert deep < 0.05

    def test_show_3d_returns_figure(self):
        plotly = pytest.importorskip("plotly")
        G = _make_graph(p=4, rings=2)
        fig = show_3d(G, circular_profile(scale=1.0), r=1.0, n_across_edge=4)
        assert isinstance(fig, plotly.graph_objects.Figure)


class TestDualCirclePackings:
    def test_does_not_mutate_input(self):
        G = _make_graph(p=4, rings=2)
        n_v_before = len(G.vertices)
        n_f_before = len(G.faces)
        _ = build_dual_circle_packings(G)
        assert len(G.vertices) == n_v_before
        assert len(G.faces) == n_f_before

    def test_returns_consistent_graph(self):
        G = _make_graph(p=4, rings=2)
        G_ortho = build_dual_circle_packings(G)
        G_ortho.check_consistency()

    def test_styling_attributes_present(self):
        from pleat import half

        G = _make_graph(p=4, rings=2)
        G_ortho = build_dual_circle_packings(G)

        face_circles = vertex_circles = tangent_points = 0
        for v in G_ortho.vertices.union(G_ortho.faces):
            assert "color_key" in v
            assert "vertex_radius" in v
            pre = v.get("pre_conway")
            if isinstance(pre, half.Face):
                assert v["color_key"] == (1.0, 0.0, 0.0, 0.3)
                assert v["vertex_radius"] > 0.0
                face_circles += 1
            elif isinstance(pre, half.Vertex):
                assert v["color_key"] == (0.0, 0.0, 1.0, 0.3)
                assert v["vertex_radius"] > 0.0
                vertex_circles += 1
            else:
                tangent_points += 1
        assert face_circles > 0
        assert vertex_circles > 0

        for h in G_ortho.halfedges:
            assert h["color_key"] == (0.0, 0.0, 0.0)

    def test_face_and_vertex_circles_match_tangent_distance(self):
        # The radius assigned to each face/vertex circle is the distance from
        # its centre to *any* adjacent tangent point. By construction all
        # tangent points on the same circle should be equidistant.
        from pleat import half

        G = _make_graph(p=4, rings=2)
        G_ortho = build_dual_circle_packings(G)
        for v in G_ortho.vertices.union(G_ortho.faces):
            pre = v.get("pre_conway")
            if not isinstance(pre, (half.Face, half.Vertex)):
                continue
            r = v["vertex_radius"]
            dists = [float(np.linalg.norm(v["pos"] - h.dest["pos"])) for h in v.outgoing_iter()]
            assert dists, "expected at least one outgoing halfedge"
            np.testing.assert_allclose(dists, r, atol=1e-9)

    def test_show_dual_circle_packings_runs(self):
        # Smoke test only: rendering may pop a matplotlib figure but must not
        # raise.
        import matplotlib

        matplotlib.use("Agg")
        G = _make_graph(p=4, rings=2)
        show_dual_circle_packings(G)


class TestTriangleTwist:
    @staticmethod
    def _hub_cp():
        import pleat as ec

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
