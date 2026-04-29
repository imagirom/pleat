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
    spherical_profile,
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

    def test_spherical_profile_flat_region(self):
        flat_until = 0.6
        p = spherical_profile(flat_until=flat_until)
        # The first segment of the (t, l) curve is flat (l = 0).
        assert p.l[0] == pytest.approx(0.0)
        # Some points should still lie in the flat region.
        flat_t = p.t[p.l == 0]
        assert flat_t.size >= 1

    def test_spherical_profile_rejects_bad_arg(self):
        with pytest.raises(ValueError):
            spherical_profile(flat_until=1.0)
        with pytest.raises(ValueError):
            spherical_profile(flat_until=-0.1)

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
            assert 'color_key' in h
        # Reverses share color.
        for h in CP.halfedges:
            assert h['color_key'] == h.rev['color_key']

    def test_curve_pos_on_red_edges(self):
        G = _make_graph(p=4, rings=2)
        CP = make_intersecting_cylinders(G, circular_profile(), r=1.0)
        red = [h for h in CP.halfedges if h['color_key'] == (1.0, 0.0, 0.0)]
        assert len(red) > 0
        for h in red:
            assert 'curve_pos' in h
            curve = h['curve_pos']
            assert curve.ndim == 2 and curve.shape[1] == 2
            # Reverse halfedge stores reversed polyline.
            np.testing.assert_allclose(h['curve_pos'], h.rev['curve_pos'][::-1])

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

    def test_spherical_profile_pipeline(self):
        G = _make_graph(p=4, rings=2)
        CP = make_intersecting_cylinders(G, spherical_profile(flat_until=0.5), r=1.0)
        CP.check_consistency()
        assert len(CP.halfedges) > 0


class TestTriangleTwist:
    @staticmethod
    def _hub_cp():
        import eucare as ec
        G = ec.io.load_graph('graphs/irregular2.heg')
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
