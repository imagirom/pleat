"""Tests for `.p` (CirclePack) file I/O via :mod:`pleat.io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.geometries import EuclideanGeometry, PoincareDiskModel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "circlepack"


def _hex_triangulation(rings: int = 2):
    return from_tiles(platonic(3), rings=rings)


# ---------------------------------------------------------------------------
# Low-level parser / writer (CirclePackData round-trip)
# ---------------------------------------------------------------------------


class TestParseWriteIdentity:
    """parse → write → parse must be the identity at the CirclePackData level."""

    @pytest.mark.parametrize("name", ["egg_a.p", "circular1.p", "hyp_7575around6.p"])
    def test_roundtrip_identity(self, tmp_path, name):
        from pleat.io import parse_p_file, write_p_file

        data = parse_p_file(str(FIXTURE_DIR / name))
        out = tmp_path / name
        write_p_file(str(out), data)
        data2 = parse_p_file(str(out))

        assert data2.nodecount == data.nodecount
        assert data2.geometry == data.geometry
        assert data2.alpha == data.alpha
        assert data2.beta == data.beta
        assert data2.gamma == data.gamma
        assert data2.flowers == data.flowers
        if data.radii is None:
            assert data2.radii is None
        else:
            np.testing.assert_allclose(data2.radii, data.radii, rtol=1e-9, atol=1e-12)
        if data.centers is None:
            assert data2.centers is None
        else:
            np.testing.assert_allclose(data2.centers, data.centers, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# High-level: load_circlepack / save_circlepack
# ---------------------------------------------------------------------------


class TestLoadCirclepack:
    def test_egg_a_combinatorics(self):
        from pleat.io import load_circlepack

        G = load_circlepack(str(FIXTURE_DIR / "egg_a.p"))
        # egg_a.p declares NODECOUNT 24
        assert len(list(G.vertices)) == 24
        # geometry: euclidean
        assert G.geometry is EuclideanGeometry
        # every vertex has pos and radius attributes
        for v in G.vertices:
            assert "pos" in v.attributes
            assert "radius" in v.attributes

    def test_egg_a_tangency(self):
        """The egg_a.p fixture stores a valid packing — every edge satisfies tangency."""
        from pleat.io import load_circlepack

        G = load_circlepack(str(FIXTURE_DIR / "egg_a.p"))
        for h in G.halfedges:
            if h.face is None:
                continue
            u, v = h.orig, h.dest
            d = float(np.linalg.norm(np.asarray(u["pos"]) - np.asarray(v["pos"])))
            assert d == pytest.approx(u["radius"] + v["radius"], abs=5e-3)

    def test_hyperbolic_geometry_recognised(self):
        from pleat.io import load_circlepack

        G = load_circlepack(str(FIXTURE_DIR / "hyp_7575around6.p"))
        assert G.geometry is PoincareDiskModel


class TestSaveLoadRoundtrip:
    def test_euclidean_packed_graph_roundtrip(self, tmp_path):
        """Pack a hex triangulation, save, reload — tangency should still hold."""
        from pleat.circle_packing import pack_euclidean
        from pleat.io import load_circlepack, save_circlepack

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        out = tmp_path / "packed.p"
        save_circlepack(str(out), P)
        P2 = load_circlepack(str(out))

        assert len(list(P.vertices)) == len(list(P2.vertices))
        assert len(list(P.faces)) == len(list(P2.faces))
        # Tangency: every interior edge satisfies |pu - pv| ≈ ru + rv
        for h in P2.halfedges:
            if h.face is None:
                continue
            u, v = h.orig, h.dest
            d = float(np.linalg.norm(np.asarray(u["pos"]) - np.asarray(v["pos"])))
            assert d == pytest.approx(u["radius"] + v["radius"], abs=1e-9)

    def test_euclidean_radii_preserved(self, tmp_path):
        """Save/load must preserve the multiset of radii (modulo vertex relabeling)."""
        from pleat.circle_packing import pack_euclidean
        from pleat.io import load_circlepack, save_circlepack

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        out = tmp_path / "packed.p"
        save_circlepack(str(out), P)
        P2 = load_circlepack(str(out))

        r1 = sorted(float(v["radius"]) for v in P.vertices)
        r2 = sorted(float(v["radius"]) for v in P2.vertices)
        np.testing.assert_allclose(r1, r2, rtol=1e-9, atol=1e-12)

    def test_save_load_egg_a_preserves_radii(self, tmp_path):
        """Load fixture, save, reload — multiset of radii unchanged."""
        from pleat.io import load_circlepack, save_circlepack

        G1 = load_circlepack(str(FIXTURE_DIR / "egg_a.p"))
        out = tmp_path / "egg_a_out.p"
        save_circlepack(str(out), G1)
        G2 = load_circlepack(str(out))
        r1 = sorted(float(v["radius"]) for v in G1.vertices)
        r2 = sorted(float(v["radius"]) for v in G2.vertices)
        np.testing.assert_allclose(r1, r2, rtol=1e-9, atol=1e-12)

    def test_overwrite_guard(self, tmp_path):
        """save_circlepack refuses to overwrite by default."""
        from pleat.circle_packing import pack_euclidean
        from pleat.io import save_circlepack

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        out = tmp_path / "packed.p"
        save_circlepack(str(out), P)
        with pytest.raises((FileExistsError, AssertionError)):
            save_circlepack(str(out), P)  # second call should refuse
        save_circlepack(str(out), P, overwrite=True)  # explicit overwrite OK
