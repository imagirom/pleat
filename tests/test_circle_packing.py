"""Tests for the circle_packing module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from eucare.example_graphs import from_tiles
from eucare.example_tilesets import platonic
from eucare.geometries import EuclideanGeometry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "circlepack"
sys.path.insert(0, str(FIXTURE_DIR))


def _hex_triangulation(rings: int = 2):
    """Return a triangulated disk: regular {3, 6} tiling with `rings` rings."""
    return from_tiles(platonic(3), rings=rings)


class TestPackEuclidean:
    def test_returns_euclidean_position_heg(self):
        from eucare.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        assert P.geometry is EuclideanGeometry

    def test_every_vertex_has_radius_attribute(self):
        from eucare.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        for v in P.vertices:
            assert "radius" in v.attributes
            assert v["radius"] > 0

    def test_boundary_radii_match_uniform_input(self):
        from eucare.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=0.7)
        for v in P.vertices:
            if v.on_border():
                assert v["radius"] == pytest.approx(0.7, abs=1e-12)

    def test_tangency_holds_at_every_edge(self):
        from eucare.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        for h in P.halfedges:
            if h.face is None:
                continue  # border halfedge — its rev is the interior copy
            u, v = h.orig, h.dest
            d = float(np.linalg.norm(u["pos"] - v["pos"]))
            assert d == pytest.approx(u["radius"] + v["radius"], abs=1e-9)

    def test_interior_angle_sums_are_2pi(self):
        from eucare.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        for v in P.vertices:
            if v.on_border():
                continue
            angle_sum = 0.0
            for h in v.outgoing_iter():
                if h.face is None:
                    continue
                # angle at v in the triangle (v, h.dest, h.pre.orig)
                a = v["pos"]
                b = h.dest["pos"]
                c = h.pre.orig["pos"]  # third triangle vertex
                ab = b - a
                ac = c - a
                cos_angle = float(np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac)))
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle_sum += float(np.arccos(cos_angle))
            assert angle_sum == pytest.approx(2 * np.pi, abs=1e-8)

    def test_regular_hex_lattice_yields_uniform_radii(self):
        """In a regular {3,6} tiling with uniform boundary, all radii equal."""
        from eucare.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        radii = [v["radius"] for v in P.vertices]
        assert all(r == pytest.approx(1.0, abs=1e-9) for r in radii)

    def test_raises_on_non_triangulated_input(self):
        from eucare.circle_packing import pack_euclidean

        # platonic(6) = hexagonal tiling — faces are hexagons, not triangles
        G = from_tiles(platonic(6), rings=2)
        with pytest.raises(ValueError, match="triangul"):
            pack_euclidean(G, boundary_radii=1.0)


class TestPackHyperbolic:
    def test_returns_hyperbolic_position_heg(self):
        from eucare.circle_packing import pack_hyperbolic
        from eucare.geometries import PoincareDiskModel

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        assert P.geometry is PoincareDiskModel

    def test_radii_are_positive_and_inside_disk(self):
        from eucare.circle_packing import pack_hyperbolic

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        for v in P.vertices:
            assert "radius" in v.attributes
            assert v["radius"] > 0
            # Each circle (center + radius) fits inside the unit disk.
            assert abs(v["pos"]) + v["radius"] <= 1.0 + 1e-9

    def test_positions_inside_unit_disk(self):
        from eucare.circle_packing import pack_hyperbolic

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        for v in P.vertices:
            z = v["pos"]
            assert abs(z) < 1.0

    def test_euclidean_tangency_holds(self):
        """In the Poincaré model, hyperbolic tangency = euclidean tangency."""
        from eucare.circle_packing import pack_hyperbolic

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        for h in P.halfedges:
            if h.face is None:
                continue
            u, v = h.orig, h.dest
            d = abs(u["pos"] - v["pos"])
            assert d == pytest.approx(u["radius"] + v["radius"], abs=1e-9)

    def test_interior_hyperbolic_angle_sums_are_2pi(self):
        """Interior x-radii (recovered from stored (c, r_euc)) satisfy angle-sum = 2π."""
        from eucare.circle_packing import (
            pack_hyperbolic,
            _hyperbolic_angle_sum,
            _x_radius_from_euclidean,
        )

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        # Convert each vertex's stored (c, r_euc) back to its intrinsic x-radius.
        x_radii = {v: _x_radius_from_euclidean(v["pos"], v["radius"]) for v in P.vertices}
        for v in P.vertices:
            if v.on_border():
                continue
            pairs = []
            for h in v.outgoing_iter():
                if h.face is None:
                    continue
                u = h.dest
                w = h.nex.dest
                pairs.append((x_radii[u], x_radii[w]))
            theta = _hyperbolic_angle_sum(x_radii[v], pairs)
            assert theta == pytest.approx(2 * np.pi, abs=1e-7)

    def test_horocycle_boundary_supported(self):
        """boundary_x_radii = 1.0 produces a maximal packing: boundary tangent to unit circle."""
        from eucare.circle_packing import pack_hyperbolic

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=1.0)
        # Each boundary vertex's circle is internally tangent to the unit circle:
        # |c| + r = 1.
        for v in P.vertices:
            if v.on_border():
                assert abs(v["pos"]) + v["radius"] == pytest.approx(1.0, abs=1e-9)
        # Each pair of adjacent circles is tangent.
        for h in P.halfedges:
            if h.face is None:
                continue
            u, v = h.orig, h.dest
            d = abs(u["pos"] - v["pos"])
            assert d == pytest.approx(u["radius"] + v["radius"], abs=1e-9)


@pytest.mark.golden
class TestGoldenAgainstCirclePack:
    """Compare radii against CirclePack-generated .p fixtures (opt-in via marker)."""

    @staticmethod
    def _load(name: str):
        from _parser import parse_p_file, build_heg_from_flowers  # type: ignore

        data = parse_p_file(str(FIXTURE_DIR / name))
        G, idx2v = build_heg_from_flowers(data)
        return data, G, idx2v

    def test_egg_a_euclidean(self):
        """egg_a.p: 24-vertex euclidean packing with uniform boundary radius 0.025."""
        from eucare.circle_packing import pack_euclidean

        data, G, idx2v = self._load("egg_a.p")
        # Extract boundary radii from the .p file (boundary vertices are those
        # where flower[0] != flower[-1]).
        boundary_radii = {}
        for i, neighbors in data.flowers.items():
            if neighbors[0] != neighbors[-1]:
                boundary_radii[idx2v[i]] = float(data.radii[i])
        P = pack_euclidean(G, boundary_radii=boundary_radii, copy_graph=False)
        # Compare per-vertex radii against the file.
        for i in range(data.nodecount):
            v = idx2v[i]
            expected = float(data.radii[i])
            got = float(v["radius"])
            assert got == pytest.approx(expected, rel=1e-6, abs=1e-8), f"vertex {i+1}: expected {expected}, got {got}"

    def test_circular1_euclidean(self):
        """circular1.p: 37-vertex euclidean packing."""
        from eucare.circle_packing import pack_euclidean

        data, G, idx2v = self._load("circular1.p")
        boundary_radii = {}
        for i, neighbors in data.flowers.items():
            if neighbors[0] != neighbors[-1]:
                boundary_radii[idx2v[i]] = float(data.radii[i])
        P = pack_euclidean(G, boundary_radii=boundary_radii, copy_graph=False)
        for i in range(data.nodecount):
            v = idx2v[i]
            expected = float(data.radii[i])
            got = float(v["radius"])
            assert got == pytest.approx(expected, rel=1e-6, abs=1e-8), f"vertex {i+1}: expected {expected}, got {got}"

    def test_hyp_7575around6_hyperbolic(self):
        """hyp_7575around6.p: 18-vertex hyperbolic packing with finite boundary x-radii.

        The .p file stores x-radii; our output stores euclidean radii in the
        Poincaré disk. Convert back to compare. Note: the file's boundary
        radii are slightly non-uniform (0.0501-0.0508), suggesting CirclePack
        also iterated boundary radii; we hold ours fixed, so interior radii
        are compared with ~5% tolerance.
        """
        from eucare.circle_packing import pack_hyperbolic, _x_radius_from_euclidean

        data, G, idx2v = self._load("hyp_7575around6.p")
        boundary_x_radii = {}
        for i, neighbors in data.flowers.items():
            if neighbors[0] != neighbors[-1]:
                boundary_x_radii[idx2v[i]] = float(data.radii[i])
        P = pack_hyperbolic(G, boundary_x_radii=boundary_x_radii, copy_graph=False)
        # Compare per-vertex x-radii (recovered from stored euclidean form).
        for i in range(data.nodecount):
            v = idx2v[i]
            expected = float(data.radii[i])
            got_x = _x_radius_from_euclidean(v["pos"], v["radius"])
            if v.on_border():
                assert got_x == pytest.approx(
                    expected, rel=1e-6, abs=1e-9
                ), f"boundary vertex {i+1}: expected x={expected}, got x={got_x}"
            else:
                assert got_x == pytest.approx(
                    expected, rel=0.05
                ), f"interior vertex {i+1}: expected x={expected}, got x={got_x}"
