"""Tests for the circle_packing module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.geometries import EuclideanGeometry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "circlepack"


def _hex_triangulation(rings: int = 2):
    """Return a triangulated disk: regular {3, 6} tiling with `rings` rings."""
    return from_tiles(platonic(3), rings=rings)


class TestPackEuclidean:
    def test_returns_euclidean_position_heg(self):
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        assert P.geometry is EuclideanGeometry

    def test_every_vertex_has_radius_attribute(self):
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        for v in P.vertices:
            assert "radius" in v.attributes
            assert v["radius"] > 0

    def test_boundary_radii_match_uniform_input(self):
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=0.7)
        for v in P.vertices:
            if v.on_border():
                assert v["radius"] == pytest.approx(0.7, abs=1e-12)

    def test_tangency_holds_at_every_edge(self):
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        for h in P.halfedges:
            if h.face is None:
                continue  # border halfedge — its rev is the interior copy
            u, v = h.orig, h.dest
            d = float(np.linalg.norm(u["pos"] - v["pos"]))
            assert d == pytest.approx(u["radius"] + v["radius"], abs=1e-9)

    def test_interior_angle_sums_are_2pi(self):
        from pleat.circle_packing import pack_euclidean

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
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_radii=1.0)
        radii = [v["radius"] for v in P.vertices]
        assert all(r == pytest.approx(1.0, abs=1e-9) for r in radii)

    def test_raises_on_non_triangulated_input(self):
        from pleat.circle_packing import pack_euclidean

        # platonic(6) = hexagonal tiling — faces are hexagons, not triangles
        G = from_tiles(platonic(6), rings=2)
        with pytest.raises(ValueError, match="triangul"):
            pack_euclidean(G, boundary_radii=1.0)


class TestBoundaryAngles:
    def test_boundary_angles_from_positions_recovers_input_angles(self):
        """Angles computed from a flat tiling sum to π(F - 2V_int) automatically."""
        from pleat.circle_packing import boundary_angles_from_positions

        G = _hex_triangulation()
        angles = boundary_angles_from_positions(G)
        n_faces = len(G.faces)
        n_int = sum(1 for v in G.vertices if not v.on_border())
        expected = np.pi * (n_faces - 2 * n_int)
        assert sum(angles.values()) == pytest.approx(expected, abs=1e-10)

    def test_pack_euclidean_with_from_positions_angles_matches_input_layout(self):
        """Angle-mode roundtrip on a regular hex patch: resulting packing is the input (up to gauge)."""
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G, boundary_angles="from_positions")
        # All interior angle sums equal 2π.
        for v in P.vertices:
            if v.on_border():
                continue
            angle_sum = 0.0
            for h in v.outgoing_iter():
                if h.face is None:
                    continue
                a = v["pos"]
                b = h.dest["pos"]
                c = h.pre.orig["pos"]
                ab = b - a
                ac = c - a
                cos_a = float(np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac)))
                cos_a = max(-1.0, min(1.0, cos_a))
                angle_sum += float(np.arccos(cos_a))
            assert angle_sum == pytest.approx(2 * np.pi, abs=1e-8)
        # Edge tangencies hold (basic sanity).
        for h in P.halfedges:
            if h.face is None:
                continue
            u, v = h.orig, h.dest
            d = float(np.linalg.norm(u["pos"] - v["pos"]))
            assert d == pytest.approx(u["radius"] + v["radius"], abs=1e-8)

    def test_scalar_boundary_angle_validates_sum(self):
        """A uniform scalar broadcasts to all boundary vertices; the resulting
        sum must satisfy Gauss-Bonnet or the call must error."""
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        # 1.0 is essentially never the per-vertex angle satisfying Σθ = π(F − 2V_int)
        # on a generic tiling — must raise.
        with pytest.raises(ValueError, match="Gauss-Bonnet|sum"):
            pack_euclidean(G, boundary_angles=1.0)

    def test_bad_sum_raises(self):
        """Manual angles that don't satisfy Σ θ = π(F - 2V_int) must error."""
        from pleat.circle_packing import boundary_angles_from_positions, pack_euclidean

        G = _hex_triangulation()
        angles = boundary_angles_from_positions(G)
        # Perturb one angle so the global sum is wrong. Use copy_graph=False so
        # the dict (keyed by G's vertices) lines up with the working graph.
        first = next(iter(angles))
        angles[first] += 0.1
        with pytest.raises(ValueError, match="Gauss-Bonnet|sum"):
            pack_euclidean(G, boundary_angles=angles, copy_graph=False)

    def test_specifying_both_radii_and_angles_raises(self):
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        with pytest.raises(ValueError, match="not both"):
            pack_euclidean(G, boundary_radii=1.0, boundary_angles=1.0)

    def test_default_is_uniform_radii(self):
        """When neither boundary_radii nor boundary_angles is given, defaults to uniform radii=1."""
        from pleat.circle_packing import pack_euclidean

        G = _hex_triangulation()
        P = pack_euclidean(G)
        for v in P.vertices:
            if v.on_border():
                assert v["radius"] == pytest.approx(1.0, abs=1e-9)


class TestPackHyperbolic:
    def test_returns_hyperbolic_position_heg(self):
        from pleat.circle_packing import pack_hyperbolic
        from pleat.geometries import PoincareDiskModel

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        assert P.geometry is PoincareDiskModel

    def test_radii_are_positive_and_inside_disk(self):
        from pleat.circle_packing import pack_hyperbolic

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        for v in P.vertices:
            assert "radius" in v.attributes
            assert v["radius"] > 0
            # Each circle (center + radius) fits inside the unit disk.
            assert abs(v["pos"]) + v["radius"] <= 1.0 + 1e-9

    def test_positions_inside_unit_disk(self):
        from pleat.circle_packing import pack_hyperbolic

        G = _hex_triangulation()
        P = pack_hyperbolic(G, boundary_x_radii=0.5)
        for v in P.vertices:
            z = v["pos"]
            assert abs(z) < 1.0

    def test_euclidean_tangency_holds(self):
        """In the Poincaré model, hyperbolic tangency = euclidean tangency."""
        from pleat.circle_packing import pack_hyperbolic

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
        from pleat.circle_packing import (
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
        from pleat.circle_packing import pack_hyperbolic

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


class TestGoldenAgainstCirclePack:
    """Compare radii against CirclePack-generated .p fixtures."""

    @staticmethod
    def _load(name: str):
        from pleat.io.circlepack import parse_p_file, _build_heg_from_data

        data = parse_p_file(str(FIXTURE_DIR / name))
        G, idx2v = _build_heg_from_data(data)
        return data, G, idx2v

    def test_egg_a_euclidean(self):
        """egg_a.p: 24-vertex euclidean packing with uniform boundary radius 0.025."""
        from pleat.circle_packing import pack_euclidean

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
        from pleat.circle_packing import pack_euclidean

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
        from pleat.circle_packing import pack_hyperbolic, _x_radius_from_euclidean

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
