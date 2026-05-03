"""Tests for eucare.base geometry utilities."""

import numpy as np
import pytest

from eucare.base import (
    angle,
    angle_to_axis,
    apply_affine,
    barycentric_to_euclidean_map,
    edge_lengths,
    euclidean_to_barycentric_map,
    find_affine,
    in_angles,
    line_intersection,
    nearest_neighbor,
    orientation,
    project_to_line,
    regular_poly_points,
    rotation_matrix,
    signed_area,
    tri_grid_point,
    unit_vector,
    unit_vector_to_vector,
)


class TestUnitVector:
    def test_basic_directions(self):
        np.testing.assert_allclose(unit_vector(0), [1, 0], atol=1e-15)
        np.testing.assert_allclose(unit_vector(np.pi / 2), [0, 1], atol=1e-15)
        np.testing.assert_allclose(unit_vector(np.pi), [-1, 0], atol=1e-15)

    def test_array_input(self):
        result = unit_vector(np.array([0, np.pi / 2]))
        assert result.shape == (2, 2)

    def test_unit_length(self):
        alphas = np.linspace(0, 2 * np.pi, 100)
        vectors = unit_vector(alphas)
        norms = np.linalg.norm(vectors, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-15)


class TestAngleToAxis:
    def test_basic(self):
        assert abs(angle_to_axis(np.array([1, 0]))) < 1e-15
        assert abs(angle_to_axis(np.array([0, 1])) - np.pi / 2) < 1e-15


class TestRegularPolyPoints:
    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8, 12])
    def test_n_points(self, n):
        pts = regular_poly_points(n)
        assert pts.shape == (n, 2)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_equal_side_lengths(self, n):
        pts = regular_poly_points(n)
        lengths = edge_lengths(pts)
        np.testing.assert_allclose(lengths, lengths[0], atol=1e-12)

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_equal_angles(self, n):
        pts = regular_poly_points(n)
        angles = in_angles(pts)
        expected = np.pi * (n - 2) / n
        np.testing.assert_allclose(angles, expected, atol=1e-12)


class TestRotationMatrix:
    def test_identity(self):
        np.testing.assert_allclose(rotation_matrix(0), np.eye(2), atol=1e-15)

    def test_quarter_turn(self):
        v = np.array([1, 0])
        rotated = v @ rotation_matrix(np.pi / 2)
        np.testing.assert_allclose(rotated, [0, 1], atol=1e-15)


class TestSignedArea:
    def test_unit_square_ccw(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert abs(signed_area(pts) - 1.0) < 1e-12

    def test_unit_square_cw(self):
        pts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert abs(signed_area(pts) + 1.0) < 1e-12

    def test_triangle(self):
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        assert abs(signed_area(pts) - 0.5) < 1e-12


class TestOrientation:
    def test_ccw(self):
        pts = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        assert orientation(pts) == 1

    def test_cw(self):
        pts = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)
        assert orientation(pts) == -1

    def test_colinear(self):
        pts = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
        assert orientation(pts) == 0


class TestLineIntersection:
    def test_perpendicular(self):
        line1 = np.array([[0, 0], [1, 0]], dtype=float)
        line2 = np.array([[0.5, -1], [0.5, 1]], dtype=float)
        result = line_intersection(line1, line2)
        np.testing.assert_allclose(result, [0.5, 0], atol=1e-12)

    def test_parallel_raises(self):
        line1 = np.array([[0, 0], [1, 0]], dtype=float)
        line2 = np.array([[0, 1], [1, 1]], dtype=float)
        with pytest.raises(ValueError, match="do not intersect"):
            line_intersection(line1, line2)


class TestFindAffine:
    def test_identity_like(self):
        line0 = np.array([[0, 0], [1, 0]], dtype=float)
        line1 = np.array([[0, 0], [1, 0]], dtype=float)
        M = find_affine(line0, line1)
        assert M.shape == (3, 2)


class TestProjectToLine:
    def test_project_origin_on_x_axis(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0]])
        pts = np.array([[0.5, 1.0]])
        result = project_to_line(line, pts)
        np.testing.assert_allclose(result, [[0.5, 0.0]], atol=1e-12)


class TestAngle:
    def test_right_angle(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([0.0, 1.0])
        # Angle at b from a to c, CCW from b->a to b->c.
        np.testing.assert_allclose(angle(a, b, c), np.pi / 2 * 3, atol=1e-12)


class TestUnitVectorToVector:
    def test_zero_alpha_returns_unit_vector(self):
        seg = np.array([[0.0, 0.0], [2.0, 0.0]])
        out = unit_vector_to_vector(0.0, seg)
        # Length 1, starts at seg[0], same direction.
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[1], [1.0, 0.0], atol=1e-12)


class TestTriGridPoint:
    def test_origin(self):
        np.testing.assert_allclose(tri_grid_point(0, 0), [0.0, 0.0])

    def test_basis(self):
        np.testing.assert_allclose(tri_grid_point(1, 0), [1.0, 0.0])


class TestApplyAffine:
    def test_translation(self):
        # 2x2 identity + translation by (1, 2)
        M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        out = apply_affine(pts, M)
        np.testing.assert_allclose(out, [[1.0, 2.0], [2.0, 2.0]], atol=1e-12)


class TestNearestNeighbor:
    def test_returns_index(self):
        data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        query = np.array([0.9, 0.0])
        pt, idx = nearest_neighbor(data, query, return_index=True)
        assert idx == 1
        np.testing.assert_allclose(pt, [1.0, 0.0])


class TestBarycentric:
    def test_roundtrip_centroid(self):
        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        to_bary = euclidean_to_barycentric_map(tri)
        from_bary = barycentric_to_euclidean_map(tri)
        centroid = np.mean(tri, axis=0)
        bary = to_bary(centroid)
        np.testing.assert_allclose(bary, [1 / 3, 1 / 3, 1 / 3], atol=1e-5)
        np.testing.assert_allclose(from_bary(bary), centroid, atol=1e-5)
