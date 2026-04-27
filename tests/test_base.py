"""Tests for eucare.base geometry utilities."""
import numpy as np
import pytest
from eucare.base import (
    unit_vector, angle_to_axis, angle, in_angles, edge_lengths,
    regular_poly_points, rotation_matrix, signed_area, orientation,
    line_intersection, project_to_line, find_affine,
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
