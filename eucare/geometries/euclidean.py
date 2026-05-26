"""Euclidean plane geometry backend.

Flat 2D geometry, used for ordinary tilings of the plane.  Points are 2D
numpy arrays; rotations are represented as 2x2 matrices.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from .base import Geometry, Transform


class EuclideanGeometry(Geometry):
    """Flat 2D Euclidean geometry with standard vector operations."""

    @classmethod
    def origin(cls) -> NDArray[Any]:
        return np.array([0, 0])

    @classmethod
    def translation(cls, p1: NDArray[Any], p2: NDArray[Any]) -> Transform:
        def translate(p: NDArray[Any]) -> NDArray[Any]:
            return p + p2 - p1

        return translate

    @classmethod
    def _rotate_around_origin(cls, a1: float) -> Transform:
        rot_mat = np.array([[np.cos(a1), np.sin(a1)], [-np.sin(a1), np.cos(a1)]])

        def origin_rotate(p: NDArray[Any]) -> NDArray[Any]:
            return p @ rot_mat

        return origin_rotate

    @classmethod
    def center_of_mass(cls, points: NDArray[Any], masses: NDArray[Any] | None = None) -> NDArray[Any]:
        assert len(points.shape) == 2 and points.shape[-1] == 2, f"{points.shape}"
        if masses is not None:
            masses = masses / np.sum(masses) * len(points)
            points = points * masses[..., None]
        return np.mean(points, axis=0)

    @classmethod
    def distance_to_origin(cls, p: NDArray[Any]) -> float:
        return float(np.linalg.norm(p))

    @classmethod
    def angle_to_axis(cls, p: NDArray[Any]) -> float:
        return np.arctan2(p[..., 1], p[..., 0])

    @classmethod
    def point_along_axis(cls, x: float) -> NDArray[Any]:
        return np.array([x, 0])

    @classmethod
    def to_euclidean(cls, pts: NDArray[Any]) -> NDArray[Any]:
        return pts

    @classmethod
    def archimedean_side_length(cls, faces_around_corner: Iterable[int], eps: float = 1e-6) -> float:
        euclidean_vertex_angle = sum(np.pi * (n - 2) / n for n in faces_around_corner)
        if abs(euclidean_vertex_angle - 2 * np.pi) < eps:
            return 1
        else:
            raise ValueError(f"Vertex {faces_around_corner} is impossible in euclidean geometry.")
