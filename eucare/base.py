"""2D geometry primitives: vectors, angles, areas, intersections, and coordinate transforms.

All functions take and return ``numpy.ndarray`` of shape ``(..., 2)`` for points
and angles in radians.  The convention is right-handed: positive rotation is
counter-clockwise; :func:`signed_area` is positive for CCW polygons.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numba import jit
from numpy.typing import ArrayLike, NDArray

pi = np.pi
tau = 2 * np.pi


def unit_vector(alpha: ArrayLike) -> NDArray[np.float64]:
    """Return unit vector(s) at angle *alpha* (radians).

    Accepts a scalar or array; the trailing axis of length 2 is appended.
    """
    return np.stack([np.cos(alpha), np.sin(alpha)], axis=-1)


def angle_to_axis(vectors: NDArray) -> NDArray:
    """Return the angle (radians) each 2D vector makes with the positive x-axis.

    Result is in ``(-pi, pi]`` (per :func:`numpy.arctan2`).
    """
    return np.arctan2(vectors[..., 1], vectors[..., 0])


def angle(a: NDArray, b: NDArray, c: NDArray) -> NDArray:
    """Return the interior angle at vertex *b* on the path ``a -> b -> c``, in ``[0, 2*pi)``."""
    return (angle_to_axis(a - b) - angle_to_axis(c - b)) % (2 * np.pi)


def in_angles(points: NDArray) -> NDArray:
    """Interior angles of a polygon given its ordered vertices (shape ``(n, 2)``).

    Vertices are assumed to be in CCW order; angles are returned in ``[0, 2*pi)``.
    """
    edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    edge_angles = angle_to_axis(edge_vectors)
    return (np.pi + edge_angles - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2 * np.pi)


def edge_lengths(points: NDArray) -> NDArray:
    """Edge lengths of a polygon given its ordered vertices (shape ``(n, 2)``)."""
    edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    return np.linalg.norm(edge_vectors, axis=1)


def edge_lengths_and_in_angles(points: NDArray, geometry) -> tuple[list[float], list[float]]:
    """Return ``(edge_lengths, interior_angles)`` for a polygon under *geometry*.

    *geometry* must expose ``distance(p, q)`` and ``angle(p, q, r)`` (see
    :mod:`eucare.geometries`).
    """
    edge_lengths = [geometry.distance(p1, p2) for p1, p2 in zip(points, np.concatenate([points[1:], points[:1]]))]
    in_angles = [
        geometry.angle(p1, p2, p3)
        for p1, p2, p3 in zip(
            points, np.concatenate([points[1:], points[:1]]), np.concatenate([points[2:], points[:2]])
        )
    ]
    return edge_lengths, in_angles


def unit_vector_to_vector(alpha: float, vector: NDArray) -> NDArray:
    """Rotate direction *alpha* onto the line through ``vector[0]`` and ``vector[1]``.

    Returns a length-2 segment starting at ``vector[0]`` and pointing in the
    direction obtained by rotating ``vector[1] - vector[0]`` by *alpha*.
    """
    return np.array([vector[0], vector[0] + unit_vector(angle_to_axis(vector[1] - vector[0]) + alpha)])


def tri_grid_point(i: int, j: int) -> NDArray:
    """Return the point at integer coordinates ``(i, j)`` on the standard triangular grid."""
    return unit_vector(0) * i + unit_vector(np.pi / 3) * j


def regular_poly_points(n: int) -> NDArray:
    """Vertices of a unit-edge-length regular *n*-gon centered at the origin."""
    return unit_vector(np.linspace(0, tau, n, endpoint=False)) / (2 * np.sin(pi / n))


def apply_affine(vectors: NDArray, matrix: NDArray) -> NDArray:
    """Apply a 2D affine transform to an array of 2D points.

    Args:
        vectors: Points of shape ``(..., 2)``.
        matrix: A ``(3, 2)`` matrix where rows 0-1 are the linear part and row 2 is the translation.
    """
    return np.dot(
        np.concatenate([vectors, np.ones_like(vectors[..., :1])], axis=-1),
        matrix,
    )


def rotation_matrix(alpha: float) -> NDArray:
    """Return a 2x2 rotation matrix.

    Use as ``point @ R`` to rotate *point* CCW by *alpha*.
    """
    s, c = np.sin(alpha), np.cos(alpha)
    return np.array([[c, s], [-s, c]])


def find_affine(line0: NDArray, line1: NDArray) -> NDArray:
    """Affine transform (``(3, 2)`` matrix) mapping directed segment *line0* to *line1*.

    The transform is rotation + uniform scaling + translation; in particular,
    it is conformal (no shear).
    """
    lines = np.stack([line0, line1])
    relative = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(relative, axis=1)
    scale = lengths[1] / lengths[0]
    angles = angle_to_axis(relative)
    angle = angles[1] - angles[0]
    linear = scale * rotation_matrix(angle)
    offset = lines[1, 1] - lines[0, 1].dot(linear)
    return np.concatenate([linear, offset[None]])


def nearest_neighbor(data: NDArray, query: NDArray, return_index: bool = True):
    """Return the nearest point in *data* to *query* (brute-force).

    Args:
        data: Points of shape ``(N, D)``.
        query: A single point of shape ``(D,)``.
        return_index: If True, return ``(point, index)``; otherwise return the point.
    """
    if len(data.shape) > len(query.shape):
        query = query[None]
    index = np.argmin(np.linalg.norm(data - query, axis=-1))
    return data[index], index if return_index else data[index]


@jit(nopython=True)
def signed_area(pts):
    """Signed area of a polygon (positive = CCW). Numba-accelerated.

    *pts* must have shape ``(n, 2)``; the polygon is closed implicitly.
    """
    assert pts.shape[1] == 2
    pts_rot = np.concatenate((pts[1:], pts[:1]))
    return np.sum(pts[:, 0] * pts_rot[:, 1] - pts[:, 1] * pts_rot[:, 0]) / 2.0


@jit(nopython=True)
def orientation(pts, eps=0):
    """Return ``+1`` (CCW), ``-1`` (CW), or ``0`` (degenerate) for a polygon.

    Numba-accelerated.  *eps* is the absolute-area threshold for degeneracy.
    """
    area = signed_area(pts)
    if abs(area) <= eps:
        return 0
    return 2 * int(area > 0) - 1


def euclidean_to_barycentric_map(tri: NDArray) -> Callable[[NDArray], NDArray]:
    """Return a function converting Euclidean 2D points to barycentric coords w.r.t. *tri*.

    *tri* is a ``(3, 2)`` array of triangle vertices.
    """
    tri = np.array(tri, dtype=np.float32)

    def inner(point):
        mat = np.repeat(tri[None, :], 3, axis=0)
        mat[np.eye(3, dtype=bool)] = point
        coords = np.array([signed_area(pts) for pts in mat], dtype=np.float32)
        return coords / np.sum(coords)

    return inner


def barycentric_to_euclidean_map(tri: NDArray) -> Callable[[NDArray], NDArray]:
    """Return a function converting barycentric coords back to 2D points w.r.t. *tri*."""

    def inner(barycentric_coords):
        return tri.T @ barycentric_coords

    return inner


def project_to_line(line: NDArray, points: NDArray) -> NDArray:
    """Orthogonally project *points* onto the infinite line through ``line[0]`` and ``line[1]``."""
    v = line[1] - line[0]
    v /= np.linalg.norm(v)
    return np.sum((points - line[0]) * v, axis=-1, keepdims=True) * v + line[0]


def line_intersection(line1: NDArray, line2: NDArray) -> NDArray:
    """Return the intersection point of two infinite lines (each given as two points).

    Raises:
        ValueError: if the lines are parallel or coincident.
    """
    diff = np.stack([line[0] - line[1] for line in (line1, line2)])

    div = np.linalg.det(diff)
    if div == 0:
        raise ValueError("lines do not intersect (parallel or coincident)")

    d = np.array([np.linalg.det(line) for line in (line1, line2)])
    return np.array([np.linalg.det(np.stack([d, dif])) for dif in diff.T]) / div
