"""2D geometry primitives: vectors, angles, areas, intersections, and coordinate transforms."""
import numpy as np
from numba import jit

pi = np.pi
tau = 2 * np.pi


def unit_vector(alpha):
    """Return unit vector(s) at angle *alpha* (radians). Accepts scalar or array."""
    return np.stack([np.cos(alpha), np.sin(alpha)], axis=-1)


def angle_to_axis(vectors):
    """Return the angle (radians) each 2D vector makes with the positive x-axis."""
    return np.arctan2(vectors[..., 1], vectors[..., 0])


def angle(a, b, c):
    """Return the interior angle at vertex *b* in triangle a-b-c, in [0, 2pi)."""
    return (angle_to_axis(a - b) - angle_to_axis(c - b)) % (2*np.pi)


def in_angles(points):
    """Return interior angles of a polygon given its ordered vertices (shape (n, 2))."""
    edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    edge_angles = angle_to_axis(edge_vectors)
    return (np.pi + edge_angles - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2*np.pi)


def edge_lengths(points):
    """Return edge lengths of a polygon given its ordered vertices (shape (n, 2))."""
    edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    return np.linalg.norm(edge_vectors, axis=1)


def edge_lengths_and_in_angles(points, geometry):
    """Return (edge_lengths, interior_angles) using a pluggable *geometry* backend."""
    edge_lengths = [geometry.distance(p1, p2) for p1, p2 in zip(points, np.concatenate([points[1:], points[:1]]))]
    in_angles = [geometry.angle(p1, p2, p3)
                 for p1, p2, p3 in zip(points,
                                       np.concatenate([points[1:], points[:1]]),
                                       np.concatenate([points[2:], points[:2]]))]
    return edge_lengths, in_angles


def unit_vector_to_vector(alpha, vector):
    """Rotate direction *alpha* onto the line defined by *vector* (pair of 2D points)."""
    return np.array([vector[0], vector[0] + unit_vector(angle_to_axis(vector[1]-vector[0]) + alpha)])


def tri_grid_point(i, j):
    """Return a point on the standard triangular grid at integer coordinates (i, j)."""
    return unit_vector(0) * i + unit_vector(np.pi / 3) * j


def regular_poly_points(n):
    """Return vertices of a unit-edge-length regular *n*-gon centered at the origin."""
    return unit_vector(np.linspace(0, tau, n, endpoint=False)) / (2 * np.sin(pi/n))


def apply_affine(vectors, matrix):
    """Apply a 2D affine transform (3x2 matrix) to an array of 2D points."""
    return np.dot(
        np.concatenate([vectors, np.ones_like(vectors[..., :1])], axis=-1),
        matrix
    )


def rotation_matrix(alpha):
    """Return a 2x2 rotation matrix. Use as ``point @ R`` to rotate by *alpha* CCW."""
    s, c = np.sin(alpha), np.cos(alpha)
    return np.array([[c, s], [-s, c]])


def find_affine(line0, line1):
    """Find the affine transform (3x2 matrix) mapping directed segment *line0* to *line1*."""
    lines = np.stack([line0, line1])
    relative = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(relative, axis=1)
    scale = lengths[1] / lengths[0]
    angles = angle_to_axis(relative)
    angle = angles[1] - angles[0]
    linear = scale * rotation_matrix(angle)
    offset = lines[1, 1] - lines[0, 1].dot(linear)
    return np.concatenate([
        linear,
        offset[None]
    ])


def nearest_neighbor(data, query, return_index=True):
    """Return the nearest point in *data* to *query* (brute-force)."""
    if len(data.shape) > len(query.shape):
        query = query[None]
    index = np.argmin(np.linalg.norm(data - query, axis=-1))
    return data[index], index if return_index else data[index]


@jit(nopython=True)
def signed_area(pts):
    """Signed area of a polygon (positive = CCW). Numba-accelerated."""
    assert pts.shape[1] == 2
    pts_rot = np.concatenate((pts[1:], pts[:1]))
    return np.sum(pts[:, 0] * pts_rot[:, 1] - pts[:, 1] * pts_rot[:, 0]) / 2.0


@jit(nopython=True)
def orientation(pts, eps=0):
    """Return +1 (CCW), -1 (CW), or 0 (degenerate) for a polygon. Numba-accelerated."""
    area = signed_area(pts)
    if abs(area) <= eps:
        return 0
    return 2*int(area > 0)-1


def euclidean_to_barycentric_map(tri):
    """Return a function that converts Euclidean 2D points to barycentric coordinates w.r.t. *tri*."""
    tri = np.array(tri, dtype=np.float32)
    def inner(point):
        mat = np.repeat(tri[None, :], 3, axis=0)
        mat[np.eye(3, dtype=bool)] = point
        coords = np.array([signed_area(pts) for pts in mat], dtype=np.float32)
        return coords / np.sum(coords)
    return inner


def barycentric_to_euclidean_map(tri):
    """Return a function that converts barycentric coordinates to Euclidean 2D points w.r.t. *tri*."""
    def inner(barycentric_coords):
        return tri.T @ barycentric_coords
    return inner


def project_to_line(line, points):
    """Orthogonally project *points* onto the infinite line through *line[0]* and *line[1]*."""
    v = line[1] - line[0]
    v /= np.linalg.norm(v)
    return np.sum((points - line[0]) * v, axis=-1, keepdims=True) * v + line[0]


def line_intersection(line1, line2):
    """Return the intersection point of two infinite lines (each given as two points).

    Raises:
        ValueError: If lines are parallel or coincident.
    """
    diff = np.stack([l[0] - l[1] for l in (line1, line2)])

    div = np.linalg.det(diff)
    if div == 0:
        raise ValueError('lines do not intersect (parallel or coincident)')

    d = np.array([np.linalg.det(l) for l in (line1, line2)])
    return np.array([np.linalg.det(np.stack([d, dif])) for dif in diff.T]) / div
