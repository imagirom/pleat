import numpy as np
from numba import jit

pi = np.pi
tau = 2 * np.pi


def unit_vector(alpha):
    return np.stack([np.cos(alpha), np.sin(alpha)], axis=-1)


def angle_to_axis(vectors):
    return np.arctan2(vectors[..., 1], vectors[..., 0])


def angle(a, b, c):
    return (angle_to_axis(a - b) - angle_to_axis(c - b)) % (2*np.pi)


def in_angles(points):
    edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    edge_angles = angle_to_axis(edge_vectors)
    return (np.pi + edge_angles - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2*np.pi)


def edge_lengths(points):
    edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    return np.linalg.norm(edge_vectors, axis=1)


def edge_lengths_and_in_angles(points, geometry):
    edge_lengths = [geometry.distance(p1, p2) for p1, p2 in zip(points, np.concatenate([points[1:], points[:1]]))]
    in_angles = [geometry.angle(p1, p2, p3)
                 for p1, p2, p3 in zip(points,
                                       np.concatenate([points[1:], points[:1]]),
                                       np.concatenate([points[2:], points[:2]]))]

    # edge_vectors = np.concatenate([points[1:], points[:1]]) - points
    # edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    # edge_angles = angle_to_axis(edge_vectors)
    # in_angles = (np.pi + edge_angles - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2*np.pi)
    return edge_lengths, in_angles


def unit_vector_to_vector(alpha, vector):
    return np.array([vector[0], vector[0] + unit_vector(angle_to_axis(vector[1]-vector[0]) + alpha)])


def tri_grid_point(i, j):
    return unit_vector(0) * i + unit_vector(np.pi / 3) * j


def regular_poly_points(n):
    return unit_vector(np.linspace(0, tau, n, endpoint=False)) / (2 * np.sin(pi/n))


def apply_affine(vectors, matrix):
    return np.dot(
        np.concatenate([vectors, np.ones_like(vectors[..., :1])], axis=-1),
        matrix
    )


def rotation_matrix(alpha):
    s, c = np.sin(alpha), np.cos(alpha)
    return np.array([[c, s], [-s, c]])


def find_affine(line0, line1):
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
    if len(data.shape) > len(query.shape):
        query = query[None]
    index = np.argmin(np.linalg.norm(data - query, axis=-1))
    return data[index], index if return_index else data[index]


@jit(nopython=True)
def signed_area(pts):
    # pts = np.array(pts) # delete for jit
    assert pts.shape[1] == 2
    pts_rot = np.concatenate((pts[1:], pts[:1]))
    return np.sum(pts[:, 0] * pts_rot[:, 1] - pts[:, 1] * pts_rot[:, 0]) / 2.0


@jit(nopython=True)
def orientation(pts, eps=0):
    area = signed_area(pts)
    if abs(area) <= eps:
        return 0
    return 2*int(area > 0)-1


def euclidean_to_barycentric_map(tri):
    tri = np.array(tri, dtype=np.float32)
    def inner(point):
        mat = np.repeat(tri[None, :], 3, axis=0)
        mat[np.eye(3, dtype=np.bool)] = point
        coords = np.array([signed_area(pts) for pts in mat], dtype=np.float32)
        return coords / np.sum(coords)
    return inner


def barycentric_to_euclidean_map(tri):
    def inner(barycentric_coords):
        return tri.T @ barycentric_coords
    return inner


def project_to_line(line, points):
    v = line[1] - line[0]
    v /= np.linalg.norm(v)
    return np.sum((points - line[0]) * v, axis=-1, keepdims=True) * v + line[0]


def line_intersection(line1, line2):
    diff = np.stack([l[0] - l[1] for l in (line1, line2)])

    div = np.linalg.det(diff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = np.array([np.linalg.det(l) for l in (line1, line2)])
    return np.array([np.linalg.det(np.stack([d, dif])) for dif in diff.T]) / div
