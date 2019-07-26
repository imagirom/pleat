import numpy as np

pi = np.pi
tau = 2 * np.pi


def unit_vector(alpha):
    return np.stack([np.cos(alpha), np.sin(alpha)], axis=-1)


def angle_to_axis(vectors):
    return np.arctan2(vectors[..., 1], vectors[..., 0])


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
