import numpy as np
from .base import Geometry


class EuclideanGeometry(Geometry):
    @classmethod
    def origin(cls):
        return np.array([0, 0])

    @classmethod
    def translation(cls, p1, p2):
        def translate(p):
            return p + p2 - p1

        return translate

    @classmethod
    def _rotate_around_origin(cls, a1):
        rot_mat = np.array([[np.cos(a1), np.sin(a1)], [-np.sin(a1), np.cos(a1)]])

        def origin_rotate(p):
            return p @ rot_mat

        return origin_rotate

    @classmethod
    def center_of_mass(cls, points, masses=None):
        assert len(points.shape) == 2 and points.shape[-1] == 2, f'{points.shape}'
        if masses is not None:
            masses = masses / np.sum(masses) * len(points)
            points = points * masses[..., None]
        return np.mean(points, axis=0)

    @classmethod
    def distance_to_origin(cls, p):
        return np.linalg.norm(p)

    @classmethod
    def angle_to_axis(cls, p):
        return np.arctan2(p[..., 1], p[..., 0])

    @classmethod
    def point_along_axis(cls, x):
        return np.array([x, 0])

    @classmethod
    def to_euclidean(cls, pts):
        return pts

    @classmethod
    def archimedian_side_length(cls, faces_around_corner, eps=1e-6):
        euclidean_vertex_angle = sum(np.pi * (n - 2) / n for n in faces_around_corner)
        if abs(euclidean_vertex_angle - 2 * np.pi) < eps:
            return 1
        else:
            raise ValueError(f'Vertex {faces_around_corner} is impossible in euclidean geometry.')
