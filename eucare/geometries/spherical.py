import numpy as np
from .base import Geometry


def _rot_x_mat(a1):
    return np.array([
        [1, 0, 0],
        [0, np.cos(a1), -np.sin(a1)],
        [0, np.sin(a1), np.cos(a1)]
    ])


def _rot_z_mat(a1):
    return np.array([
        [np.cos(a1), np.sin(a1), 0, ],
        [-np.sin(a1), np.cos(a1), 0],
        [0, 0, 1]
    ])


class SphereModel(Geometry):
    @classmethod
    def origin(cls):
        return np.array([1, 0, 0])

    @classmethod
    def translation(cls, p1, p2):
        def origin_translation_mat(p1):
            a1 = np.arctan2(p1[2], p1[1])
            m1 = _rot_x_mat(-a1)
            m2 = _rot_z_mat(-np.arccos(p1[0]))
            m3 = _rot_x_mat(a1)
            return m3 @ m2 @ m1

        def minus(p):
            return np.array([p[0], *-p[1:]])

        m1 = origin_translation_mat(minus(p1))
        m2 = origin_translation_mat(minus(m1 @ p2))
        m3 = origin_translation_mat(p1)

        mat = m3 @ m2 @ m1

        def translate(p):
            return p @ mat

        return translate

    @classmethod
    def _rotate_around_origin(cls, a1):
        mat = _rot_x_mat(a1).T

        def origin_rotate(p):
            return p @ mat

        return origin_rotate

    @classmethod
    def center_of_mass(cls, points, masses=None):
        if masses is not None:
            masses = masses / np.sum(masses) * len(points)
            points = points * masses[..., None]
        result = np.mean(points, axis=0)
        result /= np.linalg.norm(result)
        return result

    @classmethod
    def distance_to_origin(cls, p):
        return np.arccos(np.clip(p[0], -1, 1))

    @classmethod
    def angle_to_axis(cls, p):
        return np.arctan2(p[2], p[1])

    @classmethod
    def point_along_axis(cls, x):
        return np.array([np.cos(x), np.sin(x), 0])

    @classmethod
    def archimedean_side_length(cls, faces_around_corner):
        return super().archimedean_side_length(faces_around_corner, bracket=[0.1, 2*np.pi/max(faces_around_corner)])

    # --- Methods Specific to this geometry ---

    @classmethod
    def stereographic_projection(cls, pts):
        """stereographic projection with pole at (-1, 0, 0)"""
        return 2 * pts[..., 1:] / (pts[..., :1] + 1)

    @classmethod
    def to_euclidean(cls, pts):
        return cls.stereographic_projection(pts)
