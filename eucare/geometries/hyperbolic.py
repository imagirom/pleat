"""Hyperbolic geometry via the Poincare disk model."""

import numpy as np
from .base import Geometry


def apply_mobius(mat, points):
    """Apply a Mobius transformation given by a 2x2 matrix to complex-valued points."""
    return (mat[0, 0] * points + mat[0, 1]) / (mat[1, 0] * points + mat[1, 1])


class MobiusTransform():
    """A Mobius transformation represented as a 2x2 complex matrix."""

    def __init__(self, mat):
        if not isinstance(mat, np.ndarray):
            mat = np.array(mat)
        assert mat.shape == (2, 2), f'{mat.shape}'
        self.mat = mat

    def __call__(self, points):
        return apply_mobius(self.mat, points)

    def __matmul__(self, other):
        assert isinstance(other, MobiusTransform), f'{type(other)}'
        return MobiusTransform(self.mat @ other.mat)

    def __pow__(self, exponent):
        return MobiusTransform(np.linalg.matrix_power(self.mat, exponent))

    def __repr__(self):
        return f'MobiusTransform({self.mat.tolist()})'


# TODO: maybe have another class for the hyperboloid model

def complex_to_real(z):
    """Convert complex numbers to real 2D coordinate arrays."""
    return np.stack([z.real, z.imag], axis=-1)


def real_to_complex(x):
    """Convert real 2D coordinate arrays to complex numbers."""
    assert x.shape[-1] == 2
    return x[..., 0] + 1j * x[..., 1]


def poincare_to_hyperboloid(z):
    """Map Poincare disk coordinates to the hyperboloid model."""
    pts = complex_to_real(z)
    squared_norm = (pts ** 2).sum(-1, keepdims=True)
    return np.concatenate([(1 + squared_norm), 2 * pts], axis=-1) / (1 - squared_norm)


def hyperboloid_to_poincare(v):
    """Map hyperboloid model coordinates back to the Poincare disk."""
    return real_to_complex(v[..., 1:] / (1 + v[..., :1]))


def hyperboloid_centroid(vs, ms=None, axis=None):
    """Compute the centroid on the hyperboloid model, optionally weighted by masses."""
    if axis is None:
        assert len(vs.shape) == 2
        axis = 0
    ms = np.ones(list(vs.shape[:-1]), dtype=vs.dtype) if ms is None else ms
    mean = (vs * ms[..., None]).mean(axis)
    mean /= np.sqrt(mean[..., :1] ** 2 - (mean[..., 1:] ** 2).sum(-1, keepdims=True))
    return mean


def poincare_centroid(zs, ms=None, axis=None):
    """Compute the centroid of points in the Poincare disk via the hyperboloid model."""
    return hyperboloid_to_poincare(hyperboloid_centroid(poincare_to_hyperboloid(zs), ms, axis))


class PoincareDiskModel(Geometry):
    """Hyperbolic geometry using the Poincare disk model with complex coordinates."""

    @classmethod
    def origin(cls):
        return 0 + 0j

    @classmethod
    def translation(cls, p1, p2):
        if p2 == 0:
            p1, p2 = 0, -p1
        if p1 == 0:
            return MobiusTransform([[1, p2], [p2.conjugate(), 1]])
        m1 = cls.translation(p2, 0)
        m2 = cls.translation(0, -m1(p1))
        m3 = cls.translation(0, p2)
        return m3 @ m2 @ m1

    @classmethod
    def rotation(cls, p1, a1):
        if p1 == 0:
            return MobiusTransform([[np.exp(1j * a1), 0], [0, 1]])
        return cls.translation(p1, 0) @ cls.rotation(0, a1) @ cls.translation(0, p1)

    @classmethod
    def center_of_mass(cls, points, masses=None):
        return poincare_centroid(points, masses)

    @classmethod
    def distance_to_origin(cls, p):
        return 2 * np.arctanh(np.linalg.norm(p))

    @classmethod
    def angle_to_axis(cls, p):
        return np.arctan2(p.imag, p.real)

    @classmethod
    def point_along_axis(cls, x):
        return np.sign(x) * np.tanh(np.abs(x) / 2)

    @classmethod
    def to_euclidean(cls, pts):
        return complex_to_real(pts)
