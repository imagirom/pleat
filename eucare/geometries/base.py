"""Abstract base class and shared utilities for 2D geometry backends."""

import numpy as np
from collections import Counter
from scipy.optimize import root_scalar
from functools import partial


def root_return(func):
    """Decorator that extracts the root from a root_scalar result or raises on failure."""
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        if result.converged:
            return result.root
        else:
            raise ValueError(f'No root was found')
    return inner


class Geometry:
    """Base class for 2D geometries (Euclidean, hyperbolic, spherical)."""

    @classmethod
    def origin(cls):
        """Return the origin of the geometry."""
        raise NotImplementedError

    @classmethod
    def translation(cls, p1, p2):
        """Return a callable that translates points from p1 to p2."""
        raise NotImplementedError

    @classmethod
    def _rotate_around_origin(cls, a1):
        """Return a callable that rotates points by angle a1 around the origin."""
        raise NotImplementedError

    @classmethod
    def center_of_mass(cls, points, masses=None):
        """Compute the center of mass of point masses at the given positions."""
        raise NotImplementedError

    @classmethod
    def distance_to_origin(cls, p):
        """Compute the distance from point p to the origin."""
        raise NotImplementedError

    @classmethod
    def angle_to_axis(cls, p):
        """Compute the angle from the standard ray to the ray from the origin through p."""
        raise NotImplementedError

    @classmethod
    def point_along_axis(cls, x):
        """Return the point on the standard axis at signed distance x from the origin."""
        raise NotImplementedError

    @classmethod
    def to_euclidean(cls, pts):
        """Convert points from this geometry's representation to Euclidean 2D coordinates."""
        raise NotImplementedError

    @classmethod
    def invert(cls, p):
        """Return the point diametrically opposite to p through the origin."""
        return cls.translation(p, cls.origin())(cls.origin())

    @classmethod
    def rotation(cls, p1, a1):
        """Return a callable that rotates points by angle a1 around point p1."""
        t1 = cls.translation(p1, cls.origin())
        r = cls._rotate_around_origin(a1)
        t2 = cls.translation(cls.origin(), p1)

        def rotate(pts):
            return t2(r(t1(pts)))

        return rotate

    @classmethod
    def angle(cls, p1, p2, p3):
        """Return the angle from p1 to p3 with apex at p2."""
        p1, p3 = cls.translation(p2, cls.origin())(np.array([p1, p3]))
        a1, a3 = cls.angle_to_axis(p1), cls.angle_to_axis(p3)
        return (a1 - a3) % (2 * np.pi)

    @classmethod
    def to_polar(cls, p):
        """Compute polar coordinates (distance, angle) of point p."""
        return cls.distance_to_origin(p), cls.angle_to_axis(p)

    @classmethod
    def distance(cls, p1, p2):
        """Compute the distance between points p1 and p2."""
        return cls.distance_to_origin(cls.translation(p1, cls.origin())(p2))

    @classmethod
    def from_polar(cls, r, a):
        """Construct a point from polar coordinates (radius r, angle a)."""
        return cls.rotation(cls.origin(), a)(cls.point_along_axis(r))

    @classmethod
    def unit_vector(cls, a):
        """Return the point at unit distance from the origin at angle a."""
        return cls.from_polar(r=1, a=a)

    @classmethod
    def construct_next_poly_point(cls, a, b, angle, length):
        """Construct point c such that angle(a, b, c) = angle and dist(b, c) = length."""
        a0 = cls.translation(b, cls.origin())(a)
        c0 = cls.from_polar(length, cls.angle_to_axis(a0) - angle)
        c = cls.translation(cls.origin(), b)(c0)
        return c

    @classmethod
    def regular_poly_in_angle(cls, n, r):
        """Compute the interior angle of a regular n-gon with circumradius r."""
        return 2 * cls.angle(cls.from_polar(r, -np.pi / n), cls.from_polar(r, np.pi / n), cls.origin())

    @classmethod
    def regular_poly_side_length(cls, n, r):
        """Compute the side length of a regular n-gon with circumradius r."""
        return cls.distance(cls.from_polar(r, -np.pi / n), cls.from_polar(r, np.pi/n))

    @classmethod
    def platonic_side_length(cls, n, k):
        """Find the side length of a regular n-gon with interior angle 2*pi/k."""
        raise NotImplementedError

    @classmethod
    @root_return
    def platonic_side_length_to_radius(cls, n, l):
        """Find the circumradius of a regular n-gon with side length l."""
        return root_scalar(lambda r: cls.regular_poly_side_length(n, r) - l, x0=0.1, x1=0.01)

    @classmethod
    @root_return
    def archimedean_side_length(cls, faces_around_corner, **archimedean_side_length_root_kwargs):
        """Find the common side length for an Archimedean vertex with the given face types."""
        multiplicities = Counter(faces_around_corner)

        def length_to_angle_deficit(l):
            return sum(k * cls.regular_poly_in_angle(n, cls.platonic_side_length_to_radius(n, l))
                       for n, k in multiplicities.items()) - 2 * np.pi

        if not archimedean_side_length_root_kwargs:
            archimedean_side_length_root_kwargs = dict(x0=1, x1=2)

        return root_scalar(lambda l: np.sign(l) * length_to_angle_deficit(abs(l)),
                           **archimedean_side_length_root_kwargs)

    @classmethod
    def archimedean_side_length_and_angles(cls, faces_around_corner):
        """Return the side length and a dict of interior angles for an Archimedean vertex."""
        length = cls.archimedean_side_length(faces_around_corner)
        return length, {n: cls.regular_poly_in_angle(n, cls.platonic_side_length_to_radius(n, length))
                        for n in set(faces_around_corner)}

    @classmethod
    def barycentric_to_euclidean_map(cls, tri):
        """Return a callable mapping barycentric coordinates to points in the given triangle."""
        return lambda masses: cls.center_of_mass(tri, masses)

    # TODO: implement from triangle coordinates (also to?)
    # @classmethod
