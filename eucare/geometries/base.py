import numpy as np
from collections import Counter
from scipy.optimize import root_scalar
from functools import partial


def root_return(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        if result.converged:
            return result.root
        else:
            raise ValueError(f'No root was found')
    return inner


class Geometry:
    """base class for various 2D geometries"""

    @classmethod
    def origin(cls):
        """return the origin of the geometry"""
        raise NotImplementedError

    @classmethod
    def translation(cls, p1, p2):
        """
        should return a translation (acting on points) from p1 to p2,
        or, if p2 is None one from the origin to p1.
        """
        raise NotImplementedError

    @classmethod
    def _rotate_around_origin(cls, a1):
        """
        should return a rotation of angle a around the origin.
        no need to implement this if rotation is directly implemented for all points
        """
        raise NotImplementedError

    @classmethod
    def center_of_mass(cls, points, masses=None):
        """compute the center of mass of point masses at specified positions and with specified masses"""
        raise NotImplementedError

    @classmethod
    def distance_to_origin(cls, p):
        """compute the distance between point p and the origin"""
        raise NotImplementedError

    @classmethod
    def angle_to_axis(cls, p):
        """compute the angle between the standard ray and the ray from the origin through point p"""
        raise NotImplementedError

    @classmethod
    def point_along_axis(cls, x):
        """return a point along the standard axis, parametrized by x, signed distance to the origin"""
        raise NotImplementedError

    @classmethod
    def to_euclidean(cls, pts):
        raise NotImplementedError

    @classmethod
    def invert(cls, p):
        return cls.translation(p, cls.origin())(cls.origin())

    @classmethod
    def rotation(cls, p1, a1):
        # should return a rotation of angle a around p1
        t1 = cls.translation(p1, cls.origin())
        r = cls._rotate_around_origin(a1)
        t2 = cls.translation(cls.origin(), p1)

        def rotate(pts):
            return t2(r(t1(pts)))

        return rotate

    @classmethod
    def angle(cls, p1, p2, p3):
        # should return the angle between p1 and p3 with apex p2
        p1, p3 = cls.translation(p2, cls.origin())(np.array([p1, p3]))
        a1, a3 = cls.angle_to_axis(p1), cls.angle_to_axis(p3)
        return (a1 - a3) % (2 * np.pi)

    @classmethod
    def to_polar(cls, p):
        # compute polar coordinates (r, a) of point p
        return cls.distance_to_origin(p), cls.angle_to_axis(p)

    @classmethod
    def distance(cls, p1, p2):
        return cls.distance_to_origin(cls.translation(p1, cls.origin())(p2))

    @classmethod
    def from_polar(cls, r, a):
        # construct a point from polar coordinates radius=r and angle=a
        return cls.rotation(cls.origin(), a)(cls.point_along_axis(r))

    @classmethod
    def unit_vector(cls, a):
        return cls.from_polar(r=1, a=a)

    @classmethod
    def construct_next_poly_point(cls, a, b, angle, length):
        """construct the point c such that angle(a, b, c)=angle and |bc|=length"""
        a0 = cls.translation(b, cls.origin())(a)
        c0 = cls.from_polar(length, cls.angle_to_axis(a0) - angle)
        c = cls.translation(cls.origin(), b)(c0)
        return c

    @classmethod
    def regular_poly_in_angle(cls, n, r):
        return 2 * cls.angle(cls.from_polar(r, -np.pi / n), cls.from_polar(r, np.pi / n), cls.origin())

    @classmethod
    def regular_poly_side_length(cls, n, r):
        return cls.distance(cls.from_polar(r, -np.pi / n), cls.from_polar(r, np.pi/n))

    @classmethod
    def platonic_side_length(cls, n, k):
        """find the sidelength of a regular polygon with n sides and an in-angle of 2*pi/k"""
        raise NotImplementedError

    @classmethod
    @root_return
    def platonic_side_length_to_radius(cls, n, l):
        """find the radius of a regular n-gon of side length l"""
        return root_scalar(lambda r: cls.regular_poly_side_length(n, r) - l, x0=0.1, x1=0.01)

    @classmethod
    @root_return
    def archimedian_side_length(cls, faces_around_corner, **archimedian_side_length_root_kwargs):
        multiplicities = Counter(faces_around_corner)

        def length_to_angle_deficit(l):
            return sum(k * cls.regular_poly_in_angle(n, cls.platonic_side_length_to_radius(n, l))
                       for n, k in multiplicities.items()) - 2 * np.pi

        if not archimedian_side_length_root_kwargs:
            archimedian_side_length_root_kwargs = dict(x0=1, x1=2)

        return root_scalar(lambda l: np.sign(l) * length_to_angle_deficit(abs(l)),
                           **archimedian_side_length_root_kwargs)

    @classmethod
    def archimedian_side_length_and_angles(cls, faces_around_corner):
        length = cls.archimedian_side_length(faces_around_corner)
        return length, {n: cls.regular_poly_in_angle(n, cls.platonic_side_length_to_radius(n, length))
                        for n in set(faces_around_corner)}

    # TODO: implement from triangle coordinates (also to?)
    # @classmethod
