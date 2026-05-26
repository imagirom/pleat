"""Abstract base class and shared utilities for 2D geometry backends.

A :class:`Geometry` exposes the operations that the rest of eucare needs to
lay out tiles in arbitrary curvature: ``translation``, rotation, distance,
angle, and centre of mass.  Each backend (Euclidean, hyperbolic, spherical)
implements a small set of primitives; the rest are derived in this base
class.

A point in the Euclidean and hyperbolic backends is a 2D position; in the
spherical backend it is a 3D unit vector.  Use :meth:`Geometry.to_euclidean`
to obtain a 2D image suitable for plotting.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

Point = Any
Transform = Callable[[Point], Point]


def root_return(func: Callable[..., Any]) -> Callable[..., float]:
    """Decorator that extracts the root from a root_scalar result or raises on failure."""

    def inner(*args: Any, **kwargs: Any) -> float:
        result = func(*args, **kwargs)
        if result.converged:
            return float(result.root)
        else:
            raise ValueError("No root was found")

    return inner


class Geometry:
    """Base class for 2D geometries (Euclidean, hyperbolic, spherical)."""

    @classmethod
    def origin(cls) -> Point:
        """Return the origin of the geometry."""
        raise NotImplementedError

    @classmethod
    def translation(cls, p1: Point, p2: Point) -> Transform:
        """Return a callable that translates points from p1 to p2."""
        raise NotImplementedError

    @classmethod
    def _rotate_around_origin(cls, a1: float) -> Transform:
        """Return a callable that rotates points by angle a1 around the origin."""
        raise NotImplementedError

    @classmethod
    def center_of_mass(cls, points: NDArray[Any], masses: NDArray[Any] | None = None) -> Point:
        """Compute the center of mass of point masses at the given positions."""
        raise NotImplementedError

    @classmethod
    def distance_to_origin(cls, p: Point) -> float:
        """Compute the distance from point p to the origin."""
        raise NotImplementedError

    @classmethod
    def angle_to_axis(cls, p: Point) -> float:
        """Compute the angle from the standard ray to the ray from the origin through p."""
        raise NotImplementedError

    @classmethod
    def point_along_axis(cls, x: float) -> Point:
        """Return the point on the standard axis at signed distance x from the origin."""
        raise NotImplementedError

    @classmethod
    def to_euclidean(cls, pts: NDArray[Any]) -> NDArray[Any]:
        """Convert points from this geometry's representation to Euclidean 2D coordinates."""
        raise NotImplementedError

    @classmethod
    def invert(cls, p: Point) -> Point:
        """Return the point diametrically opposite to p through the origin."""
        return cls.translation(p, cls.origin())(cls.origin())

    @classmethod
    def rotation(cls, p1: Point, a1: float) -> Transform:
        """Return a callable that rotates points by angle a1 around point p1."""
        t1 = cls.translation(p1, cls.origin())
        r = cls._rotate_around_origin(a1)
        t2 = cls.translation(cls.origin(), p1)

        def rotate(pts: Point) -> Point:
            return t2(r(t1(pts)))

        return rotate

    @classmethod
    def angle(cls, p1: Point, p2: Point, p3: Point) -> float:
        """Return the angle from p1 to p3 with apex at p2."""
        p1, p3 = cls.translation(p2, cls.origin())(np.array([p1, p3]))
        a1, a3 = cls.angle_to_axis(p1), cls.angle_to_axis(p3)
        return float((a1 - a3) % (2 * np.pi))

    @classmethod
    def to_polar(cls, p: Point) -> tuple[float, float]:
        """Compute polar coordinates (distance, angle) of point p."""
        return cls.distance_to_origin(p), cls.angle_to_axis(p)

    @classmethod
    def distance(cls, p1: Point, p2: Point) -> float:
        """Compute the distance between points p1 and p2."""
        return cls.distance_to_origin(cls.translation(p1, cls.origin())(p2))

    @classmethod
    def from_polar(cls, r: float, a: float) -> Point:
        """Construct a point from polar coordinates (radius r, angle a)."""
        return cls.rotation(cls.origin(), a)(cls.point_along_axis(r))

    @classmethod
    def unit_vector(cls, a: float) -> Point:
        """Return the point at unit distance from the origin at angle a."""
        return cls.from_polar(r=1, a=a)

    @classmethod
    def construct_next_poly_point(cls, a: Point, b: Point, angle: float, length: float) -> Point:
        """Construct point c such that angle(a, b, c) = angle and dist(b, c) = length."""
        a0 = cls.translation(b, cls.origin())(a)
        c0 = cls.from_polar(length, cls.angle_to_axis(a0) - angle)
        c = cls.translation(cls.origin(), b)(c0)
        return c

    @classmethod
    def regular_poly_in_angle(cls, n: int, r: float) -> float:
        """Compute the interior angle of a regular n-gon with circumradius r."""
        return 2 * cls.angle(cls.from_polar(r, -np.pi / n), cls.from_polar(r, np.pi / n), cls.origin())

    @classmethod
    def regular_poly_side_length(cls, n: int, r: float) -> float:
        """Compute the side length of a regular n-gon with circumradius r."""
        return cls.distance(cls.from_polar(r, -np.pi / n), cls.from_polar(r, np.pi / n))

    @classmethod
    def platonic_side_length(cls, n: int, k: int) -> float:
        """Find the side length of a regular n-gon with interior angle 2*pi/k."""
        raise NotImplementedError

    @classmethod
    @root_return
    def platonic_side_length_to_radius(cls, n: int, l: float) -> Any:
        """Find the circumradius of a regular n-gon with side length l."""
        return root_scalar(lambda r: cls.regular_poly_side_length(n, r) - l, x0=0.1, x1=0.01)

    @classmethod
    @root_return
    def archimedean_side_length(
        cls, faces_around_corner: Iterable[int], **archimedean_side_length_root_kwargs: Any
    ) -> Any:
        """Find the common side length for an Archimedean vertex with the given face types."""
        multiplicities = Counter(faces_around_corner)

        def length_to_angle_deficit(l: float) -> float:
            return (
                sum(
                    k * cls.regular_poly_in_angle(n, cls.platonic_side_length_to_radius(n, l))
                    for n, k in multiplicities.items()
                )
                - 2 * np.pi
            )

        if not archimedean_side_length_root_kwargs:
            archimedean_side_length_root_kwargs = dict(x0=1, x1=2)

        return root_scalar(
            lambda l: np.sign(l) * length_to_angle_deficit(abs(l)), **archimedean_side_length_root_kwargs
        )

    @classmethod
    def archimedean_side_length_and_angles(cls, faces_around_corner: Iterable[int]) -> tuple[float, dict[int, float]]:
        """Return the side length and a dict of interior angles for an Archimedean vertex."""
        length = cls.archimedean_side_length(faces_around_corner)
        return length, {
            n: cls.regular_poly_in_angle(n, cls.platonic_side_length_to_radius(n, length))
            for n in set(faces_around_corner)
        }

    @classmethod
    def barycentric_to_euclidean_map(cls, tri: NDArray[Any]) -> Callable[[NDArray[Any]], Point]:
        """Return a callable mapping barycentric coordinates to points in the given triangle."""
        return lambda masses: cls.center_of_mass(tri, masses)

    # TODO: implement from triangle coordinates (also to?)
    # @classmethod
