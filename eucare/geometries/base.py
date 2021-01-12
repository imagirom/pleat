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
        c0 = cls.from_polar(length, cls.angle_to_axis(a0) + angle)
        c = cls.translation(cls.origin(), b)(c0)
        return c

    # TODO: implement from triangle coordinates (also to?)
    # @classmethod
