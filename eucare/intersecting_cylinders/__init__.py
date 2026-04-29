"""Intersecting-cylinders curved origami crease patterns.

This subpackage builds curved-fold crease patterns from a tiling whose faces have
mutually-tangential incenters. The high-level entry point is
:func:`make_intersecting_cylinders`; cross-sections are described by
:class:`Profile` (see :func:`circular_profile`, :func:`spherical_profile`).
"""

from .pipeline import make_intersecting_cylinders
from .profiles import Profile, circular_profile, spherical_profile
from .triangle_twist import convert_all_to_triangle_twists, convert_to_triangle_twist

__all__ = [
    "Profile",
    "circular_profile",
    "spherical_profile",
    "make_intersecting_cylinders",
    "convert_to_triangle_twist",
    "convert_all_to_triangle_twists",
]
