"""Intersecting-cylinders curved origami crease patterns.

This subpackage builds curved-fold crease patterns from a tiling whose faces have
mutually-tangential incenters. The high-level entry point is
:func:`make_intersecting_cylinders`; cross-sections are described by
:class:`Profile` (see :func:`circular_profile`); the flat top-view projection
is computed by :func:`top_view`; an interactive 3D preview is provided by
:func:`show_3d`.
"""

from .mesh3d import show_3d, to_3d_mesh
from .pipeline import make_intersecting_cylinders, top_view
from .profiles import Profile, circular_profile
from .triangle_twist import convert_all_to_triangle_twists, convert_to_triangle_twist

__all__ = [
    "Profile",
    "circular_profile",
    "make_intersecting_cylinders",
    "top_view",
    "to_3d_mesh",
    "show_3d",
    "convert_to_triangle_twist",
    "convert_all_to_triangle_twists",
]
