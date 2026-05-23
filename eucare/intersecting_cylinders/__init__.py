"""Intersecting-cylinders curved origami crease patterns.

This subpackage builds curved-fold crease patterns from a tiling whose face
incircles are mutually tangential. The pattern places a small twist at every
face incenter and curved triangles whose flat sides connect adjacent incenters;
each curved triangle's apex meets at an original tiling vertex, where (with
``r = 1``) a *spike* is formed. With ``0 < r < 1`` a flat polygon (dual to the
original vertex) appears instead of the spike, and the curved triangles become
curved quadrilaterals reaching down to it.

The high-level entry point is :func:`make_intersecting_cylinders`;
cross-sections are described by :class:`Profile` (see :func:`circular_profile`);
the flat top-view projection is computed by :func:`top_view`; an interactive
3D preview is provided by :func:`show_3d`.
"""

from .mesh3d import show_3d, to_3d_mesh
from .pipeline import make_intersecting_cylinders, top_view
from .profiles import Profile, circular_profile, parabolic_profile
from .triangle_twist import convert_all_to_triangle_twists, convert_to_triangle_twist

__all__ = [
    "Profile",
    "circular_profile",
    "parabolic_profile",
    "make_intersecting_cylinders",
    "top_view",
    "to_3d_mesh",
    "show_3d",
    "convert_to_triangle_twist",
    "convert_all_to_triangle_twists",
]
