"""Shrink-rotate origami crease patterns.

Pipeline overview
-----------------

A shrink-rotate tessellation is built from an input tiling ``G`` in three
steps:

1. **Reciprocal figure** — :func:`reciprocal_figure` solves a linear system
   to place a dual point on each face of ``G`` such that the dual edges are
   perpendicular to the primal edges. These dual points serve as the
   rotation centers in step 3.
2. **Topological subdivision** — the Conway operator
   :func:`eucare.conway.shrink_rotate_graph` splits each face into a smaller
   central polygon plus a ring of quadrilateral *twist* faces. Twist faces
   carry a ``'shrink_rotate'`` attribute.
3. **Geometric shrink + rotate** — each central polygon is rotated by an
   angle ``alpha`` and scaled by a factor ``factor`` around its
   reciprocal-figure center. With the right parameter pair this is
   guaranteed to be flat-foldable.

The high-level entry point is :func:`shrink_rotate_pattern`. Crease
mountain/valley assignment is done by :func:`assign_shrink_rotate_creases`,
driven by the orientation marks set up via the helpers in
:mod:`eucare.shrink_rotate.crease_orientation`. An interactive widget for
exploring the parameter space is :class:`ShrinkRotateExplorer`.
"""

from __future__ import annotations

from .crease_orientation import (
    THIS_WAY,
    assign_this_way_by_distance,
    assign_this_way_by_face_area,
    assign_this_way_by_face_bfs,
    assign_this_way_by_face_degree,
    assign_this_way_by_face_z_order,
    assign_this_way_by_vertex_z_order,
    assign_this_way_from_center,
    clear_this_way,
)
from .pipeline import (
    assign_shrink_rotate_creases,
    shrink_rotate_pattern,
)
from .reciprocal_figures import reciprocal_figure

__all__ = [
    "THIS_WAY",
    "assign_shrink_rotate_creases",
    "assign_this_way_by_distance",
    "assign_this_way_by_face_area",
    "assign_this_way_by_face_bfs",
    "assign_this_way_by_face_degree",
    "assign_this_way_by_face_z_order",
    "assign_this_way_by_vertex_z_order",
    "assign_this_way_from_center",
    "clear_this_way",
    "reciprocal_figure",
    "shrink_rotate_pattern",
]


def __getattr__(name: str):
    """Lazily expose :class:`ShrinkRotateExplorer` (avoids importing matplotlib eagerly)."""
    if name == "ShrinkRotateExplorer":
        from .widgets import ShrinkRotateExplorer

        return ShrinkRotateExplorer
    raise AttributeError(f"module 'eucare.shrink_rotate' has no attribute {name!r}")
