"""Back-compatibility shim — the implementation moved.

Public API has been split between :mod:`eucare.shrink_rotate` (pipeline,
reciprocal figures, crease orientation, widgets) and
:mod:`eucare.flat_foldable` (Kawasaki/Maekawa flat-foldability tests).

This module re-exports the names that used to live here so that existing
code keeps working. The ``make_SRG`` alias has been retired — call
:func:`eucare.shrink_rotate.shrink_rotate_pattern` directly (it now performs
crease assignment by default; pass ``assign_creases=False`` for the bare
topology variant).
"""

from __future__ import annotations

# from .flat_foldable import kawasaki_sum, max_kawasaki_sum
# from .shrink_rotate import (
#     THIS_WAY,
#     assign_shrink_rotate_creases,
#     assign_this_way_by_distance,
#     assign_this_way_by_face_area,
#     assign_this_way_by_face_bfs,
#     assign_this_way_by_face_degree,
#     assign_this_way_by_face_z_order,
#     assign_this_way_by_vertex_z_order,
#     assign_this_way_from_center,
#     clear_this_way,
#     reciprocal_figure,
#     shrink_rotate_pattern,
# )
# from .utils import random_directed_set

# __all__ = [
#     "THIS_WAY",
#     "assign_shrink_rotate_creases",
#     "assign_this_way_by_distance",
#     "assign_this_way_by_face_area",
#     "assign_this_way_by_face_bfs",
#     "assign_this_way_by_face_degree",
#     "assign_this_way_by_face_z_order",
#     "assign_this_way_by_vertex_z_order",
#     "assign_this_way_from_center",
#     "clear_this_way",
#     "kawasaki_sum",
#     "max_kawasaki_sum",
#     "random_directed_set",
#     "reciprocal_figure",
#     "shrink_rotate_pattern",
# ]
