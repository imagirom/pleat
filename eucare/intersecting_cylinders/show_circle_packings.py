"""Visualize the two dual circle packings underlying an intersecting-cylinders tiling.

An :mod:`eucare.intersecting_cylinders` crease pattern is built from a tiling
whose face incircles are mutually tangential. Together with the *vertex
circles* (centred at each original vertex and passing through the tangent
points on the incident edges) this gives two interleaved circle packings whose
circles touch pairwise at the edge tangent points.

:func:`show_dual_circle_packings` builds the ortho-Conway graph of the input
tiling, repositions every edge-midpoint vertex to the actual tangent point,
and styles the result so the two circle families are rendered as filled red
(face incircles) and blue (vertex circles) discs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import half
from .mesh3d import _build_ortho_with_tangent_points

if TYPE_CHECKING:
    from ..half import EuclideanPositionHEG


_FACE_COLOR = (1.0, 0.0, 0.0, 0.3)
_VERTEX_COLOR = (0.0, 0.0, 1.0, 0.3)
_TANGENT_COLOR = (0.0, 0.0, 0.0)
_TANGENT_RADIUS = 0.01
_VERTEX_CIRCLE_LINE_WIDTH = 0.005


def build_dual_circle_packings(G: "EuclideanPositionHEG") -> "EuclideanPositionHEG":
    """Return a styled ortho-graph representing the two dual circle packings.

    The returned graph is a fresh copy: the input ``G`` is not modified. Every
    vertex carries a ``color_key`` and ``vertex_radius`` attribute; circles
    coming from original faces use red fill, circles coming from original
    vertices use blue fill, and the tangent-point vertices are drawn as small
    black dots. Halfedges are coloured black and have ``line_width`` set so
    that only the vertex circles are outlined.

    Args:
        G: A 2D Euclidean tiling whose faces have well-defined incenters such
            that the corresponding incircles are mutually tangential at every
            edge.

    Returns:
        A copy of the ortho-Conway graph of ``G`` with cosmetic attributes
        set, ready to pass to :meth:`HalfEdgeGraph.show`.
    """
    G_ortho = _build_ortho_with_tangent_points(G)
    G_ortho.normalize_positions()

    for v in G_ortho.vertices.union(G_ortho.faces):
        pre = v.get("pre_conway")
        if isinstance(pre, half.Face) and isinstance(v, half.Vertex):
            v["color_key"] = _FACE_COLOR
            v["vertex_radius"] = float(np.linalg.norm(v["pos"] - v.any_outgoing.dest["pos"]))
            for h in v.outgoing_iter():
                h["line_width"] = 0.0
                h.rev["line_width"] = 0.0
        elif isinstance(pre, half.Vertex) and isinstance(v, half.Vertex):
            v["color_key"] = _VERTEX_COLOR
            v["vertex_radius"] = float(np.linalg.norm(v["pos"] - v.any_outgoing.dest["pos"]))
            for h in v.outgoing_iter():
                h["line_width"] = _VERTEX_CIRCLE_LINE_WIDTH
                h.rev["line_width"] = _VERTEX_CIRCLE_LINE_WIDTH
        else:
            v["vertex_radius"] = _TANGENT_RADIUS
            v["color_key"] = _TANGENT_COLOR

    for h in G_ortho.halfedges:
        h["color_key"] = _TANGENT_COLOR

    return G_ortho


def show_dual_circle_packings(G: "EuclideanPositionHEG", **show_kwargs: Any) -> Any:
    """Render the two dual circle packings of a tiling.

    This is a thin convenience wrapper around :func:`build_dual_circle_packings`
    that calls :meth:`HalfEdgeGraph.show` on the resulting styled graph with
    ``render_faces=False``.

    Args:
        G: A 2D Euclidean tiling -- see :func:`build_dual_circle_packings` for
            the requirements on ``G``.
        **show_kwargs: Forwarded to :meth:`HalfEdgeGraph.show`. ``render_faces``
            defaults to ``False`` but can be overridden.

    Returns:
        Whatever :meth:`HalfEdgeGraph.show` returns.
    """
    show_kwargs.setdefault("render_faces", False)
    G_ortho = build_dual_circle_packings(G)
    return G_ortho.show(**show_kwargs)
