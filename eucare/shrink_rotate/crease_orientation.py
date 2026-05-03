"""Halfedge orientation marks for crease pattern stacking order.

Many origami pipelines need to know, for each interior edge, which of the
two adjacent faces is "above" the other in the folded model. We encode
this via a single halfedge attribute, the **THIS_WAY** mark:

    The face whose halfedge carries ``THIS_WAY`` lies *below* the face on
    the opposite side, in the folded model.

(The naming is a historical artifact; the mark is consumed by
:func:`eucare.shrink_rotate.assign_shrink_rotate_creases` to choose
mountain vs. valley fold assignments for the crease pattern.)

This module provides several strategies for setting THIS_WAY automatically.
All assigners share two important conventions:

1. **Border halfedges are skipped.** THIS_WAY is meaningful only on
   interior edges, where two faces meet.
2. **Already-assigned edges are not overwritten.** Each helper checks for
   an existing THIS_WAY on either side of an edge and leaves it alone if
   present. This makes the helpers *chainable* — apply a primary strategy
   first, then a tiebreaker:

   .. code-block:: python

       # Primary: faces with larger area on top.
       assign_this_way_by_face_area(G)
       # Tiebreaker on equal-area edges: distance from center.
       assign_this_way_by_distance(G)

   Use :func:`clear_this_way` to start over.

Strategies
----------

* :func:`assign_this_way_by_face_z_order` and
  :func:`assign_this_way_by_vertex_z_order` — explicit numeric ``z_order``
  attribute (e.g. set by user code or by a BFS).
* :func:`assign_this_way_by_face_bfs` — BFS distance from a chosen source
  face; nearer faces lie on top.
* :func:`assign_this_way_from_center` — convenience: BFS from the face
  closest to the geometric centroid of ``G``.
* :func:`assign_this_way_by_distance` — distance of face midpoints from a
  point (default: centroid); nearer faces lie on top.
* :func:`assign_this_way_by_face_degree` — larger-degree (or smaller-, see
  flag) faces lie on top.
* :func:`assign_this_way_by_face_area` — larger-area faces lie on top by
  default.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .. import base
from ..half import Face, HalfEdgeGraph
from ..search_trees import face_bfs_tree

THIS_WAY = "this_way"
"""Halfedge attribute name marking the *lower* side of an interior edge."""


def _interior_unassigned_halfedges(G: HalfEdgeGraph) -> Iterable:
    """Yield halfedges that are interior and have no THIS_WAY mark on either side."""
    for e in G.halfedges:
        if e.on_border() or e.rev.on_border():
            continue
        if THIS_WAY in e.attributes or THIS_WAY in e.rev.attributes:
            continue
        yield e


def clear_this_way(G: HalfEdgeGraph) -> None:
    """Remove all THIS_WAY marks from halfedges of *G*.

    Call before re-running an assignment from scratch.
    """
    for e in G.halfedges:
        if THIS_WAY in e.attributes:
            del e[THIS_WAY]


def assign_this_way_by_face_z_order(G: HalfEdgeGraph, key: str = "z_order") -> None:
    """Use a numeric face attribute ``key`` to orient each interior edge.

    The halfedge whose face has the *larger* value of ``key`` is left
    unmarked; the opposite halfedge gets THIS_WAY (i.e. its face lies
    below). Ties are broken using the mean ``key`` over the faces incident
    to each endpoint.

    Skips edges that already have THIS_WAY assigned on either side.
    """
    for e in _interior_unassigned_halfedges(G):
        f1, f2 = e.face, e.rev.face
        if f1[key] > f2[key]:
            e[THIS_WAY] = True
        elif f1[key] < f2[key]:
            e.rev[THIS_WAY] = True
        else:
            z_orig = np.mean([f[key] for f in e.orig.true_face_iter()])
            z_dest = np.mean([f[key] for f in e.dest.true_face_iter()])
            if z_orig > z_dest:
                e[THIS_WAY] = True
            elif z_dest > z_orig:
                e.rev[THIS_WAY] = True


def assign_this_way_by_vertex_z_order(G: HalfEdgeGraph, key: str = "z_order") -> None:
    """Like :func:`assign_this_way_by_face_z_order` but keyed on endpoint vertices.

    Skips edges that already have THIS_WAY assigned on either side.
    """
    for e in _interior_unassigned_halfedges(G):
        v1, v2 = e.orig, e.dest
        if v1[key] > v2[key]:
            e[THIS_WAY] = True
        elif v1[key] < v2[key]:
            e.rev[THIS_WAY] = True
        else:
            z_orig = np.mean([v[key] for v in e.face.vertex_iter()])
            z_dest = np.mean([v[key] for v in e.rev.face.vertex_iter()])
            if z_orig > z_dest:
                e[THIS_WAY] = True
            elif z_dest > z_orig:
                e.rev[THIS_WAY] = True


def _bfs_face_depth(G: HalfEdgeGraph, source: Face) -> dict:
    """Return BFS depth (#face-adjacencies from *source*) keyed by face id."""
    depth = {id(source): 0}
    queue = deque([source])
    while queue:
        f = queue.popleft()
        d = depth[id(f)]
        for e in f.halfedge_iter():
            if e.rev.on_border():
                continue
            f2 = e.rev.face
            if id(f2) in depth:
                continue
            depth[id(f2)] = d + 1
            queue.append(f2)
    return depth


def assign_this_way_by_face_bfs(G: HalfEdgeGraph, source: Face) -> None:
    """Use BFS depth from *source* face: shallower faces lie on top.

    Faces visited later (greater BFS depth) are marked as lying below; the
    halfedge of the deeper-visited face on each interior edge gets
    THIS_WAY.

    Skips edges that already have THIS_WAY assigned on either side.
    """
    # Reuse the canonical BFS spanner so the traversal matches other helpers.
    _ = face_bfs_tree(G, source)
    depth = _bfs_face_depth(G, source)
    for e in _interior_unassigned_halfedges(G):
        d_here = depth.get(id(e.face))
        d_other = depth.get(id(e.rev.face))
        if d_here is None or d_other is None:
            continue
        if d_here > d_other:
            e[THIS_WAY] = True
        elif d_other > d_here:
            e.rev[THIS_WAY] = True


def _graph_centroid(G: HalfEdgeGraph) -> np.ndarray:
    """Return the centroid of all vertex positions of *G*."""
    return np.mean(np.array([v["pos"] for v in G.vertices]), axis=0)


def _face_nearest_to(G: HalfEdgeGraph, point: np.ndarray) -> Face:
    """Return the face of *G* whose midpoint is closest to *point*."""
    return min(
        (f for f in G.faces if not f.on_border()),
        key=lambda f: float(np.linalg.norm(np.asarray(f.midpoint()) - point)),
    )


def assign_this_way_from_center(G: HalfEdgeGraph) -> None:
    """Convenience: BFS from the face nearest the geometric centroid of *G*.

    Equivalent to selecting the centermost face and calling
    :func:`assign_this_way_by_face_bfs`. Skips already-assigned edges.
    """
    center = _graph_centroid(G)
    src = _face_nearest_to(G, center)
    assign_this_way_by_face_bfs(G, src)


def assign_this_way_by_distance(G: HalfEdgeGraph, point=None) -> None:
    """Orient interior edges by distance of face midpoints from *point*.

    The face whose midpoint is *farther* from *point* lies below (gets
    THIS_WAY). When *point* is ``None`` the centroid of *G* is used.

    Skips already-assigned edges.
    """
    point = _graph_centroid(G) if point is None else np.asarray(point)
    for e in _interior_unassigned_halfedges(G):
        d_here = float(np.linalg.norm(np.asarray(e.face.midpoint()) - point))
        d_other = float(np.linalg.norm(np.asarray(e.rev.face.midpoint()) - point))
        if d_here > d_other:
            e[THIS_WAY] = True
        elif d_other > d_here:
            e.rev[THIS_WAY] = True


def assign_this_way_by_face_degree(G: HalfEdgeGraph, larger_on_top: bool = True) -> None:
    """Orient interior edges by face degree (number of incident edges).

    With *larger_on_top* (the default), the smaller-degree face is marked
    as lying below. Set ``larger_on_top=False`` to invert.

    Skips already-assigned edges.
    """
    for e in _interior_unassigned_halfedges(G):
        d_here = e.face.order()
        d_other = e.rev.face.order()
        if d_here == d_other:
            continue
        higher = (d_here > d_other) == larger_on_top
        # *higher* faces lie on top, so the *lower* face's halfedge gets THIS_WAY.
        if higher:
            e.rev[THIS_WAY] = True
        else:
            e[THIS_WAY] = True


def _signed_area(face: Face) -> float:
    pts = np.array([v["pos"] for v in face.vertex_iter()])
    return abs(base.signed_area(pts))


def assign_this_way_by_face_area(G: HalfEdgeGraph, larger_on_top: bool = True) -> None:
    """Orient interior edges by face area.

    With *larger_on_top* (the default), the smaller-area face lies below.
    Set ``larger_on_top=False`` to invert.

    Skips already-assigned edges.
    """
    areas = {id(f): _signed_area(f) for f in G.faces if not f.on_border()}
    for e in _interior_unassigned_halfedges(G):
        a_here = areas.get(id(e.face))
        a_other = areas.get(id(e.rev.face))
        if a_here is None or a_other is None or a_here == a_other:
            continue
        higher = (a_here > a_other) == larger_on_top
        if higher:
            e.rev[THIS_WAY] = True
        else:
            e[THIS_WAY] = True
