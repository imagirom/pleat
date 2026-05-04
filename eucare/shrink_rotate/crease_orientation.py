"""Halfedge orientation marks for crease pattern stacking order.

Many origami pipelines need to know, for each interior edge, which of the
two adjacent faces is above the other in the folded model. We encode
this via a single halfedge attribute, the **THIS_WAY** mark:

    The face whose halfedge carries ``THIS_WAY`` lies *above* the face on
    the opposite side, in the folded model.

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
from ..half import Face, HalfEdgeGraph, Vertex
from ..search_trees import face_bfs_tree, vertex_bfs_tree

THIS_WAY = "this_way"
"""Halfedge attribute name marking the *upper* side of an interior edge (h.face will lie above h.rev.face)."""


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
    above). Ties are broken using the mean ``key`` over the faces incident
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


def _bfs_face_depth(source: Face | set[Face]) -> dict:
    """Return BFS depth (#face-adjacencies from *source*) keyed by face id."""
    depth = {source: 0} if isinstance(source, Face) else {f: 0 for f in source}
    for orig, dest in face_bfs_tree(source):
        depth[dest] = depth[orig] + 1
    return depth


def _bfs_vertex_depth(source: Vertex | set[Vertex]) -> dict:
    """Return BFS depth (#edge-adjacencies from *source*) keyed by vertex id."""
    depth = {source: 0} if isinstance(source, Vertex) else {v: 0 for v in source}
    for orig, dest in vertex_bfs_tree(source):
        depth[dest] = depth[orig] + 1
    return depth


def assign_this_way_by_bfs(G: HalfEdgeGraph, source: Vertex | Face | set[Vertex] | set[Face]) -> None:
    """Use BFS depth from *source* face: shallower lies on top.

    Skips edges that already have THIS_WAY assigned on either side.
    """
    if not isinstance(source, set):
        source = {source}
    source_faces = {f for s in source for f in (s.true_face_iter() if isinstance(s, Vertex) else [s])}
    source_vertices = {v for s in source for v in (s.vertex_iter() if isinstance(s, Face) else [s])}

    assign_this_way_by_face_bfs(G, source_faces)  # primary: face BFS
    assign_this_way_by_vertex_bfs(G, source_vertices)  # tiebreaker: vertex BFS


def assign_this_way_by_face_bfs(G: HalfEdgeGraph, source: Face | set[Face]) -> None:
    """Use BFS depth from *source* face: shallower faces lie on top.

    Faces visited later (greater BFS depth) are marked as lying below; the
    halfedge of the deeper-visited face on each interior edge gets
    THIS_WAY.

    Skips edges that already have THIS_WAY assigned on either side.
    """
    # Reuse the canonical BFS spanner so the traversal matches other helpers.
    depth = _bfs_face_depth(source)
    for e in _interior_unassigned_halfedges(G):
        d_here = depth.get(e.face)
        d_other = depth.get(e.rev.face)
        if d_here is None or d_other is None:
            continue
        if d_here > d_other:
            e.rev[THIS_WAY] = True
        elif d_other > d_here:
            e[THIS_WAY] = True


def assign_this_way_by_vertex_bfs(G: HalfEdgeGraph, source: Vertex | set[Vertex]) -> None:
    """Use BFS depth from *source* vertex: shallower faces lie on top.

    Faces visited later (greater BFS depth) are marked as lying below; the
    halfedge of the deeper-visited face on each interior edge gets
    THIS_WAY.

    Skips edges that already have THIS_WAY assigned on either side.
    """
    depth = _bfs_vertex_depth(source)
    for e in _interior_unassigned_halfedges(G):
        d_here = depth.get(e.orig)
        d_other = depth.get(e.dest)
        if d_here is None or d_other is None:
            continue
        if d_here > d_other:
            e.rev[THIS_WAY] = True
        elif d_other > d_here:
            e[THIS_WAY] = True


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
            e.rev[THIS_WAY] = True
        elif d_other > d_here:
            e[THIS_WAY] = True
        else:  # faces have the same distance; go by distance of endpoints of the edge
            d_orig = float(np.linalg.norm(np.asarray(e.orig["pos"]) - point))
            d_dest = float(np.linalg.norm(np.asarray(e.dest["pos"]) - point))
            if d_orig > d_dest:
                e.rev[THIS_WAY] = True
            elif d_dest > d_orig:
                e[THIS_WAY] = True


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
        if higher:
            e[THIS_WAY] = True
        else:
            e.rev[THIS_WAY] = True


def assign_this_way_by_face_area(G: HalfEdgeGraph, larger_on_top: bool = True) -> None:
    """Orient interior edges by face area.

    With *larger_on_top* (the default), the smaller-area face lies below.
    Set ``larger_on_top=False`` to invert.

    Skips already-assigned edges.
    """
    areas = {f: f.area() for f in G.faces if not f.on_border()}
    for e in _interior_unassigned_halfedges(G):
        a_here = areas.get(e.face)
        a_other = areas.get(e.rev.face)
        if a_here is None or a_other is None or a_here == a_other:
            continue
        higher = (a_here > a_other) == larger_on_top
        if higher:
            e[THIS_WAY] = True
        else:
            e.rev[THIS_WAY] = True
