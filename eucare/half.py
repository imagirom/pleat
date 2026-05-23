"""Half-edge data structure (DCEL) for planar graphs.

This module implements the core data structure used throughout eucare.  A half-edge
graph stores topology via ``HalfEdge``, ``Vertex``, and ``Face`` objects linked
through ``rev``/``nex``/``pre`` pointers, with progressive specializations that
add interior angles (``InAngleHEG``), pluggable geometry (``GeometricHEG``), and
Euclidean vertex positions (``EuclideanPositionHEG``).

Class hierarchy::

    AttributeObject → IdObject → HalfEdge, Vertex, Face
    HalfEdgeGraph → CyclicHalfedgeGraph
    HalfEdgeGraph → InAngleHEG → GeometricHEG → EuclideanPositionHEG
"""

from __future__ import annotations

import heapq
import logging
from copy import copy, deepcopy
from itertools import chain
from math import pi

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .base import edge_lengths_and_in_angles, signed_area
from .geometries import EuclideanGeometry

logger = logging.getLogger(__name__)


class AttributeObject:
    """Mixin providing dict-like attribute storage via ``obj['key']``."""

    def __init__(self) -> None:
        super(AttributeObject, self).__init__()
        self.attributes: dict = dict()

    def has_attributes(self) -> bool:
        """Return True if this object has any attributes set."""
        return bool(self.attributes)

    def __getitem__(self, attr):
        return self.attributes[attr]

    def __setitem__(self, attr, value) -> None:
        self.attributes[attr] = value

    def __iter__(self):
        return iter(self.attributes)

    def __delitem__(self, key) -> None:
        del self.attributes[key]

    def __contains__(self, item) -> bool:
        return item in self.attributes

    def get(self, attr, *args, **kwargs):
        """Return ``self.attributes.get(attr, *args, **kwargs)``."""
        return self.attributes.get(attr, *args, **kwargs)

    def keys(self):
        """Return a view of the attribute keys."""
        return self.attributes.keys()

    def values(self):
        """Return a view of the attribute values."""
        return self.attributes.values()

    def items(self):
        """Return a view of the (key, value) attribute items."""
        return self.attributes.items()


class IdObject(AttributeObject):
    """AttributeObject with an auto-incrementing ``id`` attribute per subclass.

    Call ``IdObject.reset_ids()`` to reset all counters (useful between
    independent test runs).
    """

    current_ids: dict = dict()

    def __init__(self) -> None:
        super(IdObject, self).__init__()
        cls = type(self)
        IdObject.current_ids[cls] = IdObject.current_ids.get(cls, 0) + 1
        self["id"] = IdObject.current_ids[cls]

    @classmethod
    def reset_ids(cls) -> None:
        """Reset id counters.

        When called on :class:`IdObject` itself, clears every counter.
        When called on a subclass, resets only that subclass's counter.
        """
        if cls is IdObject:
            # print('resetting all ids')
            IdObject.current_ids = dict()
        else:
            IdObject.current_ids[cls] = 0


def check_cyclic_iterator_consistency(iterator) -> None:
    """Assert that every item yielded by ``iterator`` is unique (no cycles repeat)."""
    items = set()
    for item in iterator:
        assert item not in items, f"{iterator}, {item}, {items}"
        items.add(item)


class HalfEdge(IdObject):
    """A directed half-edge in the DCEL.

    Each undirected edge is represented by a pair of half-edges linked via
    ``rev``.  Navigation around a face uses ``nex`` / ``pre``; to traverse around a vertex use ``nex.rev``/``pre.rev``.  Border edges have ``face`` set to ``None`` on the half-edge pointing into the graph.

    Attributes:
        rev: The reverse (twin) half-edge.
        nex: The next half-edge around the same face.
        pre: The previous half-edge around the same face.
        orig: The origin vertex.
        dest: The destination vertex.
        face: The face to the left, or ``None`` for border edges.
    """

    printname = "HE"

    def __init__(
        self,
        rev: HalfEdge | None = None,
        nex: HalfEdge | None = None,
        pre: HalfEdge | None = None,
        orig: Vertex | None = None,
        dest: Vertex | None = None,
        face: Face | None = None,
    ) -> None:
        super(HalfEdge, self).__init__()
        # HalfEdge references
        self.rev = rev
        self.nex = nex
        self.pre = pre

        # Vertex references
        self.orig = orig
        self.dest = dest

        # Face reference
        self.face = face

    # def __repr__(self):
    #    return f'{self}({self.orig},{self.dest})'

    def __str__(self):
        return f'H{self["id"]}'

    def on_border(self) -> bool:
        """Return True if this half-edge has no face on its left side."""
        return self.face is None

    def check_consistency(self) -> None:
        """Assert that ``self.rev.rev is self``."""
        assert self.rev.rev is self, f"{self}, {self.rev}, {self.rev.rev}"

    def midpoint(self) -> np.ndarray:
        """Return the midpoint of the underlying edge in Euclidean position space."""
        return np.mean([v["pos"] for v in (self.orig, self.dest)], axis=0)


class Vertex(IdObject):
    """A vertex in the DCEL.

    Links to one arbitrary outgoing half-edge (``any_outgoing``).  All incident
    half-edges and faces are reachable via ``outgoing_iter()`` and
    ``face_iter()``.
    """

    printname = "V"

    def __init__(self, any_outgoing: HalfEdge | None = None) -> None:
        super(Vertex, self).__init__()
        self.any_outgoing = any_outgoing

    def outgoing_iter(self):
        """Yield outgoing half-edges in counter-clockwise order around this vertex."""
        initial = self.any_outgoing
        current = initial
        while True:
            yield current
            current = current.pre.rev
            if current is initial:
                break

    def reverse_outgoing_iter(self):
        """Yield outgoing half-edges in clockwise order around this vertex."""
        initial = self.any_outgoing
        current = initial
        while True:
            yield current
            current = current.rev.nex
            if current is initial:
                break

    def order(self) -> int:
        """Return the number of edges incident to this vertex (its degree)."""
        return len(list(self.outgoing_iter()))

    def incoming_iter(self):
        """Yield incoming half-edges in counter-clockwise order around this vertex."""
        for h in self.outgoing_iter():
            yield h.rev

    def face_iter(self):
        """Yield adjacent faces (including ``None`` for border) in CCW order."""
        for h in self.outgoing_iter():
            yield h.face

    def true_face_iter(self):
        """Yield only the non-``None`` adjacent faces."""
        for f in self.face_iter():
            if f is not None:
                yield f

    def vertex_iter(self):
        """Yield neighbouring vertices in counter-clockwise order around this vertex."""
        for h in self.outgoing_iter():
            yield h.dest

    def common_faces_iter(self, other: "Vertex"):
        """Yield faces incident to both this vertex and ``other``."""
        fs = set(other.face_iter())
        for f in self.face_iter():
            if f in fs and f is not None:
                yield f

    def on_border(self) -> bool:
        """Return True if this vertex has at least one border half-edge."""
        return any(h.on_border() for h in self.outgoing_iter())

    def get_outgoing_border(self) -> HalfEdge:
        """Return the unique outgoing border half-edge, asserting there is exactly one."""
        # search for borders
        border_edges = [h for h in self.outgoing_iter() if h.face is None]
        assert len(border_edges) > 0, f"Vertex {self} does not lie on a border. {self.order()}"
        assert len(border_edges) < 2, f"Vertex {self} has multiple adjacent border edges. Please specify one."
        return border_edges[0]

    def combine_with(self, other: "Vertex") -> "Vertex":
        """Merge ``other``'s attributes into this vertex (this vertex's values win on conflict)."""
        # This method might be used in subclasses to e.g. average positions when vertices are combined.
        # For now, just use one of them and combine the attributes.
        attrs = copy(other.attributes)
        attrs.update(self.attributes)
        self.attributes = attrs
        return self

    def check_consistency(self) -> None:
        """Assert that ``outgoing_iter`` is a finite cycle and incident edges agree on ``orig``/``dest``."""
        check_cyclic_iterator_consistency(self.outgoing_iter())
        check_cyclic_iterator_consistency(self.reverse_outgoing_iter())
        for e in self.outgoing_iter():
            assert e.orig is self, f"{self}, {e.orig}, {e}"
        for e in self.incoming_iter():
            assert e.dest is self, f"{self}, {e.dest}, {e}"
        assert self.order() > 0, f"{self}, {self.order()}"

    def angle_sum(self) -> float:
        """Return the sum of interior angles at this vertex (excluding border faces)."""
        return sum(h["in_angle"] for h in self.incoming_iter() if h.face is not None)


class Face(IdObject):
    """A face (polygon) in the DCEL.

    Links to one arbitrary bounding half-edge (``any_side``).  Iterate
    over boundary half-edges with ``halfedge_iter()`` and vertices with
    ``vertex_iter()``.
    """

    def __init__(self, any_side: HalfEdge | None = None) -> None:
        super(Face, self).__init__()
        self.any_side = any_side

    def halfedge_iter(self):
        """Yield boundary half-edges in counter-clockwise order around this face."""
        initial = self.any_side
        current = initial
        while True:
            yield current
            current = current.nex
            if current is initial:
                break

    def reverse_halfedge_iter(self):
        """Yield the reverse of each boundary half-edge (i.e. those facing outward)."""
        for h in self.halfedge_iter():
            yield h.rev

    def order(self) -> int:
        """Return the number of edges (= sides) of this polygonal face."""
        return len(list(self.halfedge_iter()))

    def vertex_iter(self):
        """Yield boundary vertices in counter-clockwise order around this face."""
        for h in self.halfedge_iter():
            yield h.orig

    def face_iter(self):
        """Yield neighbouring faces (including ``None`` for border) in CCW order."""
        for h in self.halfedge_iter():
            yield h.rev.face

    def true_face_iter(self):
        """Yield only the non-``None`` neighbouring faces."""
        for f in self.face_iter():
            if f is not None:
                yield f

    def on_border(self) -> bool:
        """Return True if any boundary half-edge of this face is adjacent to a border edge."""
        return any(h.rev.on_border() for h in self.halfedge_iter())

    def outgoing_edge_at(self, v: "Vertex") -> HalfEdge:
        """Return the boundary half-edge of this face originating at vertex ``v``."""
        return next(h for h in self.halfedge_iter() if h.orig is v)

    def incoming_edge_at(self, v: "Vertex") -> HalfEdge:
        """Return the boundary half-edge of this face ending at vertex ``v``."""
        return next(h for h in self.halfedge_iter() if h.dest is v)

    def common_halfedge_iter(self, other: "Face"):
        """Yield half-edges of this face whose reverse lies on ``other``."""
        assert isinstance(other, Face)
        hs = set(h.rev for h in other.halfedge_iter())
        for h in self.halfedge_iter():
            if h in hs:
                yield h

    def common_vertex_iter(self, other: "Face"):
        """Yield vertices shared with ``other``."""
        vs = set(other.vertex_iter())
        for v in self.vertex_iter():
            if v in vs:
                yield v

    def common_face_iter(self, other: "Face"):
        """Yield faces neighbouring both this face and ``other``."""
        fs = set(other.face_iter())
        for f in self.face_iter():
            if f in fs:
                yield f

    def midpoint(self) -> np.ndarray:
        """Return the (cached) face midpoint, falling back to the vertex centroid."""
        return self.attributes.get("midpoint", np.mean([v["pos"] for v in self.vertex_iter()], axis=0))

    def pseudo_incenter(self) -> np.ndarray:
        """Return the pseudo-incenter (true incenter for tangential polygons)."""
        return pseudo_incenter(self)

    def recompute_lengths_and_angles(self, geometry) -> None:
        """Recompute the ``length`` and ``in_angle`` attributes from current vertex positions."""
        points = np.stack([v["pos"] for v in self.vertex_iter()])
        lengths, angles = edge_lengths_and_in_angles(points, geometry)
        for e, length, angle in zip(self.halfedge_iter(), lengths, angles):
            e["length"] = length
            e["in_angle"] = angle

    def area(self) -> float:
        """Return the (signed) Euclidean area of this face from its vertex positions."""
        return signed_area(np.stack([v["pos"] for v in self.vertex_iter()]))

    def check_consistency(self) -> None:
        """Assert that ``halfedge_iter`` is a finite cycle and every boundary edge points back to this face."""
        check_cyclic_iterator_consistency(self.halfedge_iter())
        check_cyclic_iterator_consistency(self.reverse_halfedge_iter())
        for e in self.halfedge_iter():
            assert e.face is self, f"{self}, {e.face}, {e}"
        assert self.order() > 1, f"{self}, {self.order()}"


def pseudo_incenter(f: Face | np.ndarray) -> np.ndarray:
    """Return a face's pseudo-incenter (true incenter for tangential polygons).

    For tangential polygons the result is the actual incenter. Otherwise it
    is the side-length-weighted centroid of the vertices, which is a smooth
    interior point convenient for label placement.
    """
    if isinstance(f, np.ndarray):
        ps = f
    else:
        ps = np.array([v["pos"] for v in f.vertex_iter()])
    lengths = np.linalg.norm(np.roll(ps, 1, axis=0) - np.roll(ps, 2, axis=0), axis=-1)
    incenter = np.sum(lengths[:, None] * ps, axis=0) / np.sum(lengths)
    return incenter


def pseudo_circumcenter(ps, return_radius=False) -> np.ndarray | tuple[np.ndarray, float]:
    """Compute the "best fit" circumcenter of a polygon.

    Finds the interior point that minimizes the standard deviation of
    distances to the vertices.

    Args:
        ps (array_like): Polygon vertices, shape (N, 2).
        return_radius (bool): If True, also return the circumradius.

    Returns:
        The circumcenter (shape ``(2,)``). If ``return_radius`` is True, a
            ``(center, radius)`` tuple is returned instead, where ``radius``
            is the circumradius.
    """
    ps = np.asarray(ps, dtype=float)

    if ps.ndim != 2 or ps.shape[1] != 2:
        raise ValueError("ps must have shape (N, 2)")
    if ps.shape[0] < 3:
        # For 1 or 2 points, just use the mean of the points.
        center = ps.mean(axis=0)
    else:
        p0 = ps[0]

        # From ||c - pi||^2 = ||c - p0||^2:
        # 2 (pi - p0) · c = ||pi||^2 - ||p0||^2
        A = 2.0 * (ps[1:] - p0)
        b = np.sum(ps[1:] ** 2, axis=1) - np.sum(p0**2)

        if np.linalg.matrix_rank(A) < 2:
            raise ValueError("Points are collinear or degenerate")

        center, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    dists = np.linalg.norm(ps - center, axis=1)
    radius = dists.mean()

    if return_radius:
        return center, radius
    return center


class HalfEdgeGraph:
    """Topology-only half-edge graph (DCEL).

    Stores sets of ``vertices``, ``halfedges``, and ``faces``.  Supports
    gluing, copying, deletion, subdivision, and consistency checking.
    Most methods mutate the graph in-place and return ``None``; use
    ``.copy()`` first if you need the original.
    """

    def __init__(self, other: "HalfEdgeGraph | None" = None) -> None:
        if other is not None:
            self.halfedges = copy(other.halfedges)
            self.vertices = copy(other.vertices)
            self.faces = copy(other.faces)
            self._any_border = other._any_border
        else:
            self.halfedges = set()
            self.vertices = set()
            self.faces = set()
            self._any_border = None
        self.simply_connected = False

    def add_graph(self, other: "HalfEdgeGraph") -> None:
        """Add all elements of ``other`` to this graph (without re-linking topology)."""
        if not isinstance(other, HalfEdgeGraph):
            raise TypeError(f"other must be a HalfEdgeGraph. Got {type(other)}")
        self.add_halfedges(other.halfedges)
        self.add_vertices(other.vertices)
        self.add_faces(other.faces)

    @property
    def order(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def size(self) -> int:
        """Number of (undirected) edges, i.e. half-edges divided by 2."""
        return len(self.halfedges) // 2

    def add_vertex(self, v: Vertex) -> None:
        """Register vertex ``v`` with this graph."""
        self.vertices.add(v)

    def add_vertices(self, vs) -> None:
        """Register an iterable of vertices with this graph."""
        self.vertices.update(vs)

    def add_face(self, f: Face) -> None:
        """Register face ``f`` with this graph."""
        self.faces.add(f)

    def add_faces(self, fs) -> None:
        """Register an iterable of faces with this graph."""
        self.faces.update(fs)

    def add_halfedge(self, h: HalfEdge) -> None:
        """Register half-edge ``h`` with this graph."""
        self.halfedges.add(h)

    def add_halfedges(self, hs) -> None:
        """Register an iterable of half-edges with this graph."""
        self.halfedges.update(hs)

    def delete_face(self, f: Face) -> None:
        """Remove a single face (and any newly-orphaned edges/vertices)."""
        self.delete_faces({f})

    def fill_holes(self):
        """Fill all 'holes' inside the graph: Add faces to enclosed areas which are not yet a face"""
        raise NotImplementedError

    def delete_faces(self, fs) -> None:
        """Delete all faces in ``fs`` and any half-edges/vertices they leave dangling."""
        self.faces.difference_update(set(fs))
        # 1. Determine edges that have to be deleted, remove them and their rev's from Graph
        # 2. Determine which still existing edges should be updated.
        #    These are the .nex and .pre's from the deleted ones and their rev's. Update the corresponding properties.
        # 3. Update any_outgoing of affected vertices, delete those where it's impossible
        affected_halfedges = {h for f in fs for h in f.halfedge_iter()}
        affected_vertices = {v for f in fs for v in f.vertex_iter()}
        for h in affected_halfedges:
            h.face = None
        # if arbitrary halfedges would be deleted, it might lead to more faces being deleted. This cannot happen here.
        deleted_halfedges = {h for h in affected_halfedges if h.rev.face is None}
        deleted_halfedges = deleted_halfedges.union({h.rev for h in deleted_halfedges})
        self.halfedges.difference_update(deleted_halfedges)

        # update properties of .nex and .pre edges of affected halfedges
        for h in deleted_halfedges:
            if h.pre in self.halfedges:
                while h.pre.nex not in self.halfedges:
                    # walk around the vertex h.orig in clockwise order
                    h.pre.nex = h.pre.nex.rev.nex
                    if h.pre.nex is h.pre.rev:
                        break

            if h.nex in self.halfedges:
                while h.nex.pre not in self.halfedges:
                    h.nex.pre = h.nex.pre.rev.pre
                    # walk around the vertex h.dest in counterclockwise order
                    if h.nex.pre is h.nex.rev:
                        break

        # delete affected vertices that no longer have a connection to the graph
        for v in affected_vertices:
            if v.any_outgoing not in self.halfedges:
                for h in v.outgoing_iter():
                    if h in self.halfedges:
                        v.any_outgoing = h
                        break
                if v.any_outgoing not in self.halfedges:  # no outgoing edge is still in graph
                    self.vertices.remove(v)

    def delete_subset(self, *items) -> None:
        """Delete a mixed collection of faces, half-edges, and vertices, repairing topology.

        Accepts individual ``Face``/``HalfEdge``/``Vertex`` objects or iterables
        of them as positional arguments. Border structure is updated; new faces
        may be created when an edge cut splits an existing face.
        """
        parsed_items = []
        for item in items:
            if isinstance(item, (Face, HalfEdge, Vertex)):
                parsed_items.append(item)
            else:
                for i in item:
                    assert isinstance(i, (Face, HalfEdge, Vertex)), f"{type(i)}, {i}"
                parsed_items.extend(item)
        faces = {f for f in parsed_items if isinstance(f, Face)}
        edges = {h for h in parsed_items if isinstance(h, HalfEdge)}
        vertices = {v for v in parsed_items if isinstance(v, Vertex)}

        # FIXME: if all border edges are deleted, there no longer is a border.
        # FIXME: this is currently not checking for non-simply connected faces. Those could appear.
        faces = set() if faces is None else set(faces)
        edges = set() if edges is None else set(edges)
        vertices = set() if vertices is None else set(vertices)

        # deleting a vertex for now amounts to deleting all outgoing edges
        # the only alternative definition that comes to mind is joining degree 2 vertices
        edges.update({h for v in vertices for h in v.outgoing_iter()})

        # implications from deleting the faces
        self.faces.difference_update(set(faces))
        adjacent_halfedges = {h for f in faces for h in f.halfedge_iter()}
        for h in adjacent_halfedges:
            h.face = None

        edges.update({h for h in adjacent_halfedges if h.rev.face is None})  # edges with no face are deleted
        edges.update({h.rev for h in edges})
        self.halfedges.difference_update(edges)

        adjacent_vertices = {h.orig for h in edges}
        adjacent_vertices.update(h.dest for h in edges)

        # update properties of .nex and .pre of deleted edges
        for h in edges:
            if h.pre in self.halfedges:
                while h.pre.nex not in self.halfedges:
                    # walk around the vertex h.orig in clockwise order
                    h.pre.nex = h.pre.nex.rev.nex
                    if h.pre.nex is h.pre.rev:
                        break

            if h.nex in self.halfedges:
                while h.nex.pre not in self.halfedges:
                    h.nex.pre = h.nex.pre.rev.pre
                    # walk around the vertex h.dest in counterclockwise order
                    if h.nex.pre is h.nex.rev:
                        break

        # sort out face references
        edges_to_check = self.halfedges.intersection({h.nex for h in edges}.union({h.pre for h in edges}))
        new_faces = set()
        while edges_to_check:
            h = edges_to_check.pop()
            if h.face is None:
                continue
            h2 = h
            new_face = None
            while h2.face is not None:  # and h2.face not in self.faces:
                h2 = h2.nex
                new_face = new_face if h2.face not in self.faces else h2.face
                if h is h2:
                    break
            if h2.face is None:  # connected to border, faces will be deleted
                new_face = None
            # new_face = None if h is h2 and (h2.face not in self.faces) else h2.face
            new_faces.add(new_face)
            if new_face:
                new_face.any_side = h2
            h3 = h2
            while True:
                assert h3 in self.halfedges
                if h3.face and h3.face is not new_face:
                    self.faces.discard(h3.face)
                h3.face = new_face
                h3 = h3.nex
                edges_to_check.discard(h3)
                if h3 is h2:
                    break

        # ..and delete faces that no longer have any edge
        for e in edges:
            if e.face not in new_faces:
                self.faces.discard(e.face)

        # delete affected vertices that no longer have a connection to the graph
        for v in adjacent_vertices:
            if v.any_outgoing not in self.halfedges:
                for h in v.outgoing_iter():
                    if h in self.halfedges:
                        v.any_outgoing = h
                        break
                if v.any_outgoing not in self.halfedges:  # no outgoing edge is still in graph
                    self.vertices.remove(v)

    def delete_edge(self, h: HalfEdge) -> None:
        """Remove the edge ``(h, h.rev)`` and merge its two adjacent faces.

        If both sides share the same face, ``h`` is required to be a dangling
        spike (one endpoint has degree 1).
        """
        if h.on_border():
            raise NotImplementedError
        if h.face is h.rev.face:
            assert (
                h.orig.order() == 1 or h.dest.order() == 1
            ), f"removal of {h} might lead to {h.face} not being simply connected."
        else:
            self.faces.remove(h.face)
        self.halfedges.difference_update({h, h.rev})
        f = h.rev.face
        f.any_side = h.rev.nex

        # assign new, bigger face to edges adjacent to deleted face
        for k in h.face.halfedge_iter():
            k.face = f

        # update nex and pre and any_outgoing where necessary
        for k in [h, h.rev]:
            k.pre.nex = k.rev.nex
            k.nex.pre = k.rev.pre
            k.dest.any_outgoing = k.nex

        # TODO  check if there are any dangling edges next to h after removal. remove them.
        # for k in [h, h.rev]:
        #     if k.orig.order() == 1 and k.orig.any_outgoing in self.halfedges:
        #         self.delete_edge(k.orig.any_outgoing)

    def join_vertex(self, v: Vertex) -> None:
        """Remove a degree-2 vertex by merging its two incident edges into one."""
        assert v.order() == 2, f"Can only join vertices of order 2, ({v}, {v.order()})"
        v_out = v.any_outgoing
        v_out.rev.nex.rev = v_out
        v_out.rev = v_out.rev.nex
        v_out.check_consistency()
        v_out.rev.check_consistency()

        for h in [v_out, v_out.rev]:
            self.halfedges.remove(h.pre)
            h.orig = h.pre.orig
            h.orig.any_outgoing = h
            h.pre = h.pre.pre
            h.pre.nex = h
            if not h.on_border():
                h.face.any_side = h
        self.vertices.remove(v)

    def join_order_2_boundary_vertices(self) -> None:
        """Simplify the boundary by joining all degree-2 vertices on the boundary."""
        to_join = []
        for v in self.vertices:
            if v.on_border() and v.order() == 2:
                to_join.append(v)
        for v in to_join:
            self.join_vertex(v)
        self.recompute_lengths_and_angles()

    def join_edge(self, h: HalfEdge) -> Vertex:
        """Contract the half-edge ``h`` (and its reverse), merging its two endpoints.

        Args:
            h: The half-edge to contract.

        Returns:
            The remaining vertex (originally ``h.orig``); ``h.dest`` is removed.
        """
        assert h.on_border() or h.face.order() > 1
        assert h.rev.on_border() or h.rev.face.order() > 1

        v1, v2 = h.orig, h.dest

        self.halfedges.difference_update((h, h.rev))
        self.vertices.remove(v2)

        for h2 in v2.outgoing_iter():
            h2.orig = v1
            h2.rev.dest = v1

        v1.combine_with(v2)

        for h2 in [h, h.rev]:
            h2.pre.nex = h2.nex
            h2.nex.pre = h2.pre
            if not h2.on_border():
                h2.face.any_side = h2.nex
            h2.dest.any_outgoing = h2.nex

        return v1

    def subdivide_edge(
        self, h: HalfEdge, copy_edge_attributes: bool = True, **vertex_attributes: object
    ) -> tuple[HalfEdge, Vertex]:
        """Insert a new vertex on the edge ``h``, splitting it into two.

        A new half-edge ``h2`` is created with ``h.nex == h2``. The reverse
        side is split symmetrically.

        Args:
            h: The half-edge to subdivide.
            copy_edge_attributes: If True, copy the attributes of ``h`` and
                ``h.rev`` to the newly created halves.
            **vertex_attributes: Attributes to set on the newly inserted vertex.

        Returns:
            ``(h2, v)`` where ``h2`` is the newly inserted half-edge
            (``h.nex == h2``) and ``v`` is the inserted vertex.
        """
        # new vertex
        v = Vertex(any_outgoing=h.rev)

        h2 = HalfEdge(nex=h.nex, pre=h, orig=v, dest=h.dest, face=h.face)
        h2.nex.pre = h2

        h2rev = HalfEdge(rev=h2, nex=h.rev, pre=h.rev.pre, orig=h2.dest, dest=h2.orig, face=h.rev.face)
        h2rev.pre.nex = h2rev
        h2rev.orig.any_outgoing = h2rev

        h2.rev = h2rev

        h.nex = h2
        h.dest = v
        h.rev.pre = h2rev
        h.rev.orig = v

        if copy_edge_attributes:
            h2.attributes = copy(h.attributes)
            h2rev.attributes = copy(h.rev.attributes)

        for key, value in vertex_attributes.items():
            v[key] = value

        self.add_vertex(v)
        self.add_halfedges([h2, h2rev])

        return h2, v

    def subdivide_face(self, f: Face, v1: Vertex, v2: Vertex, **halfedge_attributes: object) -> tuple[HalfEdge, Face]:
        """Subdivide face ``f`` along a new edge from ``v1`` to ``v2``.

        Two new half-edges ``h12`` (``v1 -> v2``) and ``h21`` (``v2 -> v1``)
        are added together with a new face ``f2`` on the ``h12`` side; the
        original face ``f`` remains on the ``h21`` side.

        Args:
            f: The face to subdivide.
            v1: First endpoint of the new edge (must lie on ``f``).
            v2: Second endpoint of the new edge (must lie on ``f``).
            **halfedge_attributes: Attributes to set on both new half-edges.

        Returns:
            ``(h12, f2)`` where ``h12`` is the new half-edge from ``v1`` to
            ``v2`` and ``f2`` is the newly created face.
        """

        v1_out = next(h for h in v1.outgoing_iter() if h.face is f)
        v2_out = next(h for h in v2.outgoing_iter() if h.face is f)

        # create the new face and halfedges and set their attributes; add them to the graph
        f2 = Face(any_side=v2_out)
        h12 = HalfEdge(rev=None, nex=v2_out, pre=v1_out.pre, orig=v1, dest=v2, face=f2)
        h21 = HalfEdge(rev=h12, nex=v1_out, pre=v2_out.pre, orig=v2, dest=v1, face=f)
        for key, value in halfedge_attributes.items():
            h12[key] = h21[key] = value
        h12.rev = h21
        self.add_face(f2)
        self.add_halfedges([h12, h21])

        # update attributes of the affected existing face and adjacent halfedges
        if f is not None:
            f.any_side = v1_out
        v1_out.pre.nex = h12
        v1_out.pre = h21
        v2_out.pre.nex = h21
        v2_out.pre = h12
        for h in f2.halfedge_iter():
            h.face = f2

        return h12, f2

    def halfedges_representing_edges(self) -> set[HalfEdge]:
        """Return one half-edge per undirected edge.

        Returns:
            A set containing exactly one of ``{h, h.rev}`` for every
            undirected edge of the graph.
        """
        result = set()
        for h in self.halfedges:
            if h.rev not in result:
                result.add(h)
        return result

    def to_networkx_undirected(self) -> nx.Graph:
        """Return a :class:`networkx.Graph` of this graph's underlying undirected topology."""
        result = nx.Graph()
        result.add_edges_from([(h.orig, h.dest) for h in self.halfedges])
        return result

    def get_any_border(self) -> HalfEdge:
        """Return some border half-edge, caching the result; raise if none exists."""
        if self._any_border is not None and self._any_border.on_border() and self._any_border in self.halfedges:
            return self._any_border
        for h in self.halfedges:
            if h.on_border():
                self._any_border = h
                return h
        raise LookupError("No border found.")

    def border_edge_iter(self):
        """Yield border half-edges. For simply connected graphs, in cyclic order."""
        if self.simply_connected:
            initial = self.get_any_border()
            current = initial
            while True:
                yield current
                current = current.nex
                if current is initial:
                    break
        else:
            for e in self.halfedges:
                if e.on_border():
                    yield e

    def border_edges(self) -> list[HalfEdge]:
        """Return all border half-edges as a list."""
        return list(self.border_edge_iter())

    def border_vertex_iter(self):
        """Yield vertices lying on a border edge."""
        for h in self.border_edge_iter():
            yield h.orig

    def border_vertices(self) -> list[Vertex]:
        """Return all border vertices as a list."""
        return list(self.border_vertex_iter())

    def glue_v2v(
        self,
        v1: Vertex | None = None,
        v2: Vertex | None = None,
        v1_out: HalfEdge | None = None,
        v2_out: HalfEdge | None = None,
    ) -> None:
        """Identify two boundary vertices, merging them into a single vertex.

        Either ``v1`` or ``v1_out`` must be specified (likewise for ``v2``).
        When only the vertex is given, an outgoing border half-edge is found
        automatically.

        Args:
            v1: First vertex to merge.
            v2: Second vertex to merge.
            v1_out: Outgoing border half-edge at ``v1``.
            v2_out: Outgoing border half-edge at ``v2``.
        """
        # check if both vertices are suited for gluing
        if v1 is None:
            assert v1_out is not None, "v1 or v1_out must be specified."
            v1 = v1_out.orig
        if v2 is None:
            assert v2_out is not None, "v2 or v2_out must be specified."
            v2 = v2_out.orig

        # search for border edges if not specified
        v1_out = v1.get_outgoing_border() if v1_out is None else v1_out
        v2_out = v2.get_outgoing_border() if v2_out is None else v2_out
        assert (v1_out.orig is v1) and (v2_out.orig is v2), f"({repr(v1_out)}, {v1}), ({repr(v2_out)}, {v2})"
        # get incoming border edges
        v1_in, v2_in = v1_out.pre, v2_out.pre
        # assert v1_out.on_border() and v2_out.on_border() and v1_in.on_border() and v2_in.on_border()

        # assign new vertex, remove old
        v = v1.combine_with(v2)
        for h in chain(v1.outgoing_iter(), v2.outgoing_iter()):
            h.orig = v
            h.rev.dest = v
        self.vertices.difference_update({v1, v2})
        self.vertices.add(v1)

        # do the shuffle
        v1_out.pre = v2_in
        v2_out.pre = v1_in
        v1_in.nex = v2_out
        v2_in.nex = v1_out

    def glue_e2e(self, e1: HalfEdge, e2: HalfEdge) -> None:
        """Identify two border half-edges, sewing the boundary together.

        ``e1`` and ``e2`` must both lie on the border (or together bound a
        single inner face that gets eliminated). Their endpoints are merged
        via :meth:`glue_v2v`.
        """
        if all(a.orig is not b.dest for a, b in [(e1, e2), (e2, e1)]):
            for e in (e1, e2):
                if not e.on_border():
                    raise ValueError(f"Cannot glue: Edge {e} not on border.")
        else:
            # handle face stuff now
            if e1.face is e2.face is None:
                pass
            elif e1.nex is e2 and e2.nex is e1:
                self.faces.remove(e1.face)
            elif e1.nex is e2:
                e1.face.any_side = e2.nex
            elif e2.nex is e1:
                e1.face.any_side = e1.nex
            else:
                raise ValueError(f"Cannot glue: Edges {[e1, e2]} of face {e1.face} are not adjacent.")

        # glue vertices
        for v1_out, v2_out in ((e1, e2.nex), (e1.nex, e2)):
            if v1_out.orig != v2_out.orig:
                self.glue_v2v(v1_out=v1_out, v2_out=v2_out)

        # eliminate double edge
        e1.rev.rev = e2.rev
        e2.rev.rev = e1.rev

        e1.orig.any_outgoing = e2.rev  # e1.rev.nex
        e1.dest.any_outgoing = e1.rev  # e1.rev.pre.rev
        self.halfedges.difference_update({e1, e2})

    def glue_graph_e2e(self, graph: "HalfEdgeGraph", e1: HalfEdge, e2: HalfEdge) -> None:
        """Add ``graph`` to ``self`` and immediately glue ``e1`` to ``e2``."""
        self.add_graph(graph)
        self.glue_e2e(e1, e2)

    def close_vertex(self, v: Vertex) -> None:
        """Close a boundary corner at ``v`` by gluing its two incident border edges."""
        # get edges to be glued
        e1 = v.get_outgoing_border()
        e2 = e1.pre
        self.glue_e2e(e1, e2)

    def twocolorable(self) -> bool:
        """Return True if every interior vertex has even order (necessary for face 2-coloring)."""
        return all([v.order() % 2 == 0 for v in self.vertices if not v.on_border()])

    def twocolor_faces(self, key: str = "color_key", initial_face: Face | None = None) -> None:
        """Two-color the faces of the graph.

        Each face will get the attribute ``key`` set to ``True`` or ``False``;
        the initial face is labelled ``True``.

        Args:
            key: Attribute name to write the boolean color into.
            initial_face: Face to start the BFS from. Defaults to an arbitrary face.
        """
        assert self.twocolorable(), "Graph is not twocolorable, since it has an inner vertex of odd order!"
        if initial_face is None:
            initial_face = next(iter(self.faces))
        yet_to_color = copy(self.faces)
        frontier = {(initial_face, False)}
        while frontier:
            face, label = frontier.pop()
            if face not in yet_to_color:
                continue
            yet_to_color.remove(face)
            face[key] = label
            for f in face.face_iter():
                frontier.add((f, not label))
        assert not yet_to_color, "Graph is not connected!"

    def execute_edge_instruction(self, h: HalfEdge, instruction=None, key: str | None = None) -> None:
        """Run a tile-gluing or growth ``instruction`` on the border half-edge ``h``.

        If ``instruction`` is omitted, it is read from ``h[key]`` (default key
        ``'instruction'``).
        """
        if instruction is None:
            key = "instruction" if key is None else key
            instruction = h[key]
        else:
            assert key is None, "Please specify not more than one of [key, instruction]."
        instruction(self, h)

    def execute_all_edge_instructions(self, instruction=None, key: str | None = None) -> None:
        """Execute :meth:`execute_edge_instruction` on every border half-edge."""
        for h in self.border_edges():
            self.execute_edge_instruction(h, instruction, key)

    def show_spring_layout(self, figsize: tuple[float, float] = (15, 15), emph_func: "callable | None" = None) -> None:
        """Display the underlying undirected graph using NetworkX's spring layout.

        Args:
            figsize: Matplotlib figure size.
            emph_func: Callable ``HalfEdge -> bool``; matching edges are drawn red.
                Defaults to highlighting edges flagged with the ``'delete'`` attribute.
        """

        G = nx.Graph()
        G.add_edges_from([(h.orig, h.dest) for h in self.halfedges])

        if emph_func is None:

            def emph_func(h):
                return h.attributes.get("delete", False)

        G.add_edges_from([(h.orig, h.dest) for h in self.halfedges if emph_func(h)], color="r")

        pos = nx.spring_layout(G.to_undirected())
        plt.figure(figsize=figsize)

        colors = [G[u][v].get("color", "b") for u, v in G.edges]
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, edge_color=colors, arrows=True)

        nx.draw_networkx_labels(G, pos)
        plt.show()

    def check_consistency(self) -> None:
        """Verify the graph's topology is internally consistent.

        Runs the local :meth:`check_consistency` of every half-edge, vertex,
        and face, and additionally verifies that every reference (``nex``,
        ``pre``, ``rev``, ``orig``, ``dest``, ``face``, ``any_outgoing``,
        ``any_side``) points to an element registered with this graph.

        Raises:
            RuntimeError: If any inconsistency is detected.
        """
        # check local consistency
        for e in self.halfedges:
            e.check_consistency()
        for f in self.faces:
            f.check_consistency()
        for v in self.vertices:
            v.check_consistency()

        # global checks
        referenced_halfedges = set()
        referenced_halfedges.update({h.nex for h in self.halfedges})
        referenced_halfedges.update({h.pre for h in self.halfedges})
        referenced_halfedges.update({h.rev for h in self.halfedges})
        referenced_halfedges.update({v.any_outgoing for v in self.vertices})
        referenced_halfedges.update({f.any_side for f in self.faces})

        referenced_vertices = set()
        referenced_vertices.update({h.orig for h in self.halfedges})
        referenced_vertices.update({h.dest for h in self.halfedges})

        referenced_faces = {h.face for h in self.halfedges}
        referenced_faces.discard(None)

        if (
            referenced_vertices != self.vertices
            or referenced_faces != self.faces
            or referenced_halfedges != self.halfedges
        ):
            logger.error("=" * 50)
            logger.error("consistency error")

            logger.error(
                "halfedges: %s, %s",
                referenced_halfedges.difference(self.halfedges),
                self.halfedges.difference(referenced_halfedges),
            )
            logger.error(
                "vertices: %s, %s",
                referenced_vertices.difference(self.vertices),
                self.vertices.difference(referenced_vertices),
            )
            logger.error(
                "faces: %s, %s", referenced_faces.difference(self.faces), self.faces.difference(referenced_faces)
            )

            reference_dict = {
                obj: set()
                for obj in referenced_halfedges.union(referenced_vertices).union(referenced_faces).union([None])
            }
            for h in self.halfedges:
                for attr in ["nex", "pre", "rev", "orig", "dest", "face"]:
                    reference_dict[getattr(h, attr)].add((h, attr))
            for v in self.vertices:
                for attr in ["any_outgoing"]:
                    reference_dict[getattr(v, attr)].add((v, attr))
            for f in self.faces:
                for attr in ["any_side"]:
                    reference_dict[getattr(f, attr)].add((f, attr))
            for obj in (
                referenced_vertices.difference(self.vertices)
                .union(referenced_faces.difference(self.faces))
                .union(referenced_halfedges.difference(self.halfedges))
            ):
                logger.error("%s referenced by %s.", obj, reference_dict[obj])
            raise RuntimeError("Graph consistency check failed. See log for details.")

    def copy(
        self, deepcopy_attributes: bool = False, return_mappings: bool = False
    ) -> "HalfEdgeGraph | tuple[HalfEdgeGraph, tuple[dict, dict, dict]]":
        """Return an independent copy of this graph.

        Args:
            deepcopy_attributes: If True, attribute dicts are deep-copied;
                otherwise they are shallow-copied.
            return_mappings: If True, also return ``(v_map, e_map, f_map)``
                dictionaries from old to new vertices, half-edges, and faces.

        Returns:
            The copied graph -- a tuple ``(graph, (v_map, e_map, f_map))`` when ``return_mappings`` is True.
        """

        def copy_with_attributes(obj):
            cls = type(obj)
            new = cls.__new__(cls)
            new.attributes = copy(obj.attributes)
            return new

        def copy_with_attributes_deep(obj):
            cls = type(obj)
            new = cls.__new__(cls)
            new.attributes = deepcopy(obj.attributes)
            return new

        copy_func = copy_with_attributes_deep if deepcopy_attributes else copy_with_attributes

        # init mappings from old to new vertex/halfedge/face objects
        v_map, e_map, f_map = [
            {obj: copy_func(obj) for obj in container} for container in (self.vertices, self.halfedges, self.faces)
        ]

        # copy other potential attributes of graph (e.g. tau for InAngleHEG)
        cls = type(self)
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.vertices = set(v_map.values())
        result.halfedges = set(e_map.values())
        result.faces = set(f_map.values())

        # copy topology
        for v, v_new in v_map.items():
            v_new.any_outgoing = e_map[v.any_outgoing]

        for f, f_new in f_map.items():
            f_new.any_side = e_map[f.any_side]

        f_map[None] = None  # to handle border
        for e, e_new in e_map.items():
            e_new.orig = v_map[e.orig]
            e_new.dest = v_map[e.dest]
            e_new.nex = e_map[e.nex]
            e_new.pre = e_map[e.pre]
            e_new.rev = e_map[e.rev]
            e_new.face = f_map[e.face]

        if not return_mappings:
            return result
        else:
            return result, (v_map, e_map, f_map)


# ------------------------------------------------ faces with in-angles ------------------------------------------------

# class AngledHalfEdge(HalfEdge):
#     def __init__(self, in_angle=None, *super_args, **super_kwargs):
#         super(AngledHalfEdge, self).__init__(*super_args, **super_kwargs)
#         self['in_angle'] = in_angle


class InAngleHEG(HalfEdgeGraph):
    """Half-edge graph with interior angles and edge lengths.

    The angle between ``e`` and ``e.nex`` is stored in ``e['in_angle']``.
    Edge lengths are stored in ``e['length']``.  Every non-border half-edge
    should carry both attributes.
    """

    def __init__(
        self,
        angle_sum: float | None = None,
        eps: float | None = None,
        other: "HalfEdgeGraph | None" = None,
    ) -> None:
        """Create an angle-aware half-edge graph.

        Args:
            angle_sum: Total angle around an interior vertex (``2*pi`` for the
                Euclidean plane, ``< 2*pi`` for hyperbolic, ``> 2*pi`` for
                spherical). Defaults to ``2*pi`` (or ``other.tau`` if given).
            eps: Tolerance for angle/length equality. Defaults to ``1e-6``.
            other: Optional graph to copy elements from (forwarded to base).
        """
        super(InAngleHEG, self).__init__(other=other)
        self.tau = 2 * pi
        self.eps = 1e-6
        if other is not None:
            self.tau = other.tau if hasattr(other, "tau") else self.tau
            self.eps = other.eps if hasattr(other, "eps") else self.eps
        # tau = 2*pi, the sum of angles
        self.tau = angle_sum if angle_sum is not None else self.tau
        # tolerance for deciding weather angles are equal
        self.eps = eps if eps is not None else self.eps

    def is_tau(self, angle: float) -> bool:
        """Return True if ``angle`` is within ``eps`` of the full turn ``tau``."""
        return abs(angle - self.tau) < self.eps

    def delete_edge(self, h: HalfEdge) -> None:
        """Remove edge ``h`` and merge the two surrounding interior angles."""
        super(InAngleHEG, self).delete_edge(h)
        for k in [h, h.rev]:
            k.pre["in_angle"] += k.rev["in_angle"]

    def close_vertex(self, v: Vertex, reverse: bool = False) -> Vertex:
        """Glue together the two border edges incident at ``v``.

        Args:
            v: Boundary vertex whose two adjacent border edges are to be sewn.
            reverse: Swaps which side's vertex is kept.

        Returns:
            The vertex that remains after the gluing.
        """
        # reverse specifies which vertex will be kept.
        e = v.get_outgoing_border()
        edges = (e, e.pre)
        if reverse:
            edges = edges[::-1]
        super(InAngleHEG, self).glue_e2e(*edges)
        # return the new vertex
        return e.rev.orig

    def autoclose_vertex(self, v: Vertex, reverse: bool = False, recursive: bool = True) -> None:
        """Close ``v`` if its boundary angle sum equals ``tau``; recurse on the new vertex.

        Args:
            v: Boundary vertex to check.
            reverse: Forwarded to :meth:`close_vertex`.
            recursive: If True, recurse into the resulting vertex.

        Raises:
            RuntimeError: If the vertex's angle sum exceeds ``tau``.
        """
        anglesum = v.angle_sum()
        if self.is_tau(anglesum):
            v_next = self.close_vertex(v, reverse=reverse)
            if recursive:
                self.autoclose_vertex(v_next, reverse=reverse, recursive=True)
        elif anglesum > self.tau:
            raise RuntimeError(f"Vertex {v} has anglesum of {anglesum} > {self.tau}")
            # assert False, f'Vertex {v} has anglesum of {anglesum} > {self.tau}'

    def glue_e2e(
        self,
        e1: HalfEdge,
        e2: HalfEdge,
        auto_close: bool = True,
        auto_close_recursive: bool = True,
    ) -> None:
        """Glue ``e1`` to ``e2`` and optionally close any resulting full corners.

        Args:
            e1: First border half-edge.
            e2: Second border half-edge.
            auto_close: If True, automatically close vertices whose interior
                angle sum reaches ``tau`` after the gluing.
            auto_close_recursive: If True, propagate auto-closing recursively.
        """
        # glue the edges
        super(InAngleHEG, self).glue_e2e(e1, e2)
        if not auto_close:
            # if the vertices should not be closed, that is it
            return
        if e1.rev.orig.on_border():  # should always be the case
            # TODO: think about reverse properly...
            self.autoclose_vertex(e1.rev.orig, reverse=False, recursive=auto_close_recursive)
        else:
            raise RuntimeError("Expected vertex to be on border after glue operation")
        if (
            e1.rev.dest in self.vertices and e1.rev.dest.on_border()
        ):  # this might not be the case, if everything is closed by the previous operation
            self.autoclose_vertex(e1.rev.dest, reverse=True, recursive=auto_close_recursive)


class GeometricHEG(InAngleHEG):
    """Half-edge graph with a pluggable geometry backend.

    The ``geometry`` attribute (default :class:`EuclideanGeometry`) provides
    ``distance``, ``angle``, and ``center_of_mass`` operations that are
    used for position recomputation and length/angle calculations.
    """

    def __init__(self, geometry: object = EuclideanGeometry, **super_kwargs: object) -> None:
        """Create a graph backed by the given geometry.

        Args:
            geometry: A geometry backend exposing ``distance``, ``angle``,
                ``construct_next_poly_point``, and ``to_euclidean``. Defaults
                to :class:`EuclideanGeometry`.
            **super_kwargs: Forwarded to :class:`InAngleHEG`.
        """
        super(GeometricHEG, self).__init__(**super_kwargs)
        self.geometry = geometry

    def positions_coincide(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Return True if positions ``p1`` and ``p2`` are within ``eps``."""
        return np.linalg.norm(p1 - p2) < self.eps

    def lengths_equal(self, l1: float, l2: float) -> bool:
        """Return True if lengths ``l1`` and ``l2`` differ by at most ``eps``."""
        return abs(l1 - l2) <= self.eps

    def glue_graph_e2e(self, graph: "HalfEdgeGraph", e1: HalfEdge, e2: HalfEdge) -> None:
        """Add ``graph`` to ``self`` and glue ``e1`` to ``e2``, then recompute positions of new vertices."""
        assert self.lengths_equal(e1.rev["length"], e2.rev["length"])
        super().glue_graph_e2e(graph, e1, e2)
        e = (e1 if e1 in graph.halfedges else e2).rev
        self.recompute_positions(edge_to_start=e, faces=graph.faces)

    def join_edge(self, h: HalfEdge) -> Vertex:
        """Contract edge ``h`` and place the resulting vertex at the midpoint."""
        new_pos = (h.orig["pos"] + h.dest["pos"]) / 2
        v = super().join_edge(h)
        v["pos"] = new_pos
        return v

    def recompute_positions(self, edge_to_start: HalfEdge | None = None, faces: "set[Face] | None" = None) -> None:
        """Recompute vertex positions from edge lengths and interior angles.

        Performs a heap-ordered BFS over ``faces``, propagating positions
        outward from the edge ``edge_to_start`` (whose two endpoints keep
        their existing positions). Faces are processed in order of the
        smallest accumulated propagation depth, which keeps numerical error
        from compounding on the longest paths first.

        Args:
            edge_to_start: Non-border half-edge whose endpoints anchor the
                computation. Defaults to the longest interior edge.
            faces: Iterable of faces to process. Defaults to all faces.
        """
        if faces is None:
            faces = self.faces
        if len(faces) == 0:
            return  # nothing to do
        if edge_to_start is None:
            # select longest edge
            hs = [h for h in self.halfedges if not h.on_border()]
            lengths = [h["length"] for h in hs]
            edge_to_start = hs[np.argmax(lengths)]
        if edge_to_start.on_border():
            raise ValueError(f"edge_to_start must not be on border, got {edge_to_start}")

        # delete old positions
        vertices = set.union(set(), *(f.vertex_iter() for f in faces)).difference(
            {edge_to_start.orig, edge_to_start.dest}
        )
        for v in vertices:
            if "pos" in v.attributes:
                del v["pos"]

        # compute positions for new vertices, face by face
        yet_to_process = [(0, 0, (edge_to_start.face, edge_to_start.nex))]  # (error, iteration, (face, edge))
        heapq.heapify(yet_to_process)
        processed_faces = set()
        err_dict = {edge_to_start.orig: 0, edge_to_start.dest: 0}
        i = 0
        while yet_to_process:
            err, _, (f, initial) = heapq.heappop(yet_to_process)
            if f in processed_faces:
                continue
            processed_faces.add(f)
            e = initial
            while True:
                assert "pos" in e.orig.attributes, f"{e}"
                assert "pos" in e.pre.orig.attributes, f"{e},{e.pre}"
                if "pos" in e.dest.attributes:
                    # nothing to do here, both positions already determined
                    pass
                else:
                    e.dest["pos"] = self.construct_next_point(
                        a=e.pre.orig["pos"], b=e.orig["pos"], angle=e.pre["in_angle"], length=e["length"]
                    )
                    err_dict[e.dest] = err_dict[e.orig] + 1
                opposite_face = e.rev.face
                if opposite_face in faces and opposite_face not in processed_faces:
                    i += 1
                    heapq.heappush(
                        yet_to_process, (max(err_dict[e.orig], err_dict[e.dest]), i, (opposite_face, e.rev.nex))
                    )
                e = e.nex
                if e is initial:
                    break

    def construct_next_point(self, a: np.ndarray, b: np.ndarray, angle: float, length: float) -> np.ndarray:
        """Construct the point ``c`` such that ``angle(a, b, c) == angle`` and ``|bc| == length``."""
        return self.geometry.construct_next_poly_point(a, b, angle, length)
        # next_angle = angle_to_axis(b - a) + np.pi - angle
        # direction = unit_vector(next_angle)
        # return b + length * direction

    def recompute_lengths_and_angles(self) -> None:
        """Recompute every ``length`` and ``in_angle`` attribute from current positions."""
        for f in self.faces:
            f.recompute_lengths_and_angles(geometry=self.geometry)
        for h in self.border_edges():
            h["length"] = h.rev["length"]

    def get_position_view(
        self,
        vertices: list[Vertex] | None = None,
        return_vertices: bool = True,
        position_key: str = "pos",
    ) -> "np.ndarray | tuple[np.ndarray, list[Vertex]]":
        """Return a single ``(N, d)`` array view of all vertex (and curve) positions.

        Mutating the returned array updates the underlying vertex/curve
        attributes. Useful for vectorized geometric transformations.

        Args:
            vertices: Vertices whose positions to gather. Defaults to all.
            return_vertices: If True, also return the list of vertices.
            position_key: Vertex attribute to read/write.

        Returns:
            ``positions`` array -- a tuple ``(positions, vertices)`` when ``return_vertices`` is True.
        """
        vertices = list(self.vertices) if vertices is None else vertices
        positions = np.stack([v[position_key] for v in vertices])
        curved_hs = [h for h in self.halfedges if "curve_pos" in h]
        if len(curved_hs) > 0:
            curved_pos = np.concatenate([h["curve_pos"] for h in curved_hs])
            positions = np.concatenate([positions, curved_pos])
            # assign curve positions
            i = len(vertices)
            for h in curved_hs:
                j = i + len(h["curve_pos"])
                h["curve_pos"] = positions[i:j]
                i = j

        # assign vertex positions
        for v, p in zip(vertices, positions):
            v[position_key] = p

        if return_vertices:
            return positions, vertices
        else:
            return positions

    def normalize_positions(self) -> None:
        """Center positions at the origin and rescale so the maximum distance is 1."""
        ps, _ = self.get_position_view()
        k = ps.copy()
        k = np.array([complex(*ki) for ki in k])
        k -= np.mean(k)
        k = k / np.max(np.abs(k))
        k = np.stack([k.real, k.imag], axis=-1)
        ps[:] = k
        self.recompute_lengths_and_angles()

    def scale_positions(self, factor: float) -> None:
        """Multiply every vertex position by ``factor`` (Euclidean geometry only)."""
        if self.geometry is not EuclideanGeometry:
            raise NotImplementedError("Scaling only implemented for Euclidean geometry.")
        for v in self.vertices:
            v["pos"] = factor * v["pos"]
        self.recompute_lengths_and_angles()

    def normalize_edge_lengths(self, mode: str = "geometric", factor: float = 1) -> None:
        """Scale the graph so the average edge length equals ``factor``.

        Args:
            mode: One of ``'geometric'``, ``'arithmetic'``, ``'harmonic'``,
                ``'min'``, ``'max'`` -- which mean to use as the reference.
            factor: Desired mean edge length after scaling.
        """
        if self.geometry is not EuclideanGeometry:
            raise NotImplementedError("Scaling only implemented for Euclidean geometry.")

        lengths = np.array([h["length"] for h in self.halfedges])
        if mode == "geometric":
            mean_length = np.exp(np.mean(np.log(lengths)))
        elif mode == "arithmetic":
            mean_length = np.mean(lengths)
        elif mode == "harmonic":
            mean_length = len(lengths) / np.sum(1 / lengths)
        elif mode == "min":
            mean_length = np.min(lengths)
        elif mode == "max":
            mean_length = np.max(lengths)
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected one of 'geometric', 'arithmetic', 'harmonic', 'min', 'max'."
            )
        factor = factor / mean_length
        self.scale_positions(factor)

    def convert_to_euclidean(self) -> None:
        """Project all positions to the Euclidean plane and switch the geometry backend."""
        for v in self.vertices:
            v["pos"] = self.geometry.to_euclidean(v["pos"])
        self.geometry = EuclideanGeometry
        self.recompute_lengths_and_angles()

    def show(
        self,
        render_faces: bool = True,
        render_edges: bool = True,
        render_vertices: bool = True,
        block: bool = True,
        figsize: tuple[float, float] | None = None,
        for_cutting: bool = False,
        filename: str = "output",
        **kwargs,
    ) -> None:
        """Render the graph with Cairo and display it inline (Jupyter) or via Matplotlib.

        The rendered PNG is also written to ``{filename}.png``.
        """
        figsize = (5, 5) if figsize is None else figsize
        import matplotlib.image as mpimg

        from .rendering import CairoRenderer

        render_position_key = "euclidean_pos"
        for v in self.vertices:
            v[render_position_key] = self.geometry.to_euclidean(v["pos"])
        renderer = CairoRenderer(path=filename + ".svg", position_key=render_position_key, **kwargs)
        surface = renderer.render_graph(
            self,
            render_faces=render_faces,
            render_edges=render_edges,
            render_vertices=render_vertices,
            for_cutting=for_cutting,
        )
        filename = filename + ".png"
        surface.write_to_png(filename)

        from IPython.display import Image, display

        display(Image(filename))

    def central_face(self) -> Face:
        """Return the face whose midpoint is closest to the origin (Euclidean only)."""
        if self.geometry is not EuclideanGeometry:
            raise NotImplementedError
        fs = list(self.faces)
        return fs[np.argmin([np.linalg.norm(f.midpoint()) for f in fs])]

    def central_vertex(self) -> Vertex:
        """Return the vertex closest to the origin (Euclidean only)."""
        if self.geometry is not EuclideanGeometry:
            raise NotImplementedError
        vs = list(self.vertices)
        return vs[np.argmin([np.linalg.norm(v["pos"]) for v in vs])]


class EuclideanPositionHEG(GeometricHEG):
    """GeometricHEG specialized for Euclidean geometry with 2D vertex positions.

    Vertices carry ``pos`` attributes (numpy arrays of shape (2,)).  Provides epsilon-based vertex merging and position-aware graph operations.
    """

    def __init__(self, **super_kwargs: object) -> None:
        """Create a Euclidean-geometry half-edge graph."""
        super().__init__(geometry=EuclideanGeometry, **super_kwargs)


# ------------------------------------------------ cyclic graph example ------------------------------------------------


def rotate_by(list_like: "list | tuple", offset: "int | tuple[int, ...]") -> "list | zip":
    """Cyclically rotate ``list_like`` by ``offset`` positions.

    Args:
        list_like: Sequence to rotate.
        offset: Either an integer shift, or an iterable of integer shifts in
            which case the function returns the zip of the corresponding
            rotations.

    Returns:
        The rotated list -- a zip iterator over rotations when ``offset`` is iterable.
    """
    if isinstance(offset, int):
        l = list(list_like)
        return l[offset:] + l[:offset]
    else:
        return zip(*(rotate_by(list_like, off) for off in offset))


def any_element(s):
    """Return an arbitrary element from the iterable ``s``."""
    return next(iter(s))


class CyclicHalfedgeGraph(HalfEdgeGraph):
    """A single-face polygon as a half-edge graph.

    Constructed from a list of vertices, creating one interior face and
    a border.  Useful as a building block for gluing larger graphs.
    """

    def __init__(
        self,
        vs: list[Vertex],
        inner_hs: list[HalfEdge] | None = None,
        outer_hs: list[HalfEdge] | None = None,
        f: Face | None = None,
    ) -> None:
        """Build a single-face cyclic graph from the vertex sequence ``vs``.

        Args:
            vs: Vertices of the polygon, in cyclic order.
            inner_hs: Optional pre-existing inner half-edges (one per vertex).
            outer_hs: Optional pre-existing outer (border) half-edges.
            f: Optional pre-existing :class:`Face` object.
        """
        super(CyclicHalfedgeGraph, self).__init__()

        # init face if necessary
        f = Face() if f is None else f
        assert isinstance(f, Face), f"{type(f)}"

        # init the halfedges, first only with the vertices
        inner_hs = [HalfEdge() for _ in vs] if inner_hs is None else inner_hs
        outer_hs = [HalfEdge() for _ in vs] if outer_hs is None else outer_hs

        # give face reference to a halfedge
        f.any_side = inner_hs[0]

        # orig and dest for inner and outer halfedges
        for h, (orig, dest) in zip(inner_hs, rotate_by(vs, (0, 1))):
            h.orig = orig
            h.dest = dest
        for h, (dest, orig) in zip(outer_hs, rotate_by(vs, (0, 1))):
            h.orig = orig
            h.dest = dest

        # nex and pre and face for inner halfedges
        for h, pre, nex in rotate_by(inner_hs, (0, -1, 1)):
            h.nex = nex
            h.pre = pre
            h.face = f
            h.orig.any_outgoing = h

        # nex and pre for outer halfedges
        for h, pre, nex in rotate_by(outer_hs, (0, 1, -1)):  # note different offsets
            h.nex = nex
            h.pre = pre

        # rev for all halfedges
        for h_inner, h_outer in zip(inner_hs, outer_hs):
            h_inner.rev = h_outer
            h_outer.rev = h_inner

        # finally, add everything to self
        self.add_vertices(vs)
        self.add_face(f)
        self.face = f
        self.add_halfedges(inner_hs)
        self.add_halfedges(outer_hs)


def make_polygon_graph(positions: np.ndarray) -> "EuclideanPositionHEG":
    """Create a single-face Euclidean polygon graph from a sequence of 2D positions.

    Args:
        positions: Iterable of 2D points, in cyclic order around the polygon.

    Returns:
        A :class:`EuclideanPositionHEG` whose only inner face is the polygon.
    """
    vs = [Vertex() for _ in range(len(positions))]
    G = CyclicHalfedgeGraph(vs)
    for v, p in zip(vs, positions):
        v["pos"] = p
    G = EuclideanPositionHEG(other=G)
    return G


class RegularNGon(CyclicHalfedgeGraph, InAngleHEG):
    """A regular *n*-gon as a half-edge graph with positions and angles."""

    def __init__(self, n: int, *super_args: object, **super_kwargs: object) -> None:
        """Construct a regular ``n``-gon with interior angle ``(n-2)/n * pi``."""
        super(RegularNGon, self).__init__(vs=[Vertex() for _ in range(n)], *super_args, **super_kwargs)
        f = any_element(self.faces)
        for e in f.halfedge_iter():
            e["in_angle"] = (n - 2) / n * pi
