"""Conway operator classes: topological substitution and geometric variant."""

from __future__ import annotations

from copy import copy
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from eucare.rendering import Rendering

from ..base import euclidean_to_barycentric_map
from ..geometries.base import Geometry
from ..half import Face, GeometricHEG, HalfEdge, HalfEdgeGraph, Vertex
from ..utils import invert_mapping


class TopologicalConwayOperator:
    """Apply a Conway operator to a half-edge graph by substituting a fundamental domain into each face triangle.

    The fundamental domain is a small half-edge graph with three marked vertices (v1, vf, v2)
    corresponding to a vertex, face center, and adjacent vertex of each triangle.
    """

    def __init__(self, graph: HalfEdgeGraph, v1: Vertex, vf: Vertex, v2: Vertex) -> None:
        """Store the fundamental domain and its three corner vertices.

        Args:
            graph: Half-edge graph encoding the fundamental domain.
            v1: Corner mapped to ``h.dest`` of each triangle.
            vf: Corner mapped to the face midpoint of each triangle.
            v2: Corner mapped to ``h.orig`` of each triangle.
        """
        self.graph = graph
        self.v1 = v1
        self.vf = vf
        self.v2 = v2
        assert all(v in graph.vertices for v in (v1, vf, v2))

    def show(self) -> None:
        """Render the fundamental domain graph for visualization."""
        cast(GeometricHEG, self.graph).show(scale=300, line_width=0.03, render_faces=False)

    def get_tri(self, h: HalfEdge) -> NDArray[np.floating[Any]] | None:
        """Return the triangle for half-edge ``h``, or None for purely topological operators."""
        return None

    def generate_graph_and_corners(
        self, tri: NDArray[np.floating[Any]] | None, h: HalfEdge | None = None
    ) -> tuple[HalfEdgeGraph, tuple[Vertex, Vertex, Vertex]]:
        """Return a copy of the fundamental-domain graph and its three corner vertices.

        Args:
            tri: Optional reference triangle for geometric subclasses.
            h: Optional original half-edge; if given, every interior vertex of
                the copy gets ``pre_conway = h`` for traceability.

        Returns:
            Tuple ``(graph_copy, (v1, vf, v2))``.
        """
        graph, (v_map, _, _) = self.graph.copy(deepcopy_attributes=False, return_mappings=True)
        if h is not None:
            for v in graph.vertices:
                if not v.on_border():  # todo: also assign 'pre_conway' to border vertices, as a set
                    v["pre_conway"] = h
        return graph, (v_map[self.v1], v_map[self.vf], v_map[self.v2])
        # return deepcopy((self.graph, (self.v1, self.vf, self.v2)))

    def __call__(
        self,
        graph: HalfEdgeGraph,
        faces: "set[Face] | None" = None,
        delete_on_border: bool = True,
        delete_inner_border: bool = False,
        copy_graph: bool = False,
    ) -> HalfEdgeGraph:
        """Apply the operator to ``graph``, optionally restricted to ``faces``.

        Args:
            graph: Input half-edge graph; mutated in place unless ``copy_graph``.
            faces: Faces to apply the operator to. Defaults to all faces.
            delete_on_border: If True, delete faces whose original border edge
                was marked for deletion.
            delete_inner_border: If True, also clear the ``delete`` flag on
                inner border edges.
            copy_graph: If True, operate on a deep-ish copy and preserve
                ``pre_conway`` references back to the original objects.

        Returns:
            The (possibly copied) graph after substitution.
        """
        obj_map: dict[Any, Any] = dict()
        if copy_graph:
            graph, (v_map_raw, h_map_raw, f_map_raw) = graph.copy(return_mappings=True)
            v_map = invert_mapping(v_map_raw)
            h_map = invert_mapping(h_map_raw)
            f_map = invert_mapping(f_map_raw)
            obj_map.update(v_map)
            obj_map.update(h_map)
            obj_map.update(f_map)
            for obj in obj_map.keys():
                if obj is not None and "pre_conway" in obj:
                    del obj["pre_conway"]

        # apply the operator to a set of halfedges in a graph
        assert isinstance(graph, HalfEdgeGraph)
        if faces is None:
            faces = graph.faces
        halfedges = [h for f in faces for h in f.halfedge_iter()]
        assert all(isinstance(h, HalfEdge) for h in halfedges)
        affected_faces = {h.face for h in halfedges}
        assert None not in affected_faces, "Cannot apply Conway operator to boundary edge"  # Or can we?
        old_halfedges = frozenset(graph.halfedges)
        v1_out_lookup: dict[HalfEdge, HalfEdge] = dict()
        v2_out_lookup: dict[HalfEdge, HalfEdge] = dict()
        vf_lookup: dict[HalfEdge, Vertex] = dict()
        vf_set: set[Vertex] = set()

        graphs_and_corners = [self.generate_graph_and_corners(self.get_tri(h), h) for h in halfedges]

        for gc, h in zip(graphs_and_corners, halfedges):
            orig_face = h.face
            assert orig_face is not None
            con_graph, (v1, vf, v2) = gc

            # add reference to old face/vertex to new face/vertex
            for new_vertex, old_obj in [(v1, h.dest), (vf, h.face), (v2, h.orig)]:
                if new_vertex.attributes.get("delete", False):
                    border_face = new_vertex.get_outgoing_border().rev.face
                    assert border_face is not None
                    border_face["pre_conway"] = old_obj
                else:
                    new_vertex["pre_conway"] = old_obj

            # glue v1, v2 to h.dest, h.orig
            graph.add_graph(con_graph)
            v1_out = v1.get_outgoing_border()
            v2_out = v2.get_outgoing_border()
            graph.glue_v2v(v1_out=h.nex, v2_out=v1_out)
            graph.glue_v2v(v1_out=h, v2_out=v2_out)
            inbetween_face = Face(any_side=h)
            h.face = inbetween_face
            k = v2_out
            orig_face.any_side = k
            while k.orig is not h.dest:
                k.face = orig_face
                k = k.nex
            k = v1_out
            while k.orig is not h.orig:
                k.face = inbetween_face
                k = k.nex
            graph.add_face(inbetween_face)

            v1_out_lookup[h] = v1_out
            v2_out_lookup[h] = v2_out
            vf_lookup[h] = vf
            vf_set.add(vf)

        # handle the insides of faces
        for h in halfedges:
            v2_out = v2_out_lookup[h]
            if v2_out.pre in old_halfedges:
                # v2_out.pre was not in the set of halfedges that the operator was applied to
                raise NotImplementedError
            current = v2_out
            while current.dest not in vf_set:
                next = current.nex
                HalfEdgeGraph.glue_e2e(graph, current, current.pre)
                current = next
            HalfEdgeGraph.glue_e2e(graph, current, current.pre)

        # handle the original edges
        to_process = copy(halfedges)
        while to_process:
            h = to_process.pop()
            if h.rev in halfedges:
                # delete old edge, then zip
                to_process.remove(h.rev)
                HalfEdgeGraph.delete_edge(graph, h)
                current = v1_out_lookup[h]
                f = current.face
                while f in graph.faces:
                    next = current.nex
                    HalfEdgeGraph.glue_e2e(graph, current, current.pre)
                    current = next
            else:
                if True:  # not delete_on_border or not h.rev.on_border(): #Fixme
                    assert h.face is not None
                    for k in h.face.halfedge_iter():
                        if (not delete_inner_border) or h.rev.on_border():
                            k["delete"] = False
                            k.rev["delete"] = False
                        k.rev["border_delete"] = True

                if h.rev.on_border():
                    assert h.face is not None
                    graph.delete_face(h.face)
                else:
                    HalfEdgeGraph.delete_edge(graph, h)
                # TODO: if on border, delete. else, do not delete adjacent. for now: delete in neither case

        # TODO: keep a record of which edges to delete in the end..
        # and a record of all affected vertices to update angles

        to_delete: set[HalfEdge] = set()
        to_process_set: set[HalfEdge] = copy(
            graph.halfedges
        )  # this is bad for performance: make everything work locally!
        to_keep: set[HalfEdge] = set()
        while to_process_set:
            h = to_process_set.pop()
            to_process_set.remove(h.rev)
            if h.attributes.get("delete", False):
                # only delete edges if their reverse also wants to be deleted
                if h.rev.attributes.get("delete", False):
                    to_delete.update({h, h.rev})
                    continue
                else:
                    pass
                    # del h.attributes['delete']
            to_keep.update({h, h.rev})

        assert not to_keep.intersection(to_delete)
        assert (
            to_keep.union(to_delete) == graph.halfedges
        ), f"{graph.halfedges.difference(to_keep.union(to_delete))}, {to_keep.union(to_delete).difference(graph.halfedges)}"

        faces_to_keep: set[Face] = set()
        faces_to_maybe_remove: set[Face | None] = set()
        while to_keep:
            # find the new faces
            h = to_keep.pop()
            initial = h
            f = h.face  # what about border?
            if f is not None:
                f.any_side = h
                faces_to_keep.add(f)
            while True:  # walk around the new face until cycle closes
                nex = h.nex
                while nex in to_delete:
                    nex = nex.rev.nex
                h.nex = nex
                nex.pre = h
                h.dest.any_outgoing = nex
                if nex is initial:
                    break
                else:
                    faces_to_maybe_remove.add(nex.face)
                    nex.face = f
                    to_keep.remove(nex)
                h = nex

        graph.halfedges.difference_update(to_delete)
        graph.faces.difference_update(faces_to_maybe_remove.difference(faces_to_keep))

        graph.vertices.difference_update([v for v in (h.orig for h in to_delete) if v.any_outgoing in to_delete])

        if delete_on_border:
            for e in list(graph.border_edges()):
                if e in graph.halfedges:
                    if e.rev.attributes.get("border_delete", False):
                        assert e.rev.face is not None
                        graph.delete_face(e.rev.face)
            # delete dangling faces
            while True:
                deleted_any = False
                for f in list(graph.faces):
                    if not any(f.face_iter()):
                        graph.delete_face(f)
                        deleted_any = True
                if not deleted_any:
                    break
        for e in graph.halfedges:
            if "border_delete" in e.attributes:
                del e["border_delete"]

        to_join = {v for v in graph.vertices if v.attributes.get("join", False)}
        for v in to_join:
            if v.order() == 2:  # TODO: check this beforehand
                HalfEdgeGraph.join_vertex(graph, v)

        if copy_graph:
            for objs in (graph.vertices, graph.halfedges, graph.faces):
                for obj in objs:
                    if "pre_conway" in obj.attributes:
                        obj["pre_conway"] = obj_map[obj["pre_conway"]]
        return graph


class GeometricConwayOperator(TopologicalConwayOperator):
    """Conway operator that assigns new vertex positions using barycentric coordinate interpolation."""

    graph: GeometricHEG
    geometry: type[Geometry] | None

    def __init__(self, graph: GeometricHEG, v1: Vertex, vf: Vertex, v2: Vertex) -> None:
        """Store the fundamental domain and convert its positions to barycentric coordinates.

        After construction, every vertex in ``self.graph`` carries barycentric
        coordinates relative to the corner triangle ``(v1, vf, v2)``. They are
        re-projected to Euclidean coordinates per target triangle inside
        :meth:`generate_graph_and_corners`.
        """
        super(GeometricConwayOperator, self).__init__(graph, v1, vf, v2)
        # convert euclidean to barycentric coordinates
        to_barycentric = euclidean_to_barycentric_map(np.array([self.v1["pos"], self.vf["pos"], self.v2["pos"]]))
        for v in self.graph.vertices:
            v["pos"] = to_barycentric(v["pos"])
        self.geometry = None

    def get_tri(self, h: HalfEdge) -> NDArray[np.floating[Any]]:
        """Return the triangle ``(h.dest, face midpoint, h.orig)`` for half-edge ``h``."""
        assert h.face is not None
        assert self.geometry is not None
        midpoint = h.face.get(
            "midpoint", self.geometry.center_of_mass(np.stack([v["pos"] for v in h.face.vertex_iter()]))
        )
        return np.array([h.dest["pos"], midpoint, h.orig["pos"]])

    def generate_graph_and_corners(
        self, tri: NDArray[np.floating[Any]] | None, h: HalfEdge | None = None
    ) -> tuple[HalfEdgeGraph, tuple[Vertex, Vertex, Vertex]]:
        """Return a copy of the domain with positions mapped from barycentric to Euclidean coordinates."""
        result, corners = super(GeometricConwayOperator, self).generate_graph_and_corners(tri, h)

        assert tri is not None
        assert self.geometry is not None
        to_euclidean = self.geometry.barycentric_to_euclidean_map(tri)
        # this could be vectorized
        for v in result.vertices:
            v["pos"] = to_euclidean(v["pos"])
        return result, corners

    def __call__(  # type: ignore[override]
        self,
        graph: GeometricHEG,
        recompute_lengths_and_angles: bool = True,
        **kwargs: Any,
    ) -> GeometricHEG:
        """Apply the geometric operator to ``graph``.

        Args:
            graph: Geometric half-edge graph to operate on.
            recompute_lengths_and_angles: If True, recompute edge lengths and
                interior angles after substitution.
            **kwargs: Forwarded to :meth:`TopologicalConwayOperator.__call__`.

        Returns:
            The transformed graph.
        """
        assert isinstance(graph, GeometricHEG)
        self.geometry = graph.geometry
        result = super().__call__(graph, **kwargs)
        assert isinstance(result, GeometricHEG)
        if recompute_lengths_and_angles:
            result.recompute_lengths_and_angles()
        return result

    #: Canonical reference triangle for visualizing the fundamental domain.
    #: Matches the ``v1, vf, v2`` layout used by every factory in :mod:`eucare.conway.factories`.
    _SHOW_REFERENCE_TRIANGLE: NDArray[np.floating[Any]] = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    def get_fundamental_domain_graph_to_render(
        self,
        delete_color: tuple[float, float, float] = (0.85, 0.15, 0.15),
        join_color: tuple[float, float, float] = (0.15, 0.65, 0.20),
        keep_color: tuple[float, float, float] = (0.30, 0.30, 0.30),
    ) -> GeometricHEG:
        """Build a copy of the fundamental-domain graph, colouring elements by their role.

        Vertices and edges flagged as ``delete`` are coloured *delete_color*
        (default: red), and edges become dashed via the existing renderer behaviour;
        vertices flagged as ``join`` are coloured *join_color* (default: green); everything else uses *keep_color*.

        Args:
            delete_color: RGB triple for delete-marked elements.
            join_color: RGB triple for join-marked vertices.
            keep_color: RGB triple for retained elements.

        Returns:
            A Euclidean-projected copy of the graph with ``color_key`` attributes set, ready to render.
        """
        from ..half import EuclideanGeometry

        # Project barycentric vertex positions back to a canonical Euclidean triangle.
        graph_copy_base, (_, _, _) = self.graph.copy(deepcopy_attributes=False, return_mappings=True)
        graph_copy = cast(GeometricHEG, graph_copy_base)
        graph_copy.geometry = EuclideanGeometry
        to_euclidean = EuclideanGeometry.barycentric_to_euclidean_map(self._SHOW_REFERENCE_TRIANGLE)
        for v in graph_copy.vertices:
            v["pos"] = to_euclidean(v["pos"])

        # Assign colours based on role.
        for v in graph_copy.vertices:
            if v.attributes.get("delete", False):
                v["color_key"] = delete_color
            elif v.attributes.get("join", False):
                v["color_key"] = join_color
            else:
                v["color_key"] = keep_color
        for h in graph_copy.halfedges:
            if h.attributes.get("delete", False) or h.rev.attributes.get("delete", False):
                h["color_key"] = delete_color
            elif "color_key" not in h.attributes:
                h["color_key"] = keep_color
        return graph_copy

    def render(self, **show_kwargs: Any) -> Rendering:
        """Render the fundamental domain graph with styling from :meth:`get_fundamental_domain_graph_to_render`."""

        fundamental_domain_graph = self.get_fundamental_domain_graph_to_render()
        kwargs: dict[str, Any] = dict(
            line_width="50%",
            face_inset=0,
        )
        kwargs.update(show_kwargs)
        return fundamental_domain_graph.render(**kwargs)

    def show(self, **kwargs: Any) -> None:
        """Render the fundamental domain graph with styling from :meth:`get_fundamental_domain_graph_to_render` and display it."""
        rendering = self.render(**kwargs)
        rendering.show()
