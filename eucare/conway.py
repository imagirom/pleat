"""Conway topological operators for transforming tilings."""
from __future__ import annotations

from copy import copy

import networkx as nx
import numpy as np

from .base import euclidean_to_barycentric_map
from .conversions import EHEG_from_nx
from .half import Face, GeometricHEG, HalfEdge, HalfEdgeGraph, Vertex
from .utils import invert_mapping


class TopologicalConwayOperator:
    """Apply a Conway operator to a half-edge graph by substituting a fundamental domain into each face triangle.

    The fundamental domain is a small half-edge graph with three marked vertices (v1, vf, v2)
    corresponding to a vertex, face center, and adjacent vertex of each triangle.
    """

    def __init__(self, graph: HalfEdgeGraph, v1: Vertex, vf: Vertex, v2: Vertex) -> None:
        self.graph = graph
        self.v1 = v1
        self.vf = vf
        self.v2 = v2
        assert all(v in graph.vertices for v in (v1, vf, v2))

    def show(self) -> None:
        """Render the fundamental domain graph for visualization."""
        self.graph.show(scale=300, line_width=0.03, render_faces=False)

    def get_tri(self, h):
        """Return the triangle for halfedge h, or None for purely topological operators."""
        return None

    def generate_graph_and_corners(self, tri, h=None):
        """Return a copy of the fundamental domain graph and its three corner vertices."""
        graph, (v_map, _, _) = self.graph.copy(deepcopy_attributes=False, return_mappings=True)
        if h is not None:
            for v in graph.vertices:
                if not v.on_border():  # todo: also assign 'pre_conway' to border vertices, as a set
                    v['pre_conway'] = h
        return graph, (v_map[self.v1], v_map[self.vf], v_map[self.v2])
        #return deepcopy((self.graph, (self.v1, self.vf, self.v2)))

    def __call__(self, graph, faces=None, delete_on_border=True, delete_inner_border=False, copy_graph=False):
        """Apply the operator to graph, optionally restricted to given faces.

        If copy_graph is True, operate on a copy and preserve pre_conway references.
        """
        if copy_graph:
            graph, (v_map, h_map, f_map) = graph.copy(return_mappings=True)
            v_map, h_map, f_map = [invert_mapping(m) for m in (v_map, h_map, f_map)]
            obj_map = dict()
            obj_map.update(v_map)
            obj_map.update(h_map)
            obj_map.update(f_map)
            for obj in obj_map.keys():
                if obj is not None and 'pre_conway' in obj:
                    del obj['pre_conway']

        # apply the operator to a set of halfedges in a graph
        assert isinstance(graph, HalfEdgeGraph)
        if faces is None:
            faces = graph.faces
        halfedges = [h for f in faces for h in f.halfedge_iter()]
        assert all(isinstance(h, HalfEdge) for h in halfedges)
        affected_faces = {h.face for h in halfedges}
        assert None not in affected_faces, 'Cannot apply Conway operator to boundary edge'  # Or can we?
        old_halfedges = frozenset(graph.halfedges)
        v1_out_lookup = dict()
        v2_out_lookup = dict()
        vf_lookup = dict()
        vf_set = set()

        graphs_and_corners = [self.generate_graph_and_corners(self.get_tri(h), h) for h in halfedges]

        for gc, h in zip(graphs_and_corners, halfedges):
            orig_face = h.face
            con_graph, (v1, vf, v2) = gc

            # add reference to old face/vertex to new face/vertex
            for new_vertex, old_obj in [(v1, h.dest), (vf, h.face), (v2, h.orig)]:
                if new_vertex.attributes.get('delete', False):
                    new_vertex.get_outgoing_border().rev.face['pre_conway'] = old_obj
                else:
                    new_vertex['pre_conway'] = old_obj

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
                    for k in h.face.halfedge_iter():
                        if (not delete_inner_border) or h.rev.on_border():
                            k['delete'] = False
                            k.rev['delete'] = False
                        k.rev['border_delete'] = True

                if h.rev.on_border():
                    graph.delete_face(h.face)
                else:
                    HalfEdgeGraph.delete_edge(graph, h)
                # TODO: if on border, delete. else, do not delete adjacent. for now: delete in neither case


        # TODO: keep a record of which edges to delete in the end..
        # and a record of all affected vertices to update angles

        to_delete = set()
        to_process = copy(graph.halfedges) # this is bad for performance: make everything work locally!
        to_keep = set()
        while to_process:
            h = to_process.pop()
            to_process.remove(h.rev)
            if h.attributes.get('delete', False):
                # only delete edges if their reverse also wants to be deleted
                if h.rev.attributes.get('delete', False):
                    to_delete.update({h, h.rev})
                    continue
                else:
                    pass
                    #del h.attributes['delete']
            to_keep.update({h, h.rev})

        assert not to_keep.intersection(to_delete)
        assert to_keep.union(to_delete) == graph.halfedges, \
            f'{graph.halfedges.difference(to_keep.union(to_delete))}, {to_keep.union(to_delete).difference(graph.halfedges)}'

        faces_to_keep = set()
        faces_to_maybe_remove = set()
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

        graph.vertices.difference_update(
            [v for v in (h.orig for h in to_delete) if v.any_outgoing in to_delete])

        if delete_on_border:
            for e in list(graph.border_edges()):
                if e in graph.halfedges:
                    if e.rev.attributes.get('border_delete', False):
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
            if 'border_delete' in e.attributes:
                del e['border_delete']

        to_join = {v for v in graph.vertices if v.attributes.get('join', False)}
        for v in to_join:
            if v.order() == 2:  # TODO: check this beforehand
                HalfEdgeGraph.join_vertex(graph, v)

        if copy_graph:
            for objs in (graph.vertices, graph.halfedges, graph.faces):
                for obj in objs:
                    if 'pre_conway' in obj.attributes:
                        obj['pre_conway'] = obj_map[obj['pre_conway']]
        return graph


class GeometricConwayOperator(TopologicalConwayOperator):
    """Conway operator that assigns new vertex positions using barycentric coordinate interpolation."""

    def __init__(self, *super_args, **super_kwargs):
        super(GeometricConwayOperator, self).__init__(*super_args, **super_kwargs)
        # convert euclidean to barycentric coordinates
        to_barycentric = euclidean_to_barycentric_map(np.array([self.v1['pos'], self.vf['pos'], self.v2['pos']]))
        for v in self.graph.vertices:
            v['pos'] = to_barycentric(v['pos'])
        self.geometry = None

    def get_tri(self, h):
        """Return the triangle (dest, face midpoint, orig) for halfedge h."""
        midpoint = h.face.get('midpoint', self.geometry.center_of_mass(np.stack([v['pos'] for v in h.face.vertex_iter()])))
        return np.array([h.dest['pos'], midpoint, h.orig['pos']])

    def generate_graph_and_corners(self, tri, h=None):
        """Return a copy of the domain with vertex positions mapped from barycentric to Euclidean coordinates."""
        result, corners = super(GeometricConwayOperator, self).generate_graph_and_corners(tri, h)

        to_euclidean = self.geometry.barycentric_to_euclidean_map(tri)
        # this could be vectorized
        for v in result.vertices:
            v['pos'] = to_euclidean(v['pos'])
        return result, corners

    def __call__(self, graph: GeometricHEG, recompute_lengths_and_angles=True, **kwargs):
        """Apply the geometric operator to a GeometricHEG, optionally recomputing lengths and angles."""
        assert isinstance(graph, GeometricHEG)
        self.geometry = graph.geometry
        result = super().__call__(graph, **kwargs)
        if recompute_lengths_and_angles:
            result.recompute_lengths_and_angles()
        return result


def dual_graph() -> GeometricConwayOperator:
    """Construct the Conway dual operator."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    ve = (0, 0)
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2, ve], delete=True)
    G.add_nodes_from([v1, v2], delete=True)
    G.add_nodes_from([ve], join=True)
    G.add_edge(ve, vf)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def kis_graph() -> GeometricConwayOperator:
    """Construct the Conway kis (raising) operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def join_graph() -> GeometricConwayOperator:
    """Construct the Conway join operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2])
    G.add_edge(v2, v1, delete=True)
    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def ambo_graph() -> GeometricConwayOperator:
    """Construct the Conway ambo (rectification) operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, 0)
    v1f = (1 / 2, -1 / 2)
    v2f = (1 / 2, 1 / 2)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v12], delete=True)
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_edges_from([[v12, v1f], [v12, v2f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def goldberg2_graph() -> GeometricConwayOperator:
    """Construct the Goldberg-2 subdivision operator."""
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, 0)
    v1f = (1 / 2, -1 / 2)
    v2f = (1 / 2, 1 / 2)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v12], delete=True)
    del G.edges[v1, v12]['delete']
    del G.edges[v12, v2]['delete']
    G.add_nodes_from([vf], delete=True)
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_edges_from([[v12, v1f], [v12, v2f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def truncate_graph(t: float = 1 / 2) -> GeometricConwayOperator:
    """Construct the Conway truncate operator with cut depth t."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12t = (0, -1 + t)
    v1ft = (t / 2, -1 + t / 2)
    v21t = (0, 1 - t)
    v2ft = (t / 2, 1 - t / 2)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1ft, vf, v2ft, v2, v21t, v12t], delete=True)
    del G.edges[v12t, v21t]['delete']
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v1ft, v2ft], join=True)
    G.add_edges_from([[v12t, v1ft], [v21t, v2ft]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def gyro_graph(g: tuple[float, float] = (1 / 4, -1 / 4)) -> GeometricConwayOperator:
    """Construct the Conway gyro operator with snub point position g."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    ve = (0, 0)
    G = nx.Graph()
    nx.add_cycle(G, [v1, vf, v2, ve], delete=True)
    G.add_edges_from([[g, ve], [g, v1], [g, vf]])
    G.add_node(ve, join=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def starify_graph(t: float = 1/3) -> GeometricConwayOperator:
    """Construct the starify operator with parameter t controlling star point depth."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v0 = (0, 0)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v0], delete='True')
    G.add_nodes_from([v0], join=True)
    G.add_edges_from([[v0, v2f], [v2f, v1f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def alternating_flagstone_graph(t: float = 1/3) -> GeometricConwayOperator:
    """Construct the alternating flagstone operator with parameter t."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v0 = (0, 0)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v0], delete=True)
    del G.edges[v1, v1f]['delete']
    del G.edges[v2, v2f]['delete']
    G.add_edge(v2f, v1, color_key=(1, 0, 0))
    G.add_nodes_from([v0], join=True)
    G.add_nodes_from([vf], delete=True)
    G.add_edges_from([[v0, v2f], [v2f, v1f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def twist_rotate_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the twist-rotate operator with parameter t."""
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12t = (0, -1 + t)
    v1ft = (t, -1 + t)
    v21t = (0, 1 - t)
    v2ft = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1ft, vf, v2ft, v2, v21t, v12t], delete=True)
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v12t, v21t], join=True)
    G.add_edges_from([[v12t, v1ft], [v21t, v2ft], [v1ft, v2ft]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    v_lookup[vf].get_outgoing_border().rev.face['twistrotate'] = True
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def loft_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the loft operator with edge offset parameter t (must be < 1)."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v1ft = (t, -1 + t)
    v2ft = (t, 1 - t)
    G = nx.Graph()
    nx.add_cycle(G, [v1, v1ft, v2ft, v2])
    G.add_nodes_from([vf], delete=True)
    G.add_edges_from([[v1ft, vf], [vf, v2ft]], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def lace_graph(t: float = 1/2, join: bool = False) -> GeometricConwayOperator:
    """Construct the lace operator with offset t. If join is True, merge the v1-v2 edge."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    vc = (t, 0)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)

    G = nx.Graph()
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2], delete=True)
    if not join:
        del G.edges[v1, v2]['delete']

    G.add_edges_from([[vc, v1], [vc, v2], [vc, v1f], [vc, v2f]])
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_nodes_from([vf], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def expand_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the Conway expand operator with offset parameter t (must be < 1)."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, -1 + t)
    v21 = (0, 1 - t)
    v1f = (t, -1 + t)
    v2f = (t, 1 - t)

    G = nx.Graph()
    nx.add_cycle(G, [v2, v21, v12, v1, v1f, vf, v2f], delete=True)

    G.add_edges_from([[v1f, v12], [v2f, v21], [v1f, v2f]])
    G.add_nodes_from([v12, v21, v1f, v2f], join=True)
    G.add_nodes_from([vf], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def flagstone_pvitelli_graph(t: float = 1/4) -> GeometricConwayOperator:
    """Construct the Pvitelli flagstone operator with parameter t (must be < 1)."""
    assert t < 1
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    vL1 = (0, -1 + t)
    vL2 = (0, 1 - t)
    vL1f = (t/2, -1 + t/2)
    vL2f = (t/2, 1 - t/2)
    vV1 = (t, -1 + t)
    vV2 = (t, 1 - t)
    vN1 = (t/2, -1 + 3/2*t)
    vN2 = (t/2, 1 - 3/2*t)
    vN1e = (0, -1 + 3/2*t)
    vN2e = (0, 1 - 3/2*t)

    G = nx.Graph()
    nx.add_cycle(G, [v2, vL2, vN2e, vN1e, vL1, v1, vL1f, vV1, vf, vV2, vL2f], delete=True)
    nx.add_cycle(G, [vL2, vN2, vN1, vL1, vV1, vV2])
    G.add_edges_from([[vL1f, vL1], [vL2f, vL2], [vN1e, vN1], [vN2e, vN2], [vN1, vV1], [vN2, vV2]])
    G.add_nodes_from([vL1f, vL2f, vN1e, vN2e], join=True)

    # label the points which will be joined that are closer to v1
    G.add_nodes_from([vL1], label='L')
    G.add_nodes_from([vV1], label='V')
    # label both copies of the inner node
    G.add_nodes_from([vN1], label='N1')
    G.add_nodes_from([vN2], label='N2')

    G.add_nodes_from([v1, v2, vf], delete=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)

    v_lookup[vf].get_outgoing_border().rev.face['is_central_polygon'] = True

    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def chamfer_graph(t: float = 1/2) -> GeometricConwayOperator:
    """Construct the Conway chamfer operator, derived from loft with the v1-v2 edge deleted."""
    assert t < 1
    result = loft_graph(t)
    e = [e for e in result.v1.outgoing_iter() if e.dest is result.v2][0]
    e.rev.attributes['delete'] = True
    e.attributes['delete'] = True
    return result
