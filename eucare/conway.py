from .half import *
from .conversions import EHEG_from_nx
from .base import barycentric_to_euclidean_map, euclidean_to_barycentric_map
from copy import deepcopy


class TopologicalConwayOperator:
    def __init__(self, graph, v1, vf, v2):
        self.graph = graph
        self.v1 = v1
        self.vf = vf
        self.v2 = v2
        assert all(v in graph.vertices for v in (v1, vf, v2))

    def show(self):
        self.graph.show(scale=300, line_width=0.03, render_faces=False)

    def generate_graph_and_corners(self, h):
        return deepcopy((self.graph, (self.v1, self.vf, self.v2)))

    def __call__(self, graph, faces=None, delete_on_border=False):
        if delete_on_border:
            raise NotImplementedError #TODO: get rid of resulting dangling edges
        # apply the operator to a set of halfedges in a graph
        assert isinstance(graph, HalfEdgeGraph)
        if faces is None:
            faces = graph.faces
        halfedges = [h for f in faces for h in f.halfedge_iter()]
        assert all(isinstance(h, HalfEdge) for h in halfedges)
        affected_faces = {h.face for h in halfedges}
        assert None not in affected_faces, f'Cannot apply Conway operator to boundary edge'  # Or can we?
        old_halfedges = frozenset(graph.halfedges)
        v1_out_lookup = dict()
        v2_out_lookup = dict()
        vf_lookup = dict()
        vf_set = set()
        for h in halfedges:
            # glue v1, v2 to h.dest, h.orig
            orig_face = h.face
            con_graph, (v1, vf, v2) = self.generate_graph_and_corners(h)
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
                if not delete_on_border or not h.rev.on_border():
                    for k in h.face.halfedge_iter():
                         k['delete'] = False
                         k.rev['delete'] = False
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
                    del h.attributes['delete']
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

        to_join = {v for v in graph.vertices if v.attributes.get('join', False)}
        for v in to_join:
            if v.order() is 2:  # TODO: check this beforehand
                HalfEdgeGraph.join_vertex(graph, v)

        return graph


class GeometricConwayOperator(TopologicalConwayOperator):
    def __init__(self, *super_args, **super_kwargs):
        super(GeometricConwayOperator, self).__init__(*super_args, **super_kwargs)
        # convert euclidean to barycentric coordinates
        self.to_barycentric = euclidean_to_barycentric_map(np.array([self.v1['pos'], self.vf['pos'], self.v2['pos']]))

    def generate_graph_and_corners(self, h):
        result, corners = super(GeometricConwayOperator, self).generate_graph_and_corners(h)
        tri = np.array([h.dest['pos'], h.face.midpoint(), h.orig['pos']])
        to_euclidean = barycentric_to_euclidean_map(tri)
        for v in result.vertices:
            v['pos'] = to_euclidean(self.to_barycentric(np.array(v['pos'])))
        return result, corners


def dual_graph():
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    ve = (0, 0)
    G = nx.Graph()
    G.add_cycle([v1, vf, v2, ve], delete=True)
    G.add_nodes_from([v1, v2], delete=True)
    G.add_nodes_from([ve], join=True)
    G.add_edge(ve, vf)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def ambo_graph():
    # define vertex positions
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12 = (0, 0)
    v1f = (1 / 2, -1 / 2)
    v2f = (1 / 2, 1 / 2)
    # construct nx graph with delete and join attributes
    G = nx.Graph()
    G.add_cycle([v1, v1f, vf, v2f, v2, v12], delete=True)
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v1f, v2f], join=True)
    G.add_edges_from([[v12, v1f], [v12, v2f]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def truncate_graph(t=1 / 2):
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    v12t = (0, -1 + t)
    v1ft = (t / 2, -1 + t / 2)
    v21t = (0, 1 - t)
    v2ft = (t / 2, 1 - t / 2)
    G = nx.Graph()
    G.add_cycle([v1, v1ft, vf, v2ft, v2, v21t, v12t], delete=True)
    del G.edges[v12t, v21t]['delete']
    G.add_nodes_from([v1, vf, v2], delete=True)
    G.add_nodes_from([v1ft, v2ft], join=True)
    G.add_edges_from([[v12t, v1ft], [v21t, v2ft]])

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def gyro_graph(g=(1 / 4, -1 / 4)):
    v1 = (0, -1)
    vf = (1, 0)
    v2 = (0, 1)
    ve = (0, 0)
    G = nx.Graph()
    G.add_cycle([v1, vf, v2, ve], delete=True)
    G.add_edges_from([[g, ve], [g, v1], [g, vf]])
    G.add_node(ve, join=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))
