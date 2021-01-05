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

    def get_tri(self, h):
        return None

    def generate_graph_and_corners(self, tri):
        graph, (v_map, _, _) = self.graph.copy(deepcopy_attributes=False, return_mappings=True)
        return graph, (v_map[self.v1], v_map[self.vf], v_map[self.v2])
        #return deepcopy((self.graph, (self.v1, self.vf, self.v2)))

    def __call__(self, graph, faces=None, delete_on_border=True, delete_inner_border=False):
        # TODO: add option to copy graph. take care of face mapping

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

        graphs_and_corners = map(self.generate_graph_and_corners, [self.get_tri(h) for h in halfedges])

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
            if v.order() is 2:  # TODO: check this beforehand
                HalfEdgeGraph.join_vertex(graph, v)
        return graph


class GeometricConwayOperator(TopologicalConwayOperator):
    def __init__(self, *super_args, **super_kwargs):
        super(GeometricConwayOperator, self).__init__(*super_args, **super_kwargs)
        # convert euclidean to barycentric coordinates
        to_barycentric = euclidean_to_barycentric_map(np.array([self.v1['pos'], self.vf['pos'], self.v2['pos']]))
        for v in self.graph.vertices:
            v['pos'] = to_barycentric(v['pos'])

    def get_tri(self, h):
        return np.array([h.dest['pos'], h.face.midpoint(), h.orig['pos']])

    def generate_graph_and_corners(self, tri):
        result, corners = super(GeometricConwayOperator, self).generate_graph_and_corners(tri)

        to_euclidean = barycentric_to_euclidean_map(tri)
        for v in result.vertices:
            v['pos'] = to_euclidean(v['pos'])
        return result, corners

    def __call__(self, graph, recompute_lengths_and_angles=True, **kwargs):
        result = super(GeometricConwayOperator, self).__call__(graph, **kwargs)
        if recompute_lengths_and_angles:
            result.recompute_lengths_and_angles()
        return result


def dual_graph():
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


def kis_graph():
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


def join_graph():
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
    nx.add_cycle(G, [v1, v1f, vf, v2f, v2, v12], delete=True)
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
    nx.add_cycle(G, [v1, v1ft, vf, v2ft, v2, v21t, v12t], delete=True)
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
    nx.add_cycle(G, [v1, vf, v2, ve], delete=True)
    G.add_edges_from([[g, ve], [g, v1], [g, vf]])
    G.add_node(ve, join=True)

    # construct EHEG and ConwayOperator
    heg, v_lookup = EHEG_from_nx(G, return_v_lookup=True)
    return GeometricConwayOperator(heg, *(v_lookup[v] for v in [v1, vf, v2]))


def starify_graph(t=1/3):
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


def twist_rotate_graph(t=1/2):
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


def loft_graph(t=1/2):
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


def chamfer_graph(t=1/2):
    assert t < 1
    result = loft_graph(t)
    e = [e for e in result.v1.outgoing_iter() if e.dest is result.v2][0]
    e.rev.attributes['delete'] = True
    e.attributes['delete'] = True
    return result
