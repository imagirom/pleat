from .half import EuclideanPositionHEG, Vertex, HalfEdge, Face, rotate_by
from .base import angle_to_axis, signed_area
import numpy as np
from copy import copy


def EHEG_from_edgelist(pts, edges):
    raise NotImplementedError


def EHEG_from_nx(nxg, positions=None, return_v_lookup=False):
    assert not nxg.is_directed()
    if positions is None:
        positions = {n: np.array(n) for n in nxg.nodes()}
    assert isinstance(positions, dict)
    result = EuclideanPositionHEG()
    v_lookup = dict()
    for n, attrs in nxg.nodes().data():
        v = Vertex()
        for key, value in attrs.items():
            v[key] = value
        v['pos'] = positions[n]
        v_lookup[n] = v
    result.add_vertices(v_lookup.values())
    h_lookup = dict()
    # orig, dest
    for n in nxg.nodes():
        v = v_lookup[n]
        h_lookup[v] = dict()
        for m in nxg[n]:
            w = v_lookup[m]
            h = HalfEdge(orig=v, dest=w)
            for key, value in nxg[n][m].items():
                h[key] = value
            h_lookup[v][w] = h
            v.any_outgoing = h
        result.add_halfedges(h_lookup[v].values())
    # rev
    for v in h_lookup:
        for w in h_lookup[v]:
            h_lookup[v][w].rev = h_lookup[w][v]
    # nex, pre
    for v in h_lookup:
        outgoing_halfedges = list(h_lookup[v].values())
        dirs = np.array([v['pos'] - h.dest['pos'] for h in outgoing_halfedges])
        angles = angle_to_axis(dirs) % (2 * np.pi)
        order = np.argsort(angles)
        outgoing_halfedges = [outgoing_halfedges[i] for i in order]
        for hrevnex, h, hprerev in rotate_by(outgoing_halfedges, (0, 1, 2)):
            h.rev.nex = hrevnex
            h.pre = hprerev.rev

    # the faces
    unassigned_edges = copy(result.halfedges)
    while unassigned_edges:
        h = next(iter(unassigned_edges))
        f = Face(any_side=h)
        result.add_face(f)
        for k in f.halfedge_iter():
            k.face = f
            unassigned_edges.remove(k)

    # detect 'outside' faces which should be None by their orientation
    for f in frozenset(result.faces):
        vertex_pos = [v['pos'] for v in f.vertex_iter()]
        if signed_area(vertex_pos) < 0:
            result.delete_face(f)

    if not return_v_lookup:
        return result
    else:
        return result, v_lookup
