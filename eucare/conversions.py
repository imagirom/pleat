"""Convert between NetworkX graphs and half-edge graph representations.

The main entry point is :func:`EHEG_from_nx`, which embeds an undirected
planar :class:`networkx.Graph` (with vertex positions) into an
:class:`~eucare.half.EuclideanPositionHEG`.  Faces are recovered from the
planar embedding implied by the cyclic angular order of edges around each
vertex; the unbounded outer face is detected as the unique negatively-
oriented face and removed.
"""

from __future__ import annotations

import logging
from copy import copy

import networkx as nx
import numpy as np

from .base import angle_to_axis
from .half import EuclideanPositionHEG, Face, HalfEdge, Vertex, rotate_by

logger = logging.getLogger(__name__)


def _delete_dangling_edges_nx(nx_graph: nx.Graph) -> int:
    """Iteratively delete degree-1 vertices in *nx_graph* in place; return the deletion count."""
    finished = False
    n_deleted = 0
    while nx_graph.order() > 0 and not finished:
        finished = True
        for n in list(nx_graph.nodes):
            if len(nx_graph[n]) == 1:
                finished = False
                nx_graph.remove_node(n)
                n_deleted += 1
    return n_deleted


def EHEG_from_nx(
    nxg: nx.Graph,
    positions: dict | None = None,
    return_v_lookup: bool = False,
) -> "EuclideanPositionHEG | tuple[EuclideanPositionHEG, dict]":
    """Convert a planar undirected :class:`networkx.Graph` to an :class:`EuclideanPositionHEG`.

    Args:
        nxg: An undirected, planar graph.  Dangling vertices (degree 1) are
            pruned with a warning; node and edge attributes are copied onto
            the resulting :class:`Vertex` / :class:`HalfEdge` objects.
        positions: Optional ``{node: 2d position}`` mapping.  Defaults to
            interpreting each node ``n`` as ``np.array(n)``.
        return_v_lookup: If True, also return ``{nx_node: Vertex}``.

    Returns:
        The resulting graph, or ``(graph, v_lookup)`` if *return_v_lookup* is True.
    """
    assert not nxg.is_directed()
    if positions is None:
        positions = {n: np.array(n) for n in nxg.nodes()}
    assert isinstance(positions, dict)
    n_dangling = _delete_dangling_edges_nx(nxg)
    if n_dangling > 0:
        logger.warning("Deleted %d dangling edges in conversion to EHG", n_dangling)
    result = EuclideanPositionHEG()
    v_lookup = dict()
    for n, attrs in nxg.nodes().data():
        v = Vertex()
        # assign node attributes
        for key, value in attrs.items():
            v[key] = value
        v["pos"] = positions[n]
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
            # assign edge attributes
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
        dirs = np.array([v["pos"] - h.dest["pos"] for h in outgoing_halfedges])
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
        # print(f.area())

    result.check_consistency()

    # detect 'outside' faces which should be None by their orientation
    # it is selected as the one with maximal negative area
    # (area 0 faces might have slightly negative areas due to numerical issues)
    outside_face = None
    current_min_area = 0
    for f in frozenset(result.faces):
        area = f.area()
        if area < current_min_area:
            current_min_area = area
            outside_face = f
    assert outside_face is not None, "Could not find an outside face to delete. Are all areas 0?"
    result.delete_face(outside_face)

    if not return_v_lookup:
        return result
    else:
        return result, v_lookup
