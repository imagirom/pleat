"""Example graph constructions: rosettes, tiling growth, and hyperbolic mappings."""

from __future__ import annotations

import logging

import numpy as np
from sympy import N, elliptic_f

from .example_tilesets import curved_zip, pgg_2x
from .half import EuclideanPositionHEG, GeometricHEG, HalfEdge, HalfEdgeGraph, InAngleHEG, Vertex
from .prototiles import ProtoTile, RhombusTile, complete_vertex_with_rhombus

logger = logging.getLogger(__name__)


# TODO https://en.wikipedia.org/wiki/File:Planar_Fractalizing_Truncated_Hexagonal_Tiling_II.png


def get_edge_with(graph: HalfEdgeGraph, func: "callable | None" = None, on_border: bool = False) -> HalfEdge:
    """Return the first half-edge satisfying ``func``, optionally restricted to border edges.

    Args:
        graph: Graph to search.
        func: Predicate ``HalfEdge -> bool``. Defaults to always-True.
        on_border: If True, search only border half-edges.

    Raises:
        LookupError: If no matching half-edge exists.
    """
    assert isinstance(graph, HalfEdgeGraph)
    func = func if func is not None else lambda _: True
    edge_iter = graph.border_edge_iter() if on_border else graph.halfedges
    for e in edge_iter:
        if func(e):
            return e
    raise LookupError("Cannot find edge with requested property")


def get_vertex_with(graph: HalfEdgeGraph, func: "callable | None" = None, on_border: bool = False) -> Vertex:
    """Return the first vertex satisfying ``func``, optionally restricted to border vertices.

    Args:
        graph: Graph to search.
        func: Predicate ``Vertex -> bool``. Defaults to always-True.
        on_border: If True, search only vertices on the border.

    Raises:
        LookupError: If no matching vertex exists.
    """
    assert isinstance(graph, HalfEdgeGraph)
    func = func if func is not None else lambda _: True
    vertex_iter = (e.orig for e in graph.border_edge_iter()) if on_border else graph.vertices
    for v in vertex_iter:
        if func(v):
            return v
    raise LookupError("Cannot find vertex with requested property")


def rosette(n: int = 8) -> EuclideanPositionHEG:
    """Construct a rosette pattern from ``n`` rhombus tiles around a central vertex."""
    assert isinstance(n, int)
    alpha = 2 * np.pi / n
    G, edgedict = RhombusTile(alpha).make_graph(add_positions=True)
    G = EuclideanPositionHEG(other=G)
    v = edgedict[0].dest
    while v.on_border():
        RhombusTile(alpha).attach_instruction(0)(G, v.get_outgoing_border())

    while True:
        concave_vertices = [v for v in G.vertices if v.on_border() and 0 < G.tau - v.angle_sum() < np.pi]
        if not concave_vertices:
            break
        for v in concave_vertices:
            complete_vertex_with_rhombus(G, v)

    return G


def complete_vertex(G: HalfEdgeGraph, v: Vertex) -> None:
    """Attach tiles around a border vertex until it is fully surrounded."""
    assert v in G.vertices, f"{v} not in the vertices of {G}"
    while v.on_border():
        G.execute_edge_instruction(v.get_outgoing_border())


def add_vertex_ring(G: HalfEdgeGraph) -> None:
    """Complete all border vertices, prioritizing those with the largest angle sum."""
    vs = [h.orig for h in G.border_edges()]
    vs = sorted(vs, key=lambda v: -v.angle_sum())
    for v in vs:
        if v in G.vertices:
            complete_vertex(G, v)


def complete_closest_vertices(G: GeometricHEG, eps: float = 1e-6) -> None:
    """Complete border vertices closest to the origin."""
    assert isinstance(G, GeometricHEG)
    vertex_dists = {e.orig: G.geometry.distance_to_origin(e.orig["pos"]) for e in G.border_edge_iter()}
    d_min = np.min(list(vertex_dists.values()))
    [complete_vertex(G, v) for v, d in vertex_dists.items() if d - d_min < eps and v.on_border()]


def from_tiles(
    tiles: list[ProtoTile],
    rings: int = 2,
    vertex_based: bool = True,
    base_tile: "int | ProtoTile | HalfEdgeGraph" = -1,
    add_positions: bool = True,
) -> HalfEdgeGraph:
    """Grow a tiling from a list of proto-tiles by expanding for the given number of rings.

    Args:
        tiles: Proto-tiles available for the tiling.
        rings: Number of expansion rings.
        vertex_based: If True, expand by completing border vertices; otherwise
            expand edge by edge.
        base_tile: Starting tile -- index into ``tiles``, a :class:`ProtoTile`,
            or a pre-built :class:`HalfEdgeGraph`.
        add_positions: If True, attach geometric positions during growth.

    Returns:
        The grown half-edge graph (:class:`GeometricHEG` if ``add_positions``
        is True, otherwise :class:`InAngleHEG`).
    """
    if isinstance(base_tile, int):
        base_tile = tiles[base_tile]
    if isinstance(base_tile, ProtoTile):
        base_tile = base_tile.make_graph(add_positions=add_positions)[0]
    if add_positions:
        assert (
            len({tile.geometry for tile in tiles}) == 1
        ), f"Geometries must agree but got {[tile.geometry for tile in tiles]}"
        assert tiles[0].geometry is not None
        G = GeometricHEG(geometry=tiles[0].geometry, other=base_tile)
    else:
        G = InAngleHEG(other=base_tile)
    if vertex_based:
        for _ in range(rings):
            add_vertex_ring(G)
    else:
        for _ in range(rings):
            for h in G.border_edges():
                if h.on_border() and h in G.halfedges:
                    if "instruction" in h.attributes:
                        G.execute_edge_instruction(h)
    return G


def pgg_2x_tiling(rings: int = 15) -> HalfEdgeGraph:
    """Construct a pgg wallpaper group tiling with the given number of rings."""
    tiles = pgg_2x()
    return from_tiles(tiles, rings)


def kised_soccer_ball() -> HalfEdgeGraph:
    """Construct a kised soccer ball (icosahedron variant) on the sphere."""
    from eucare.conway import kis_graph

    n, k = (5, 3)
    G = from_tiles(curved_zip(n, k), rings=3)
    G = kis_graph()(G)
    G.delete_subset([v for v in G.vertices if v.on_border() and v.order() < 5])
    G.delete_subset([f for f in G.faces if f.order() > 3])
    return G


def hyperbolic_square_graph(
    G: GeometricHEG | None = None,
    min_length: float = 0.05,
    dual: bool = False,
) -> GeometricHEG:
    """Map a hyperbolic tiling to the Poincare square via the Schwarz-Christoffel mapping.

    Args:
        G: Hyperbolic tiling to map. Defaults to a {7, 3} tiling with one ring.
        min_length: Minimum target edge length in the square; growth stops once
            no border vertex falls below this threshold.
        dual: If True, take the dual graph before mapping.

    Returns:
        A copy of the tiling with positions in the unit square.
    """
    import eucare as ec

    def wrapped_elliptic_f(z, m):
        return complex(N(elliptic_f(z, m)))

    def complex_to_array(z):
        return np.array([z.real, z.imag])

    def disk_to_square(z, w=1):
        return np.sqrt(1j) * wrapped_elliptic_f(np.arcsin(w * z), -1)
        # return np.sqrt(2) * wrapped_elliptic_f(np.arcsin(np.sqrt(z+1)), np.sqrt(2)/2)

    def disk_to_halfplane(z):
        return (z + 1j) / (1j * z + 1)

    if G is None:
        G = ec.example_graphs.from_tiles(ec.example_graphs.curved_platonic(7, 3), 1)

    def map_to_square(G):
        G_square = G.copy()
        G_square.geometry = ec.geometries.EuclideanGeometry
        for v in G_square.vertices:
            if "square_pos" in v:
                v["pos"] = v["square_pos"]
            else:
                v["pos"] = complex_to_array(disk_to_square(v["pos"], np.sqrt(1j).conjugate()))

        return G_square

    nv_before = None
    nv_after = G.order
    while nv_before != nv_after:
        nv_before = nv_after
        logger.debug("vertices: %d", nv_before)
        for v in G.vertices:
            if "square_pos" not in v:
                v["square_pos"] = complex_to_array(disk_to_square(v["pos"], np.sqrt(1j).conjugate()))
                for e in v.outgoing_iter():
                    if "square_pos" not in e.dest:
                        continue
                    e["square_length"] = e.rev["square_length"] = np.linalg.norm(v["square_pos"] - e.dest["square_pos"])

        queue = {}
        for v in G.vertices:
            if not v.on_border():
                continue
            v["square_length"] = np.mean([e["square_length"] for e in v.outgoing_iter() if e.on_border()])
            queue[v] = np.min(9.27037339e-01 - np.abs(v["square_pos"]))  # v['square_length']

        for v, length in reversed(sorted(queue.items(), key=lambda kv: kv[1])):
            if length > min_length:
                if v.on_border():
                    ec.example_graphs.complete_vertex(G, v)
        nv_after = G.order

    if dual:
        G = ec.conway.dual_graph()(G)
    G_square = map_to_square(G)

    return G_square
