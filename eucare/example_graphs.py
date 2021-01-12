from .half import *
from .prototiles import *
from .example_tilesets import *

def get_edge_with(graph, func=None, on_border=False):
    assert isinstance(graph, HalfEdgeGraph)
    func = func if func is not None else lambda _: True
    edge_iter = graph.border_edge_iter() if on_border else graph.halfedges
    for e in edge_iter:
        if func(e):
            return e
    raise LookupError('Cannot find edge with requested property')


def get_vertex_with(graph, func=None, on_border=False):
    assert isinstance(graph, HalfEdgeGraph)
    func = func if func is not None else lambda _: True
    vertex_iter = (e.orig for e in graph.border_edge_iter()) if on_border else graph.vertices
    for v in vertex_iter:
        if func(v):
            return v
    raise LookupError('Cannot find vertex with requested property')


def rosette(n=8):
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


def from_tiles(tiles, rings=15, vertex_based=True, base_tile=-1, add_positions=True, hyperbolic=False):
    if isinstance(base_tile, int):
        base_tile = tiles[base_tile]
    if isinstance(base_tile, ProtoTile):
        base_tile = base_tile.make_graph(add_positions=add_positions)[0]
    if add_positions:
        if not hyperbolic:
            G = EuclideanPositionHEG(other=base_tile)
        else:
            assert False
    else:
        G = InAngleHEG(other=base_tile)
    if vertex_based:
        for i in range(rings):
            for v in [h.orig for h in G.border_edges()]:
                while v in G.vertices and v.on_border():
                    G.execute_edge_instruction(v.get_outgoing_border())
    else:
        for i in range(rings):
            for h in G.border_edges():
                if h.on_border() and h in G.halfedges:
                    if 'instruction' in h.attributes:
                        G.execute_edge_instruction(h)
    return G


def pgg_2x_tiling(rings=15):
    tiles = pgg_2x()
    return from_tiles(tiles, rings)