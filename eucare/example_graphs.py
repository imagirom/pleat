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


def complete_vertex(G, v):
    assert v in G.vertices, f'{v} not in the vertices of {G}'
    while v.on_border():
        G.execute_edge_instruction(v.get_outgoing_border())


def add_vertex_ring(G):
    for v in [h.orig for h in G.border_edges()]:
        if v in G.vertices:
            complete_vertex(G, v)


def complete_closest_vertices(G, eps=1e-6):
    assert isinstance(G, GeometricHEG)
    vertex_dists = {e.orig: G.geometry.distance_to_origin(e.orig['pos']) for e in G.border_edge_iter()}
    d_min = np.min(list(vertex_dists.values()))
    [complete_vertex(G, v) for v, d in vertex_dists.items() if d-d_min < eps and v.on_border()]


def from_tiles(tiles, rings=2, vertex_based=True, base_tile=-1, add_positions=True):
    if isinstance(base_tile, int):
        base_tile = tiles[base_tile]
    if isinstance(base_tile, ProtoTile):
        base_tile = base_tile.make_graph(add_positions=add_positions)[0]
    if add_positions:
        assert len({tile.geometry for tile in tiles}) == 1, f'Geometries must agree but got {[tile.geometry for tile in tiles]}'
        assert tiles[0].geometry is not None
        G = GeometricHEG(geometry=tiles[0].geometry, other=base_tile)
    else:
        G = InAngleHEG(other=base_tile)
    if vertex_based:
        for i in range(rings):
            add_vertex_ring(G)
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