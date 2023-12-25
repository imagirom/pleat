import numpy as np
from eucare.half import HalfEdgeGraph, Vertex, IdObject, RegularNGon, CyclicHalfedgeGraph, InAngleHEG, \
    EuclideanPositionHEG
from eucare.instructions import *
from eucare.prototiles import EuclideanProtoTile, RegularEuclideanTile, complete_vertex_with_rhombus
from eucare.base import unit_vector
from copy import deepcopy, copy
from tqdm import tqdm as tqdm
import eucare.plotting as euplt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from eucare.redering import CairoRenderer
from eucare.example_tilesets import *

# tile = copy(tile)

# def RegularNGon(n):
#    return CyclicHalfedgeGraph([Vertex() for _ in range(n)])

# n = 6
# proto_tile = RegularEuclideanTile(n, edge_labels = ['a'] * n)

from eucare.half import *
from eucare.prototiles import *
render_settings = dict(line_width=0.05, face_inset=0.0,
                       render_edges=True, render_vertices=False, render_faces=True)


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


def test_graph():
    G, edgedict = RhombusTile().make_graph(add_positions=True)
    G = EuclideanPositionHEG(other=G)
    #RegularEuclideanTile(3).attach_instruction(0)(G, edgedict[0])
    RhombusTile().attach_instruction(0)(G, edgedict[0])
    return G


def attatch_test_graph(G1):
    func = lambda e: e.rev.pre['in_angle'] < np.pi/2
    e1 = get_edge_with(G1, func, on_border=True)
    G2 = test_graph()
    func = lambda e: any((edge['in_angle'] < np.pi/2 and edge.rev.on_border()) for edge in (e.rev, e.rev.pre))
    e2 = get_edge_with(G2, func, on_border=True)
    G1.glue_graph_e2e(G2, e1, e2)


alpha = 2*np.pi/9
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

G.show(**render_settings)


from eucare.conway import *

G = starify_graph(t=1/2)(G)

for e in G.border_edges():
    if e in G.halfedges:
        if any(v.attributes.get('join', False) for v in [e.orig, e.dest]):
            G.delete_face(e.rev.face)

G.show(**render_settings)
assert False


def print_graph(graph):
    for e in graph.halfedges:
        print(e.__repr__(), e.on_border(), e.face)


def render():
    renderer = CairoRenderer(width=1500, scale=30, face_inset=0.2, line_width=0.2)
    surface = renderer.render_graph(tiling, render_vertices=False, render_edges=False)
    filename = 'output.png'
    surface.write_to_png(filename)
    #surface.finish()
    img = mpimg.imread(filename)
    plt.figure(figsize=(13, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def remove_inner_hs(tiling, max=None, ratio=None):
    hs = list(tiling.halfedges)
    np.random.shuffle(hs)
    removed = 0
    if ratio is not None:
        assert max is None
        max = len(hs) * ratio
    for e in hs:
        if e in tiling.halfedges and not (

                e.on_border() or e.rev.on_border() or
                e.orig.order() < 3 or e.dest.order() < 3 or
                e.rev.face is e.face
        ):
            tiling.delete_edge(e)
            removed += 1
            if max is not None and removed >= max:
                break

tiles = t_3_3_4_3_4()


# tile = RegularNGon(n)
# print_graph(tile)
# for e in tile.border_edge_iter():
#    e['instruction'] = attatch_tile_instruction(proto_tile, 'a')


# for e in tile1.border_edge_iter():
#    e['instruction'] = instruction0


# possible problem: topology based merging can miss geometry
# NO! forgot to update any_outgoing!!

for _ in range(1):
    IdObject.reset_ids()
    tiling = EuclideanPositionHEG(eps=1e-3, other=tiles[-1].make_graph(add_positions=True)[0])
    for i in range(20):
        vertices = [e.dest for e in tiling.border_edges()]
        for v in vertices:
            k = 0
            while v.on_border():
                e_out = v.get_outgoing_border()
                #assert e_out in tiling.halfedges
                #assert e_out.orig is v, f'{e_out.orig in tiling.vertices}'
                tiling.execute_edge_instruction(e_out)
                #assert e_out not in tiling.halfedges
            # if e.on_border() and e in tiling.halfedges:
            #     tiling.execute_edge_instruction(e)
            #     print(e in tiling.halfedges)

print('checking consistency')
remove_inner_hs(tiling, ratio=1)
tiling.check_consistency()

for f in frozenset(tiling.faces):
    break
    if np.random.rand() > 1.2:
        tiling.delete_face(f)
        render()
        tiling.check_consistency()

print('rendering')

render()

# plotting with matplotlib
# plt.figure(figsize=(10, 10))
# for face in tiling.faces:
#     points = np.stack([v['pos'] for v in face.vertex_iter()])
#     euplt.plot_polygon(points)
# euplt.set_equal_aspect()
# plt.show()

#tiling.show_spring_layout()