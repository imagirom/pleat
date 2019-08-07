import numpy as np
from eucare.half import HalfEdgeGraph, Vertex, IdObject, RegularNGon, CyclicHalfedgeGraph, InAngleHEG, \
    EuclideanPositionHEG
from eucare.instructions import *
from eucare.prototiles import EuclideanProtoTile, RegularEuclideanTile
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

def print_graph(graph):
    for e in graph.halfedges:
        print(e.__repr__(), e.on_border(), e.face)

def render():
    renderer = CairoRenderer(width=1500, scale=30, face_inset=0.1, line_width=0)
    surface = renderer.render_graph(tiling)
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
remove_inner_hs(tiling, ratio=0.2)
#tiling.check_consistency()


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