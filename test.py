import numpy as np
from eucare.half import HalfEdgeGraph, Vertex, IdObject, RegularNGon, CyclicHalfedgeGraph, InAngleHEG, \
    EuclideanPositionHEG
from eucare.instructions import *
from eucare.base import unit_vector
from copy import deepcopy, copy
from tqdm import tqdm as tqdm
import eucare.plotting as euplt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from eucare.redering import CairoRenderer

# tile = copy(tile)

# def RegularNGon(n):
#    return CyclicHalfedgeGraph([Vertex() for _ in range(n)])

# n = 6
# proto_tile = RegularEuclideanTile(n, edge_labels = ['a'] * n)

def print_graph(graph):
    for e in graph.halfedges:
        print(e.__repr__(), e.on_border(), e.face)


def attatch_tile_instruction(proto_tile, label=None):
    def instruction(graph, edge):
        tile, edge_dict = proto_tile.make_graph()
        if label is not None:
            tile_edge = edge_dict[label]
        else:
            # just take any edge
            tile_edge = next(iter(edge_dict.values()))
        graph.glue_graph_e2e(tile, edge, tile_edge)

    return instruction


# define 6.4.3.4 tiling
hexagon = RegularEuclideanTile(6, edge_labels=['a', 'a', 'a', 'a', 'a', 'a'])
square = RegularEuclideanTile(4, edge_labels=['b', 'c', 'b', 'c'])
triangle = RegularEuclideanTile(3, edge_labels=['d', 'd', 'd'])
hexagon.edge_instructions['a'] = attatch_tile_instruction(square, 'b')
square.edge_instructions['b'] = attatch_tile_instruction(hexagon)
square.edge_instructions['c'] = attatch_tile_instruction(triangle)
triangle.edge_instructions['d'] = attatch_tile_instruction(square, 'c')

# define 3.3.4.3.4 tiling
cairo_tri = RegularEuclideanTile(3)
cairo_sq_A = RegularEuclideanTile(4, edge_labels=['a'] * 4)
cairo_sq_B = RegularEuclideanTile(4, edge_labels=['a'] * 4)
cairo_tri.edge_instructions[0] = attatch_tile_instruction(cairo_sq_A)
cairo_tri.edge_instructions[1] = attatch_tile_instruction(cairo_sq_B)
cairo_tri.edge_instructions[2] = attatch_tile_instruction(cairo_tri, 2)
cairo_sq_A.edge_instructions['a'] = attatch_tile_instruction(cairo_tri, 0)
cairo_sq_B.edge_instructions['a'] = attatch_tile_instruction(cairo_tri, 1)

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
    tiling = EuclideanPositionHEG(eps=1e-3, other=cairo_sq_A.make_graph(add_positions=True)[0])
    for i in range(3):
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

print('rendering')

renderer = CairoRenderer(width=1500, scale=50)
surface = renderer.render_graph(tiling)
filename = 'output.png'
surface.write_to_png(filename)
img = mpimg.imread(filename)
plt.imshow(img)
plt.axis('off')
plt.show()

# plotting with matplotlib
# plt.figure(figsize=(10, 10))
# for face in tiling.faces:
#     points = np.stack([v['pos'] for v in face.vertex_iter()])
#     euplt.plot_polygon(points)
# euplt.set_equal_aspect()
# plt.show()

#tiling.show_spring_layout()