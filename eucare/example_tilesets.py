from .prototiles import RegularEuclideanTile


class TileSet:
    def __init__(self, tiles, root_tile):
        self.tiles = tiles
        self.root_tile = root_tile

    def expand_to_area(self, area):
        raise NotImplementedError


def align_tiles(tile1, label1, tile2, label2):
    tile1.edge_instructions[label1] = tile2.attach_instruction(label2)
    tile2.edge_instructions[label2] = tile1.attach_instruction(label1)

# TODO: alternate approach: do something like f12.to_attach = [f4, f6] * 6


def platonic(n):
    assert n in (3, 4, 6)
    t = RegularEuclideanTile(n, edge_labels=['a'] * n)
    align_tiles(t, 'a', t, 'a')
    return [t]


def t_4_6_12():
    f4 = RegularEuclideanTile(4, edge_labels=[6, 12] * 2)
    f6 = RegularEuclideanTile(6, edge_labels=[4, 12] * 3)
    f12 = RegularEuclideanTile(12, edge_labels=[4, 6] * 6)
    align_tiles(f4, 6, f6, 4)
    align_tiles(f6, 12, f12, 6)
    align_tiles(f4, 12, f12, 4)
    return f4, f6, f12


def t_3_3_4_3_4():
    tri = RegularEuclideanTile(3)
    sq1 = RegularEuclideanTile(4, edge_labels=[1]*4)
    sq2 = RegularEuclideanTile(4, edge_labels=[2]*4)
    align_tiles(tri, 0, tri, 0)
    align_tiles(tri, 1, sq1, 1)
    align_tiles(tri, 2, sq2, 2)
    return sq1, sq2, tri

# for printing: TA3 (Triangle, A, side 3) or Q1. Letters in ceter, numbers in sides.


def pgg_2x():
    tri_11 = RegularEuclideanTile(3)
    tri_12 = RegularEuclideanTile(3)
    tri_21 = RegularEuclideanTile(3)
    tri_22 = RegularEuclideanTile(3)
    sq1 = RegularEuclideanTile(4)
    sq2 = RegularEuclideanTile(4)

    align_tiles(sq1, 0, sq1, 0)
    align_tiles(sq1, 1, tri_11, 2)
    align_tiles(sq1, 2, tri_22, 1)
    align_tiles(sq1, 3, tri_12, 1)
    align_tiles(tri_11, 0, tri_11, 0)
    align_tiles(tri_11, 1, tri_12, 0)

    align_tiles(sq2, 0, sq2, 0)
    align_tiles(sq2, 1, tri_22, 2)
    align_tiles(sq2, 2, tri_12, 2)
    align_tiles(sq2, 3, tri_21, 1)
    align_tiles(tri_21, 0, tri_21, 0)
    align_tiles(tri_21, 2, tri_22, 0)

    return sq1, sq2, tri_11, tri_12, tri_21, tri_22



# # define 6.4.3.4 tiling
# hexagon = RegularEuclideanTile(6, edge_labels=['a', 'a', 'a', 'a', 'a', 'a'])
# square = RegularEuclideanTile(4, edge_labels=['b', 'c', 'b', 'c'])
# triangle = RegularEuclideanTile(3, edge_labels=['d', 'd', 'd'])
# hexagon.edge_instructions['a'] = attatch_tile_instruction(square, 'b')
# square.edge_instructions['b'] = attatch_tile_instruction(hexagon)
# square.edge_instructions['c'] = attatch_tile_instruction(triangle)
# triangle.edge_instructions['d'] = attatch_tile_instruction(square, 'c')
#
# # define 3.3.4.3.4 tiling
# cairo_tri = RegularEuclideanTile(3)
# cairo_sq_A = RegularEuclideanTile(4, edge_labels=['a'] * 4)
# cairo_sq_B = RegularEuclideanTile(4, edge_labels=['a'] * 4)
# cairo_tri.edge_instructions[0] = attatch_tile_instruction(cairo_sq_A)
# cairo_tri.edge_instructions[1] = attatch_tile_instruction(cairo_sq_B)
# cairo_tri.edge_instructions[2] = attatch_tile_instruction(cairo_tri, 2)
# cairo_sq_A.edge_instructions['a'] = attatch_tile_instruction(cairo_tri, 0)
# cairo_sq_B.edge_instructions['a'] = attatch_tile_instruction(cairo_tri, 1)