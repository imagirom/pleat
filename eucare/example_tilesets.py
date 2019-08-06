from .prototiles import RegularEuclideanTile


class TileSet:
    def __init__(self, tiles, root_tile):
        self.tiles = tiles
        self.root_tile = root_tile

    def expand_to_area(self, area):
        raise NotImplementedError


def align_tiles(tile1, label1, tile2, label2):
    tile1.edge_instructions[label1] = tile2.attatch_instruction(label2)
    tile2.edge_instructions[label2] = tile1.attatch_instruction(label1)

# TODO: alternate approach: do something like f12.to_attach = [f4, f6] * 6

def t_4_6_12():
    f4 = RegularEuclideanTile(4, edge_labels=[6, 12] * 2)
    f6 = RegularEuclideanTile(6, edge_labels=[4, 12] * 3)
    f12 = RegularEuclideanTile(12, edge_labels=[4, 6] * 6)
    align_tiles(f4, 6, f6, 4)
    align_tiles(f6, 12, f12, 6)
    align_tiles(f)
    f4.edge_instructions[6] = f6.attatch_instruction(4)
    f4.edge_instructions[12] = f12.attatch_instruction(4)
    f6.edge_instructions[4] = f4.attatch_instruction(6)
    f6.edge_instructions[12] = f12.attatch_instruction(6)
    f12.edge_instructions[4] = f4.attatch_instruction(12)
    f12.edge_instructions[6] = f6.attatch_instruction(12)
    return f4, f6, f12

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