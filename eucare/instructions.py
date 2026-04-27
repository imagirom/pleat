"""Edge instructions that describe how to attach tiles to border edges during tiling growth."""

from .half import HalfEdgeGraph, HalfEdge
from copy import deepcopy
import numpy as np
from .base import angle_to_axis, unit_vector

# An edge instruction is a function with signature (HalfEdgeGraph, HalfEdge) ->


class HalfEdgeInstruction:
    """Abstract base for instructions that modify a graph at a given half-edge."""
    def __call__(self, graph, h):
        assert isinstance(graph, HalfEdgeGraph), f'{type(graph)}'
        assert isinstance(h, HalfEdge), f'{type(h)}'
        self.execute(graph, h)

    def execute(self, graph, h):
        raise NotImplementedError


def special_copy(e, exclude_attributes='instruction'):
    exclude_dict = {key: e.attributes.pop(key) for key in exclude_attributes if key in e.attributes}
    result = deepcopy(e)
    for key, value in exclude_dict.items():
        result[key] = value
    return result


def special_copy_graph(graph):
    vertices = deepcopy(graph.vertices)
    faces = deepcopy(graph.faces)

# the INSTRUCTION needs to stay constant, while the TILE changes


class GlueTileInstruction(HalfEdgeInstruction):
    """Glue a copy of a tile graph onto a border edge."""

    def __init__(self, tile, edge):
        self.tile = tile
        self.edge = edge

    def execute(self, graph, h):
        # TODO: this deepcopy solution is bad.. it leads to self.tile being stored many times.. still O(1) though..
        # Solution: only make copies of edges, vertices, faces, not their attributes
        tile, h2 = deepcopy((self.tile, self.edge))

        graph.glue_graph_e2e(tile, h2, h)

    def __deepcopy__(self, memodict={}):
        # urgh that hack
        return self

# TODO: choose this or the stuff above


def attatch_tile_instruction(proto_tile, label=None):
    """Return a callable that builds a fresh tile graph and glues it to the given edge."""

    def instruction(graph, edge):
        tile, edge_dict = proto_tile.make_graph()
        if label is not None:
            tile_edge = edge_dict[label]
        else:
            # just take any edge
            tile_edge = next(iter(edge_dict.values()))
        graph.glue_graph_e2e(tile, edge, tile_edge)
    return instruction


# TODO: make this search for 'adjacent_prototile'

