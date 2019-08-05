from .half import HalfEdgeGraph, CyclicHalfedgeGraph, HalfEdge
from copy import deepcopy

# An edge instruction is a function with signature (HalfEdgeGraph, HalfEdge) ->


class HalfEdgeInstruction:
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

