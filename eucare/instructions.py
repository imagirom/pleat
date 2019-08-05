from .half import HalfEdgeGraph, CyclicHalfedgeGraph, HalfEdge, Vertex, Face
from copy import deepcopy
import numpy as np
from .base import angle_to_axis, unit_vector

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


class ProtoTile:
    def make_graph(self):
        # return a HEG and a list of edges
        raise NotImplementedError


class EuclideanProtoTile(ProtoTile):
    def __init__(self, points=None, edge_labels=None, vertex_labels=None, edge_instructions=None):
        # points should have shape (n, 2)
        points = np.array(points)
        assert len(points.shape) == 2 and points.shape[1] == 2, f'{points.shape}'
        self.order = len(points)
        self.points = points
        self.edge_labels = edge_labels if edge_labels is not None else list(range(self.order))
        self.vertex_labels = vertex_labels if vertex_labels is not None else list(range(self.order))

        # compute edgelenths and angles
        edge_vectors = points - np.concatenate([points[1:], points[:1]])
        self.edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        edge_angles = angle_to_axis(self.points)
        self.in_angles = (edge_angles - np.pi - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2*np.pi)
        print(f'in_angles: {self.in_angles}')

        # edge_instructions can either be a list for all edges, or a dict, mapping a label to an instruction
        self.edge_instructions = edge_instructions if edge_instructions is not None else dict()

    def make_graph(self):
        outer_edge_dict = dict()

        outer_edges = [HalfEdge() for _ in range(self.order)]
        for e, label in zip(outer_edges, self.edge_labels):
            outer_edge_dict[label] = e
            e['label'] = label

        inner_edges = [HalfEdge() for _ in range(self.order)]
        for e, label, angle, length in zip(inner_edges, self.edge_labels, self.in_angles, self.edge_lengths):
            e['label'] = label
            e['in_angle'] = angle
            e['length'] = length

        vertices = [Vertex() for _ in range(self.order)]
        for v, label in zip(vertices, self.vertex_labels):
            v['label'] = label

        if isinstance(self.edge_instructions, dict):
            for e in outer_edges:
                if e['label'] in self.edge_instructions:
                    e['instruction'] = self.edge_instructions[e['label']]
        else:
            assert self.edge_instructions is None, f'{self.edge_instructions}'

        graph = CyclicHalfedgeGraph(vs=vertices, inner_hs=inner_edges, outer_hs=outer_edges)
        return graph, outer_edge_dict

    def __str__(self):
        return f"EuclideanPrototile(" \
            f"lenghts={self.edge_lengths}, " \
            f"angles={self.in_angles}, " \
            f"edge_labels={self.edge_labels}," \
            f"vertex_labels={self.vertex_labels}" \
            f")"


class RegularEuclideanTile(EuclideanProtoTile):
    def __init__(self, n, **super_kwargs):
        points = unit_vector(np.linspace(0, 2*np.pi, n, endpoint=False))
        super(RegularEuclideanTile, self).__init__(points=points, **super_kwargs)
