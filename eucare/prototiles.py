from .half import HalfEdge, CyclicHalfedgeGraph, Vertex, InAngleHEG
from .base import angle_to_axis, unit_vector, unit_vector_to_vector
from .instructions import attatch_tile_instruction
import numpy as np


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
        edge_vectors = np.concatenate([points[1:], points[:1]]) - points
        self.edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        edge_angles = angle_to_axis(edge_vectors)
        self.in_angles = (np.pi + edge_angles - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2*np.pi)

        # edge_instructions can either be a list for all edges, or a dict, mapping a label to an instruction
        self.edge_instructions = edge_instructions if edge_instructions is not None else dict()

    def make_graph(self, add_positions=False):
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

        if add_positions:
            for v, pos in zip(vertices, self.points):
                v['pos'] = pos

        if isinstance(self.edge_instructions, dict):
            for e in outer_edges:
                if e['label'] in self.edge_instructions:
                    e['instruction'] = self.edge_instructions[e['label']]
        else:
            assert self.edge_instructions is None, f'{self.edge_instructions}'

        graph = CyclicHalfedgeGraph(vs=vertices, inner_hs=inner_edges, outer_hs=outer_edges)
        return graph, outer_edge_dict

    def attach_instruction(self, label=None):
        assert label is None or label in self.edge_labels, f'{label}, {self.edge_labels}'
        return attatch_tile_instruction(self, label)

    def __str__(self):
        return f"EuclideanPrototile(" \
            f"lenghts={self.edge_lengths}, " \
            f"angles={self.in_angles}, " \
            f"edge_labels={self.edge_labels}," \
            f"vertex_labels={self.vertex_labels}" \
            f")"


class RegularEuclideanTile(EuclideanProtoTile):
    def __init__(self, n, **super_kwargs):
        points = unit_vector(np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/n) / np.sin(np.pi/n) / 2
        super(RegularEuclideanTile, self).__init__(points=points, **super_kwargs)


class RhombusTile(EuclideanProtoTile):
    def __init__(self, alpha=None, **super_kwargs):
        alpha = np.pi / 3 if alpha is None else alpha
        pts = np.concatenate([np.zeros((1, 2)), np.cumsum(unit_vector([-alpha/2, alpha/2, np.pi - alpha/2]), axis=0)])
        super(RhombusTile, self).__init__(points=pts, **super_kwargs)


def complete_vertex_with_rhombus(graph, vertex):
    assert isinstance(graph, InAngleHEG)
    edge = vertex.get_outgoing_border()
    missing_angle = graph.tau - vertex.angle_sum()
    assert missing_angle < graph.tau/2, f'{missing_angle}, {graph.tau}'
    tile = RhombusTile(missing_angle)
    new_graph, edgedict = tile.make_graph()
    graph.glue_graph_e2e(new_graph, edge, edgedict[0])
