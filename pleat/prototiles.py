"""Define tile prototypes with angles, edge lengths, and vertex positions."""

from __future__ import annotations

import numpy as np

from .base import angle_to_axis, unit_vector
from .geometries import EuclideanGeometry, PoincareDiskModel, SphereModel
from .half import CyclicHalfedgeGraph, Face, HalfEdge, InAngleHEG, Vertex
from .instructions import attatch_tile_instruction


class ProtoTile:
    """Abstract base class for tile prototypes that can produce half-edge graphs."""

    def make_graph(self) -> tuple[InAngleHEG, dict]:
        # return a HEG and a list of edges
        raise NotImplementedError


class PolygonalProtoTile(ProtoTile):
    """A tile defined by interior angles, edge lengths, and optional labels."""

    def __init__(
        self,
        in_angles: list[float],
        edge_lengths: list[float],
        edge_labels: list | None = None,
        vertex_labels: list | None = None,
        face_label: object = None,
        edge_instructions: dict | None = None,
    ) -> None:
        assert len(in_angles) == len(edge_lengths)
        self.order = len(in_angles)
        self.in_angles = in_angles
        self.edge_lengths = edge_lengths
        self.edge_labels = edge_labels if edge_labels is not None else list(range(self.order))
        self.vertex_labels = vertex_labels if vertex_labels is not None else list(range(self.order))
        self.face_label = face_label

        # edge_instructions can either be a list for all edges, or a dict, mapping a label to an instruction
        self.edge_instructions = edge_instructions if edge_instructions is not None else dict()

    def make_graph(self, add_positions: bool = False) -> tuple[CyclicHalfedgeGraph, dict]:
        """Build a half-edge graph for this tile and return it with an edge label dict."""
        outer_edge_dict = dict()

        outer_edges = [HalfEdge() for _ in range(self.order)]
        for e, label in zip(outer_edges, self.edge_labels):
            outer_edge_dict[label] = e
            e["label"] = label

        inner_edges = [HalfEdge() for _ in range(self.order)]
        for e, label, angle, length in zip(inner_edges, self.edge_labels, self.in_angles, self.edge_lengths):
            e["label"] = label
            e["in_angle"] = angle
            e["length"] = length

        vertices = [Vertex() for _ in range(self.order)]
        for v, label in zip(vertices, self.vertex_labels):
            v["label"] = label

        if add_positions:
            assert hasattr(self, "points")
            for v, pos in zip(vertices, self.points):
                v["pos"] = pos

        if isinstance(self.edge_instructions, dict):
            for e in outer_edges:
                if e["label"] in self.edge_instructions:
                    e["instruction"] = self.edge_instructions[e["label"]]
        else:
            assert self.edge_instructions is None, f"{self.edge_instructions}"

        f = Face()
        if self.face_label is not None:
            f["label"] = self.face_label
        graph = CyclicHalfedgeGraph(f=f, vs=vertices, inner_hs=inner_edges, outer_hs=outer_edges)
        return graph, outer_edge_dict

    def attach_instruction(self, label=None):
        """Return a glue instruction that attaches this tile at the given edge label."""
        assert label is None or label in self.edge_labels, f"{label}, {self.edge_labels}"
        return attatch_tile_instruction(self, label)

    def __str__(self):
        return (
            f"PolygonalProtoTile("
            f"lenghts={self.edge_lengths}, "
            f"angles={self.in_angles}, "
            f"edge_labels={self.edge_labels},"
            f"vertex_labels={self.vertex_labels}"
            f")"
        )


class RegularProtoTile(PolygonalProtoTile):
    """A regular polygon tile with uniform angles and edge lengths, supporting any geometry."""

    def __init__(self, n: int, in_angle: float, edge_length: float, **super_kwargs) -> None:
        super().__init__([in_angle] * n, [edge_length] * n, **super_kwargs)
        euclidean_angle_deficit = np.pi * (n - 2) - in_angle * n
        eps = 1e-6
        if abs(euclidean_angle_deficit) < eps:
            self.geometry = EuclideanGeometry
        elif euclidean_angle_deficit < 0:
            self.geometry = SphereModel
        else:
            self.geometry = PoincareDiskModel

        geo = self.geometry
        # calculate points
        points = [geo.origin(), geo.from_polar(edge_length, 0)]
        for _ in range(n - 2):
            points.append(geo.construct_next_poly_point(points[-2], points[-1], in_angle, edge_length))
        com = geo.center_of_mass(np.array(points))
        translate = geo.translation(com, geo.origin())
        points = np.array([translate(point) for point in points])
        angle = -geo.angle_to_axis(points[0])
        rotate = geo.rotation(geo.origin(), angle)
        self.points = np.array([rotate(point) for point in points])


class EuclideanProtoTile(PolygonalProtoTile):
    """A Euclidean tile defined by explicit 2D vertex positions."""

    def __init__(self, points=None, **super_kwargs):
        self.geometry = EuclideanGeometry
        # points should have shape (n, 2)
        points = np.array(points)
        assert len(points.shape) == 2 and points.shape[1] == 2, f"{points.shape}"
        self.points = points
        # compute edgelenths and angles
        edge_vectors = np.concatenate([points[1:], points[:1]]) - points
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        edge_angles = angle_to_axis(edge_vectors)
        in_angles = (np.pi + edge_angles - np.concatenate([edge_angles[1:], edge_angles[:1]])) % (2 * np.pi)
        super().__init__(in_angles, edge_lengths, **super_kwargs)


class RegularEuclideanTile(EuclideanProtoTile):
    """A regular n-gon in Euclidean geometry with unit edge length."""

    def __init__(self, n, **super_kwargs):
        points = unit_vector(np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n) / np.sin(np.pi / n) / 2
        super(RegularEuclideanTile, self).__init__(points=points, **super_kwargs)


class RhombusTile(EuclideanProtoTile):
    """A rhombus tile with a given acute angle alpha (default pi/3)."""

    def __init__(self, alpha=None, **super_kwargs):
        alpha = np.pi / 3 if alpha is None else alpha
        pts = np.concatenate(
            [np.zeros((1, 2)), np.cumsum(unit_vector([-alpha / 2, alpha / 2, np.pi - alpha / 2]), axis=0)]
        )
        super(RhombusTile, self).__init__(points=pts, **super_kwargs)


def complete_vertex_with_rhombus(graph, vertex):
    """Fill the remaining angle at a border vertex by gluing in a rhombus tile."""
    assert isinstance(graph, InAngleHEG)
    edge = vertex.get_outgoing_border()
    missing_angle = graph.tau - vertex.angle_sum()
    assert missing_angle < graph.tau / 2, f"{missing_angle}, {graph.tau}"
    tile = RhombusTile(missing_angle)
    new_graph, edgedict = tile.make_graph()
    graph.glue_graph_e2e(new_graph, edge, edgedict[0])
