"""Edge instructions that describe how to attach tiles to border edges during tiling growth."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .half import HalfEdge, HalfEdgeGraph
    from .prototiles import ProtoTile


def attatch_tile_instruction(proto_tile: "ProtoTile", label: object = None) -> "callable":
    """Return a callable that builds a fresh tile graph from ``proto_tile`` and glues it to a given edge.

    The newly created tile graph may itself carry instructions on its border
    edges, which will be executed when those edges are reached during growth.
    This allows recursive growth patterns to be encoded in a single proto-tile
    graph.

    Args:
        proto_tile: A tile object that can build a tile graph via ``make_graph()``.
        label: If given, the label of the edge in the tile graph to glue to
            the border edge. If None, an arbitrary edge is used.

    Returns:
        A callable ``(graph, edge) -> None`` that performs the gluing.
    """

    def instruction(graph: "HalfEdgeGraph", edge: "HalfEdge") -> None:
        tile, edge_dict = proto_tile.make_graph()
        if label is not None:
            tile_edge = edge_dict[label]
        else:
            # just take any edge
            tile_edge = next(iter(edge_dict.values()))
        graph.glue_graph_e2e(tile, edge, tile_edge)

    return instruction


# from copy import deepcopy
# from .half import HalfEdge, HalfEdgeGraph

# The below is WIP for a cleaner / more flexible framework for instructions.
# Currently not planned to be implemented, maybe later when there is a concrete use case that requires it.

# class HalfEdgeInstruction:
#     """Abstract base for instructions that modify a graph at a given half-edge."""
#     def __call__(self, graph: HalfEdgeGraph, h: HalfEdge) -> None:
#         assert isinstance(graph, HalfEdgeGraph), f'{type(graph)}'
#         assert isinstance(h, HalfEdge), f'{type(h)}'
#         self.execute(graph, h)

#     def execute(self, graph: HalfEdgeGraph, h: HalfEdge) -> None:
#         raise NotImplementedError


# def special_copy(e, exclude_attributes='instruction'):
#     exclude_dict = {key: e.attributes.pop(key) for key in exclude_attributes if key in e.attributes}
#     result = deepcopy(e)
#     for key, value in exclude_dict.items():
#         result[key] = value
#     return result

# # the INSTRUCTION needs to stay constant, while the TILE changes

# class GlueTileInstruction(HalfEdgeInstruction):
#     """Glue a copy of a tile graph onto a border edge."""

#     def __init__(self, tile: HalfEdgeGraph, edge: HalfEdge) -> None:
#         self.tile = tile
#         self.edge = edge

#     def execute(self, graph: HalfEdgeGraph, h: HalfEdge) -> None:
#         # TODO: this deepcopy solution is bad.. it leads to self.tile being stored many times.. still O(1) though..
#         # Solution: only make copies of edges, vertices, faces, not their attributes
#         tile, h2 = deepcopy((self.tile, self.edge))

#         graph.glue_graph_e2e(tile, h2, h)

#     def __deepcopy__(self, memodict={}):
#         # urgh that hack
#         return self
