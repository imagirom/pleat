"""Tests for instructions: attatch_tile_instruction."""
from __future__ import annotations

from eucare.instructions import attatch_tile_instruction
from eucare.prototiles import RegularEuclideanTile


def test_attatch_tile_instruction_glues_to_border_edge():
    tile = RegularEuclideanTile(4)
    G, _edges = tile.make_graph(add_positions=True)
    initial_face_count = len(G.faces)
    border_edge = next(h for h in G.halfedges if h.on_border())
    inst = attatch_tile_instruction(tile)
    inst(G, border_edge)
    G.check_consistency()
    assert len(G.faces) == initial_face_count + 1
