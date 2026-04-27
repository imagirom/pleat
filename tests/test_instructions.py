"""Tests for instructions: GlueTileInstruction and attatch_tile_instruction."""
from __future__ import annotations

from eucare.instructions import GlueTileInstruction, attatch_tile_instruction
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


def test_glue_tile_instruction_glues_via_template():
    tile = RegularEuclideanTile(3)
    G, _edge_dict = tile.make_graph(add_positions=True)
    template_tile, template_edges = tile.make_graph(add_positions=True)
    # The template "edge" is the OUTER border halfedge (not the inner one).
    template_edge = next(iter(template_edges.values()))
    inst = GlueTileInstruction(template_tile, template_edge)

    border_edge = next(h for h in G.halfedges if h.on_border())
    initial_faces = len(G.faces)
    inst(G, border_edge)
    G.check_consistency()
    assert len(G.faces) == initial_faces + 1


def test_halfedge_instruction_abstract_raises():
    from eucare.instructions import HalfEdgeInstruction
    from eucare.half import RegularNGon
    G = RegularNGon(3)
    h = next(iter(G.halfedges))
    inst = HalfEdgeInstruction()
    import pytest
    with pytest.raises(NotImplementedError):
        inst(G, h)


def test_special_copy_with_excluded_attributes():
    from eucare.instructions import special_copy
    from eucare.half import RegularNGon
    G = RegularNGon(3)
    h = next(iter(G.halfedges))
    h['instruction'] = 'sentinel'
    h['other'] = 42
    cp = special_copy(h, exclude_attributes=['instruction', 'other'])
    # excluded values are preserved on the copy.
    assert cp['instruction'] == 'sentinel'
    assert cp['other'] == 42


def test_special_copy_graph_stub_runs():
    from eucare.instructions import special_copy_graph
    from eucare.half import RegularNGon
    G = RegularNGon(3)
    # stub returns None; just ensure it runs.
    assert special_copy_graph(G) is None
