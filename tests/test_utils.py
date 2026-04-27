"""Tests for eucare.utils helpers."""
from __future__ import annotations

import logging

from eucare.example_graphs import rosette
from eucare.half import RegularNGon
from eucare.utils import (
    VerboseTimer,
    invert_mapping,
    print_attribute_info,
    random_directed_set,
)


def test_invert_mapping_basic():
    assert invert_mapping({1: 'a', 2: 'b'}) == {'a': 1, 'b': 2}


def test_invert_mapping_empty():
    assert invert_mapping({}) == {}


def test_random_directed_set_pairs_each_edge_once():
    G = RegularNGon(6)
    directed = random_directed_set(G)
    # Each undirected edge contributes exactly one half-edge.
    assert len(directed) * 2 == len(G.halfedges)
    # No half-edge and its reverse appear together.
    for h in directed:
        assert h.rev not in directed


def test_random_directed_set_accepts_iterable_of_halfedges():
    G = RegularNGon(4)
    directed = random_directed_set(list(G.halfedges))
    assert len(directed) * 2 == len(G.halfedges)


def test_print_attribute_info_on_graph(caplog):
    G = rosette(n=4)
    with caplog.at_level(logging.INFO, logger='eucare.utils'):
        print_attribute_info(G)
    msgs = [rec.message for rec in caplog.records]
    # The graph branch logs section headers.
    assert any('Vertices' in m for m in msgs)
    assert any('Halfedges' in m for m in msgs)
    assert any('Faces' in m for m in msgs)


def test_print_attribute_info_on_collection(caplog):
    G = rosette(n=4)
    # Add some attribute including an unhashable value (list) to exercise the TypeError branch.
    for v in G.vertices:
        v['some_label'] = 'x'
        v['unhashable'] = [1, 2, 3]
    with caplog.at_level(logging.INFO, logger='eucare.utils'):
        print_attribute_info(G.vertices)
    msgs = [rec.message for rec in caplog.records]
    assert any('some_label' in m for m in msgs)


def test_verbose_timer_round_records_intervals():
    timer = VerboseTimer()
    timer.round('a')
    timer.round('b')
    assert len(timer.rounds) == 2
    assert all(r >= 0 for r in timer.rounds)
