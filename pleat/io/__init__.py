"""File I/O for pleat graphs: the ``.heg`` half-edge format, the CirclePack
``.p`` format, and the FOLD crease-pattern format."""

from __future__ import annotations

from .circlepack import (
    CirclePackData,
    load_circlepack,
    parse_p_file,
    save_circlepack,
    write_p_file,
)
from .fold import (
    fold_to_graph,
    graph_to_fold,
    load_fold,
    open_in_origami_simulator,
    origami_simulator_button,
    origami_simulator_html,
    save_fold,
)
from .heg import dict_to_graph, graph_to_dict, load_graph, save_graph

__all__ = [
    "graph_to_dict",
    "dict_to_graph",
    "save_graph",
    "load_graph",
    "CirclePackData",
    "parse_p_file",
    "write_p_file",
    "load_circlepack",
    "save_circlepack",
    "graph_to_fold",
    "fold_to_graph",
    "save_fold",
    "load_fold",
    "origami_simulator_html",
    "open_in_origami_simulator",
    "origami_simulator_button",
]
