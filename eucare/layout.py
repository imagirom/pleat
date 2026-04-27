"""Vertex position optimisations for half-edge graphs.

Provides helpers to find the rotation angle that minimises the bounding-box
height of a graph (useful before rendering elongated tilings) and a
``min_edge_length`` query used to set tolerances elsewhere.
"""
from __future__ import annotations

from copy import copy

import numpy as np


def angle_to_height(G, angle: float) -> float:
    """Return the height of the axis-aligned bounding box of *G*'s border after rotating by *angle*."""
    border_positions = np.array([v['pos'] for v in G.border_vertex_iter()])
    rot_border_positions = border_positions @ np.array([[np.cos(angle)], [-np.sin(angle)]])
    return np.max(rot_border_positions) - np.min(rot_border_positions)


def optimal_rotation(G, angle_offset: float = 0, steps: int = 10000) -> float:
    """Return the rotation angle minimising the bounding-box height of *G*'s border.

    Args:
        G: A graph with vertex ``pos`` attributes and a border iterator.
        angle_offset: Added to the optimal angle (e.g. ``pi/2`` to optimise width instead).
        steps: Number of equally-spaced angles in ``[0, pi)`` to test.
    """
    border_positions = np.array([v['pos'] for v in G.border_vertex_iter()])

    def _angle_to_height(angle: float) -> float:
        rot_border_positions = border_positions @ np.array([[np.cos(angle)], [-np.sin(angle)]])
        return np.max(rot_border_positions) - np.min(rot_border_positions)

    angles = np.linspace(0, np.pi, steps)
    heights = [_angle_to_height(a) for a in angles]
    angle = angles[np.argmin(heights)] + angle_offset
    return angle


def rotate_graph(G, angle: float) -> None:
    """Rotate every vertex of *G* by *angle* in place."""
    ps = G.get_position_view(return_vertices=False)
    ps[:] = ps @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def optimize_rotation(G, angle_offset: float = 0) -> float:
    """Rotate *G* in place to minimise its bounding-box height; return the applied angle."""
    angle = optimal_rotation(G, angle_offset)
    rotate_graph(G, angle)
    return angle


def min_edge_length(G, include_border: bool = True) -> float:
    """Return the shortest edge length in *G*.

    Args:
        G: A graph with vertex ``pos`` attributes.
        include_border: If False, edges where either side is a border are skipped.
    """
    edges = copy(G.halfedges)
    min_length = np.inf
    while edges:
        e = edges.pop()
        edges.remove(e.rev)
        if not include_border and (e.on_border() or e.rev.on_border()):
            continue
        min_length = min(((e.orig['pos'] - e.dest['pos']) ** 2).sum(), min_length)
    return np.sqrt(min_length)
