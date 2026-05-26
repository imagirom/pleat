"""Parse a GomJau-Hogg code into a finite half-edge graph.

Pipeline:

1. :func:`polygon_placement` parses the first stage (e.g. ``"6-3-3"``) into a
   small seed graph by gluing regular polygons together.
2. :func:`apply_transform` interprets each later stage (e.g. ``"m30"``,
   ``"r(h2)"``, ``"r(c3)"``) as one or more affine transforms.
3. :func:`compile_gjh_graph` applies the transforms iteratively, expanding the
   graph until no new tiles fit within the requested bounding box.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

import eucare as ec
from eucare.base import angle_to_axis, signed_area
from eucare.conversions import EHEG_from_nx
from eucare.half import EuclideanPositionHEG, Face, Vertex
from eucare.overlap import group_closeby
from eucare.prototiles import RegularEuclideanTile

_EPS = 0.1


# --- Stage 1: polygon placement ---------------------------------------------


def seed_polygon(n: int) -> EuclideanPositionHEG:
    """Construct an isolated regular n-gon graph (oriented for use as a seed)."""
    G = EuclideanPositionHEG(other=RegularEuclideanTile(n).make_graph(add_positions=True)[0])
    if n == 3:
        # Triangles need to be re-oriented so their border edge lies on the negative x side,
        # matching the convention used by ``starting_border``.
        pos = G.get_position_view(return_vertices=False)
        pos *= -1
        pos -= pos.min(0, keepdims=True)
    return G


def _starting_border(G: EuclideanPositionHEG, seed_face: Face):
    """Return the first border half-edge to attach to (rightmost edge of the seed face)."""
    threshold = _EPS if seed_face.order() != 3 else _EPS  # kept symmetric to the notebook logic
    try:
        h = next(
            h.rev
            for h in seed_face.halfedge_iter()
            if h.rev.on_border() and max(h.orig["pos"][0], h.dest["pos"][0]) < threshold
        )
    except StopIteration as e:
        raise RuntimeError("Seed face has no border edge in the negative-x half-plane") from e

    h0 = h
    while max(h.orig["pos"][0], h.dest["pos"][0]) < _EPS:
        h = h.nex
        if h is h0:
            raise RuntimeError("No border edge found in the positive quadrant during seed walk")
    return h


def polygon_placement(code: str) -> EuclideanPositionHEG:
    """Parse the first stage of a GJH code (polygons separated by ``-`` and ``,``) into a graph.

    Args:
        code: First stage of a GJH code, e.g. ``"6"``, ``"6-3-3"``, ``"4-3-0,4"``.
            A ``0`` in a phase means "skip this attachment slot".

    Returns:
        A small Euclidean half-edge graph containing all placed polygons.
    """
    code = code.replace(" ", "")
    phases = [[int(n) for n in c.split(",")] for c in code.split("-")]

    if len(phases[0]) != 1:
        raise ValueError(f"Seed phase must consist of one polygon; got {phases[0]}")
    G = seed_polygon(phases[0][0])
    seed_face = next(iter(G.faces))

    for phase in phases[1:]:
        # Tag each existing border half-edge so we can later restrict attachment
        # to edges added in the most recent phase only.
        for h in (h for h in G.halfedges if h.on_border()):
            h["old"] = h.attributes.get("old", 0) + 1

        attach_at_list = [_starting_border(G, seed_face)]
        while True:
            attach_at_list.append(attach_at_list[-1].nex)
            if attach_at_list[-1] is attach_at_list[0]:
                break
        attach_at_list = attach_at_list[:-1]
        attach_at_list = [h for h in attach_at_list if h.attributes.get("old", 0) <= 1]

        polys = [seed_polygon(n) if n > 0 else None for n in phase]
        i = 0
        for poly in polys:
            try:
                while not (attach_at_list[i].on_border() and attach_at_list[i] in G.halfedges):
                    i += 1
                attach_at = attach_at_list[i]
            except IndexError as e:
                raise IndexError(
                    f"Not enough new edges to attach polygons {phase} "
                    f"(only {len(attach_at_list)} attachment points available)"
                ) from e
            i += 1
            if poly is None:
                continue
            G.glue_graph_e2e(poly, attach_at, next(h for h in poly.halfedges if h.on_border()))
    return G
