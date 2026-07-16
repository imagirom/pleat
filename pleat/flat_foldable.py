"""Flat-foldability tests for crease patterns.

This module collects geometric tests that determine whether a (flat) crease
pattern can be folded into a flat origami model without tearing or stretching.

Currently implemented:

* :func:`kawasaki_sum` — alternating sum of sector angles around an interior
  vertex; vanishes iff the vertex satisfies Kawasaki's theorem.
* :func:`max_kawasaki_sum` — convenience reduction over a graph or vertex set.

Future additions (e.g. Maekawa's theorem, big-little-big, etc.) belong here.
"""

from __future__ import annotations

import numpy as np

from .half import HalfEdgeGraph, Vertex


def kawasaki_sum(v: Vertex) -> float:
    """Return the Kawasaki alternating angle sum at vertex *v*.

    A flat-foldable interior vertex satisfies Kawasaki's theorem: the
    alternating sum of consecutive sector angles is zero. The sum is computed
    from the in-angles of the halfedges incoming at *v*; angles are wrapped
    into ``[0, 2π)`` before being combined with alternating signs.

    Border vertices are not constrained by Kawasaki's theorem; callers should
    filter them out (see :func:`max_kawasaki_sum`).
    """
    angles = np.abs(np.array([e["in_angle"] for e in v.incoming_iter()]))
    assert len(angles) % 2 == 0
    return np.sum(((angles + 2 * np.pi) % (2 * np.pi)) * (-1) ** np.arange(len(angles)))


def max_kawasaki_sum(vertices) -> float:
    """Return the largest absolute Kawasaki sum over interior vertices.

    *vertices* may be a :class:`HalfEdgeGraph` (interior vertices are taken
    automatically) or any iterable of vertices.
    """
    if isinstance(vertices, HalfEdgeGraph):
        vertices = [v for v in vertices.vertices if not v.on_border()]
    return np.max([kawasaki_sum(v) for v in vertices])
