"""Flat-foldability tests for crease patterns.

This module collects geometric tests that determine whether a (flat) crease
pattern can be folded into a flat origami model without tearing or stretching.

Currently implemented:

* :func:`kawasaki_sum` — alternating sum of sector angles around an interior
  vertex; vanishes iff the vertex satisfies Kawasaki's theorem.
* :func:`max_kawasaki_sum` — convenience reduction over a graph or vertex set.
* :func:`maekawa_check` — mountain/valley count balance at an interior vertex.
* :func:`is_locally_flat_foldable` — all local conditions over a whole graph,
  with per-vertex diagnostics.

Only *local* (per-vertex) conditions live here: they are necessary but not
sufficient for global flat foldability, which additionally requires a
non-self-intersecting layer ordering — an NP-hard problem, tackled by the ILP
in :func:`pleat.overlap.fold_complete`.

Future additions (e.g. big-little-big) belong here.
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


def maekawa_check(v: Vertex) -> bool:
    """Check Maekawa's theorem at interior vertex *v*.

    A flat-foldable interior vertex has ``|#mountains - #valleys| == 2``.
    Requires a crease assignment (``pleat.overlap.CREASE_ASSIGNMENT``, with
    values ``MOUNTAIN = 1`` / ``VALLEY = -1``) on all half-edges at *v*.
    """
    from .overlap import CREASE_ASSIGNMENT

    return abs(sum(h[CREASE_ASSIGNMENT] for h in v.outgoing_iter())) == 2


def is_locally_flat_foldable(graph: HalfEdgeGraph, tol: float = 1e-8) -> tuple[bool, dict[Vertex, str]]:
    """Check the local flat-foldability conditions at every interior vertex.

    Checks even degree and Kawasaki's theorem, plus Maekawa's theorem at
    vertices where all half-edges carry a crease assignment. These conditions
    are necessary but not sufficient: global flat foldability also needs a
    non-self-intersecting layer ordering, which is NP-hard to decide (see
    :func:`pleat.overlap.fold_complete` for the ILP-based solver).

    Returns:
        ``(ok, violations)`` where ``violations`` maps each failing vertex to
        a description of the violated condition.
    """
    from .overlap import CREASE_ASSIGNMENT

    violations: dict[Vertex, str] = {}
    for v in graph.vertices:
        if v.on_border():
            continue
        if v.order() % 2 != 0:
            violations[v] = f"odd degree ({v.order()})"
            continue
        ks = kawasaki_sum(v)
        if abs(ks) > tol:
            violations[v] = f"Kawasaki sum {ks:.3e} exceeds tolerance"
            continue
        if all(CREASE_ASSIGNMENT in h.attributes for h in v.outgoing_iter()) and not maekawa_check(v):
            violations[v] = "Maekawa's theorem violated (|#mountains - #valleys| != 2)"
    return not violations, violations
