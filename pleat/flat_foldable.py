"""Flat-foldability tests for crease patterns.

This module collects geometric tests that determine whether a (flat) crease
pattern can be folded into a flat origami model without tearing or stretching.

Currently implemented:

* :func:`kawasaki_sum` — alternating sum of sector angles around an interior
  vertex; vanishes iff the vertex satisfies Kawasaki's theorem.
* :func:`max_kawasaki_sum` — convenience reduction over a graph or vertex set.
* :func:`maekawa_check` — mountain/valley count balance at an interior vertex.
* :func:`folded_crease_angles` — where each crease lands in the folded state.
* :func:`local_assignment_valid` — full vertex-wise test of a crease
  assignment via the crimp recursion (big-little-big).
* :func:`is_locally_flat_foldable` — all local conditions over a whole graph,
  with per-vertex diagnostics.

Only *local* (per-vertex) conditions live here: they are necessary but not
sufficient for global flat foldability, which additionally requires a
non-self-intersecting layer ordering — an NP-hard problem, tackled by the ILP
in :func:`pleat.overlap.fold_complete`.

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


def folded_crease_angles(v: Vertex) -> np.ndarray:
    """Return the folded positions ``psi`` of the creases around *v*.

    ``psi_k = psi_{k-1} + (-1)^k * a_k`` over the sector angles ``a_k`` in
    counter-clockwise order, so ``psi[k]`` is where crease ``k`` lands in the
    folded state.  ``psi[-1]`` is exactly :func:`kawasaki_sum`, so Kawasaki's
    theorem is the statement that this cycle closes rather than an independent
    condition.
    """
    angles = np.abs(np.array([h["in_angle"] for h in v.incoming_iter()]))
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return np.cumsum(angles * (-1.0) ** np.arange(len(angles)))


def _cluster_margin(values: np.ndarray, tol: float) -> float:
    """Return the smallest gap between distinct clusters of *values*."""
    gaps = np.diff(np.sort(values))
    distinct = gaps[gaps > tol]
    return float(distinct.min()) if len(distinct) else 0.0


def _crimp(angles: list[float], mv: list[int], i: int) -> tuple[list[float], list[int]]:
    """Fold sector *i* inside: drop creases ``i`` and ``i + 1``, merge three sectors.

    Both lists are rotated so the crimp sits at the front, which keeps sectors
    and creases aligned without any modular index juggling.
    """
    n = len(angles)
    k = (i - 1) % n
    a = angles[k:] + angles[:k]  # a[0], a[1], a[2] = sectors i-1, i, i+1
    m = mv[k:] + mv[:k]  # m[0], m[1], m[2] = creases i-1, i, i+1
    return [a[0] - a[1] + a[2]] + a[3:], [m[0]] + m[3:]


def _crimp_ok(angles: list[float], mv: list[int], tol: float) -> bool:
    """Return True if some sequence of crimps folds this vertex flat.

    ``angles[i]`` is the sector between creases ``mv[i]`` and ``mv[i + 1]``
    cyclically.  Backtracks over every *weakly* minimal sector: with a tie,
    big-little-big does not force the bounding creases to differ, so committing
    to one candidate would reject valid assignments.  Degrees are at most about
    twelve, so exhaustive backtracking is free.
    """
    n = len(angles)
    if n <= 2:
        # The two survivors are the halves of one straight crease in the folded
        # model, so they must carry the same assignment.  Without this the
        # recursion accepts every degree-4 vertex, 2M-2V included.
        return mv[0] == mv[1]
    smallest = min(angles)
    for i in range(n):
        if angles[i] > smallest + tol:
            continue
        if mv[i] == mv[(i + 1) % n]:
            continue  # big-little-big: the bounding creases must differ
        new_angles, new_mv = _crimp(angles, mv, i)
        if new_angles[0] < -tol:
            continue  # the crimp would need more paper than there is
        new_angles[0] = max(new_angles[0], 0.0)
        if _crimp_ok(new_angles, new_mv, tol):
            return True
    return False


def local_assignment_valid(v: Vertex, tol: float = 1e-9) -> tuple[bool, float]:
    """Check that the crease assignment at interior vertex *v* folds flat locally.

    Returns:
        ``(valid, margin)``.  *margin* is the smallest gap between distinct
        folded crease positions (see :func:`folded_crease_angles`); a margin near
        *tol* means the vertex is symmetric enough that the verdict depends on
        tie-breaking, not that it is wrong.

    Requires ``pleat.overlap.CREASE_ASSIGNMENT`` on every half-edge at *v*.
    """
    from .overlap import CREASE_ASSIGNMENT

    incoming = list(v.incoming_iter())
    if len(incoming) % 2 != 0:
        return False, 0.0

    # ``h['in_angle']`` for incoming ``h`` is the sector *clockwise* of its
    # crease, i.e. between crease i-1 and crease i.  The crimp recursion wants
    # angles[i] between crease i and crease i+1, hence the rotation by one.
    angles = [abs(h["in_angle"]) % (2 * np.pi) for h in incoming]
    angles = angles[1:] + angles[:1]
    mv = [h[CREASE_ASSIGNMENT] for h in incoming]

    psi = folded_crease_angles(v)
    margin = _cluster_margin(psi, tol)
    eff_tol = max(tol, 1e-8)
    if abs(psi[-1]) > eff_tol * len(angles):
        return False, margin  # Kawasaki: the cycle does not close
    return _crimp_ok(angles, mv, eff_tol), margin


def is_locally_flat_foldable(graph: HalfEdgeGraph, tol: float = 1e-8) -> tuple[bool, dict[Vertex, str]]:
    """Check the local flat-foldability conditions at every interior vertex.

    Checks even degree and Kawasaki's theorem, plus the full crimp recursion
    (:func:`local_assignment_valid`) at vertices where all half-edges carry a
    crease assignment. These conditions are necessary but not sufficient:
    global flat foldability also needs a
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
        if all(CREASE_ASSIGNMENT in h.attributes for h in v.outgoing_iter()):
            valid, margin = local_assignment_valid(v)
            if not valid:
                violations[v] = f"crease assignment does not fold flat (margin {margin:.3e})"
    return not violations, violations
