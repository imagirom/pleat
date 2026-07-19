"""Cast rays through a crease pattern, transmitting at every crease.

A ray carries only its current point, direction, and face, so every step is
local: no global geometry query is ever made.  Crossing a crease transmits the
direction by ``d - 2(d.u)u``; hitting a vertex is resolved by an angular fan
walk in which the epsilon offset cancels out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .half import Face, HalfEdge, Vertex


class DegenerateRayError(ValueError):
    """The ray hit a configuration this module deliberately does not resolve."""


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Return the scalar cross product ``a x b`` of two 2-D vectors."""
    return float(a[0] * b[1] - a[1] * b[0])


def signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return the counter-clockwise angle from *a* to *b*, in ``(-pi, pi]``."""
    return float(np.arctan2(cross2(a, b), np.dot(a, b)))


def transmit(d: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Transmit direction *d* across a crease of direction *u*.

    Keeps the component crossing the crease and flips the component along it.
    This is *not* the mirror image ``2(d.u)u - d``, which flips the crossing
    component and would send the ray back into the face it came from.
    """
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    return np.asarray(d, dtype=float) - 2 * np.dot(d, u) * u


def halfedge_direction(h: HalfEdge) -> np.ndarray:
    """Return the vector from ``h.orig`` to ``h.dest`` in position space."""
    return h.dest["pos"] - h.orig["pos"]


def fan_at_vertex(
    v: Vertex,
    d: np.ndarray,
    face: Face,
    side: str = "left",
    angle_tol: float = 1e-9,
) -> tuple[list[HalfEdge], np.ndarray, Face | None]:
    """Resolve a ray that hits vertex *v* head-on, as if it passed at distance eps.

    The ray arrives with direction *d* travelling inside *face*.  It is treated
    as passing an infinitesimal distance to the *side* of *v*, transmitting
    through every crease it would meet in quick succession.  The epsilon
    cancels out exactly -- the result does not depend on how small it is, which
    is the whole point of the rule.

    What is *not* exact is the accumulating angle ``theta``, a float sum of
    sector angles.  A crease at folded angle ``theta`` is met at
    ``t = eps*cot(theta)``, so ``theta`` at ``0`` or ``pi`` means the crease is
    met at ``t = +-infinity``: not met.  (``cot`` has opposite signs at the two
    ends -- ``theta -> 0`` is met infinitely far forward, ``theta -> pi``
    infinitely far back -- but neither is met.)  Both are the boundary of the
    open interval the walk runs on, so a ``theta`` whose true value lands there
    would have the branch decided by rounding.  ``angle_tol`` widens the exit:
    ``theta`` within it of ``0`` or ``pi`` counts as "not met" and ends the walk.

    Args:
        v: The vertex the ray hits.
        d: Unit direction of travel on arrival.
        face: The face the ray is travelling in on arrival.
        side: ``"left"`` or ``"right"`` -- which side of *v* the ray passes.
        angle_tol: Radians; how close to ``0`` or ``pi`` the accumulated angle
            may come before the crease counts as not met.

    Returns:
        ``(crossed, d_out, face_out)``: the half-edges transmitted through in
        order, the outgoing direction, and the face the ray leaves into
        (``None`` if it ran off the paper).

    Raises:
        DegenerateRayError: if the ray arrives exactly along a crease, or if it
            wraps the whole vertex (see the spec -- resolving that needs the
            holonomy of the full loop rather than this local walk).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    s = 1.0 if side == "left" else -1.0

    # the boundary half-edge of `face` at v that the offset ray reaches first
    if s > 0:
        g = next(h for h in v.outgoing_iter() if h.face is face)
    else:
        g = next(h for h in v.outgoing_iter() if h.rev.face is face)

    theta = s * signed_angle(d, halfedge_direction(g))
    # theta == 0 or +-pi both mean d is collinear with the crease g: the ray
    # arrived along it.  That is a degenerate *input*, not a mid-walk rounding
    # question, so it raises rather than being absorbed by `angle_tol`.  The
    # threshold tracks `angle_tol` so there is no band in which a near-collinear
    # arrival is silently absorbed as a graze instead of raised.
    if abs(np.sin(theta)) < angle_tol:
        raise DegenerateRayError(f"ray arrives at {v} along a crease")

    d = np.asarray(d, dtype=float)
    crossed: list[HalfEdge] = []
    sign = 1.0
    degree = v.order()

    while angle_tol < theta < np.pi - angle_tol:
        if len(crossed) >= degree:
            raise DegenerateRayError(
                f"ray wraps the whole vertex {v}; resolving this needs the "
                "holonomy of the full loop, which is out of scope"
            )
        d = transmit(d, halfedge_direction(g))
        crossed.append(g)
        face = g.rev.face if s > 0 else g.face
        if face is None:
            return crossed, d, None
        theta += sign * (g.rev["in_angle"] if s > 0 else g.pre["in_angle"])
        sign = -sign
        g = g.rev.nex if s > 0 else g.pre.rev

    return crossed, d, face


@dataclass
class RayHit:
    """One point at which a ray meets the graph.

    ``halfedges`` lists the creases transmitted through here, in order; it has
    more than one entry only at a vertex fan, and may be empty when the ray
    merely grazes a corner.  ``t`` is the parameter along ``halfedges[0]``.
    """

    halfedges: list[HalfEdge]
    t: float
    position: np.ndarray
    vertex: Vertex | None
    direction_in: np.ndarray
    direction_out: np.ndarray
    face: Face | None


def first_crossing(
    face: Face,
    p: np.ndarray,
    d: np.ndarray,
    vertex_tol: float,
) -> tuple[HalfEdge, float] | None:
    """Return the first crossing of the ray ``p + t*d`` with the boundary of *face*.

    Only *face* is inspected, which is what keeps the caster local.

    ``vertex_tol`` is a **distance** in position space, and both comparisons it
    appears in are done in distance:

    * along the ray, *d* is normalised first so the ray parameter ``t`` is a
      distance; a crossing within ``vertex_tol`` of *p* is discarded, which is
      the slack that stops a ray leaving a vertex from immediately re-detecting
      the edge it just left.  Normalising also makes the result independent of
      ``|d|``, which callers have no reason to control.
    * across the edge, the edge parameter ``s`` is normalised to ``[0, 1]``, so
      the tolerance is converted to ``vertex_tol / |e|`` before comparing.  An
      ``s`` outside ``[0, 1]`` by less than that is a crossing which passes
      within ``vertex_tol`` of an endpoint; it is kept and clamped, so the
      caller sees a vertex hit rather than a miss.

    Args:
        face: The face the ray is currently inside.
        p: Point on the ray.
        d: Direction of travel; need not be normalised.
        vertex_tol: Distance; slack for both comparisons above.

    Returns:
        ``(halfedge, s)`` where *s* is the parameter along the half-edge, or
        ``None`` if the ray leaves through no edge (which should not happen in
        a well-formed face).
    """
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    best: tuple[HalfEdge, float] | None = None
    best_t = np.inf
    for h in face.halfedge_iter():
        a = h.orig["pos"]
        e = h.dest["pos"] - a
        edge_len = float(np.linalg.norm(e))
        denom = cross2(d, e)
        # |denom| = |e| sin(angle), so the threshold scales with the edge
        if edge_len == 0.0 or abs(denom) < 1e-15 * edge_len:
            continue  # degenerate, or the ray runs parallel to this edge
        t = cross2(a - p, e) / denom
        if t <= vertex_tol or t >= best_t:
            continue
        s = cross2(a - p, d) / denom
        if not -vertex_tol / edge_len <= s <= 1 + vertex_tol / edge_len:
            continue
        best, best_t = (h, float(np.clip(s, 0.0, 1.0))), t
    return best
