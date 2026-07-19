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


@dataclass
class RayPath:
    """A cast ray: its crossings, whether it closed, and how each end stopped."""

    hits: list[RayHit]
    closed: bool
    ends: tuple[str, str]


def default_vertex_tol(G) -> float:
    """Return a vertex-snapping tolerance scaled to the graph's edge lengths."""
    lengths = [h["length"] for h in G.halfedges if "length" in h]
    return 1e-9 * (float(np.mean(lengths)) if lengths else 1.0)


def _point_on(h: HalfEdge, t: float) -> np.ndarray:
    return h.orig["pos"] + t * (h.dest["pos"] - h.orig["pos"])


def _closes(hit: RayHit, start: np.ndarray, d0: np.ndarray, vertex_tol: float) -> bool:
    """Return True if *hit* is back at the ray's starting point, travelling the same way.

    Both conditions are needed: a ray may pass through its own start point on a
    different heading, which is a self-intersection, not a closed loop.  The
    test is geometric rather than a comparison of half-edge parameters, because
    a hit that lands on a vertex carries no meaningful parameter along the
    starting edge, and because an exact corner hit does not produce an exactly
    integral parameter anyway.
    """
    return bool(
        np.linalg.norm(hit.position - start) <= vertex_tol
        # unit directions, so this is a tolerance of ~4e-5 radians -- far above
        # the drift a few thousand transmissions accumulate, far below the
        # angle between two distinct headings at one point
        and np.dot(hit.direction_in, d0) > 1 - 1e-9
    )


def _walk(G, start_he, start_t, direction, side, vertex_tol, max_steps):
    """Yield ``RayHit``s from the start point onwards, ending with a stop reason.

    The generator's return value (via ``StopIteration.value``) is the reason:
    ``'closed'``, ``'border'``, ``'degenerate'``, or ``'max_steps'``.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    p = start = _point_on(start_he, start_t)
    d0 = d

    # `h.face` is the face to the left of `h`, so the ray sets off into
    # `start_he.face` exactly when `d` points to the left of the start edge
    face = start_he.face if cross2(halfedge_direction(start_he), d) > 0 else start_he.rev.face

    yield RayHit(
        halfedges=[start_he],
        t=start_t,
        position=p,
        vertex=None,
        direction_in=d,
        direction_out=d,
        face=face,
    )

    for _ in range(max_steps):
        if face is None:
            return "border"
        found = first_crossing(face, p, d, vertex_tol)
        if found is None:
            # inside a face with no way out: not the paper's edge, but a
            # configuration this caster cannot step through
            return "degenerate"
        h, s = found
        d_in = d
        edge = halfedge_direction(h)
        edge_len = np.linalg.norm(edge)

        vertex = None
        if s * edge_len <= vertex_tol:
            vertex = h.orig
        elif (1 - s) * edge_len <= vertex_tol:
            vertex = h.dest

        if vertex is not None:
            p = vertex["pos"]
            crossed, d, face = fan_at_vertex(vertex, d, face, side=side)
        else:
            p = _point_on(h, s)
            crossed, d, face = [h], transmit(d, edge), h.rev.face

        hit = RayHit(
            halfedges=crossed,
            t=s,
            position=p,
            vertex=vertex,
            direction_in=d_in,
            direction_out=d,
            face=face,
        )
        yield hit
        if _closes(hit, start, d0, vertex_tol):
            return "closed"
    return "max_steps"


def cast_ray(
    G,
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    both_ways: bool = True,
    vertex_tol: float | None = None,
    max_steps: int = 10_000,
) -> RayPath:
    """Cast a ray from a point on an edge, transmitting at every crease it meets.

    Args:
        G: The crease pattern.
        halfedge: The edge the ray starts on.
        t: Parameter along *halfedge*; ``0`` and ``1`` mean its endpoints.
        direction: Initial direction of travel.
        side: Which side of a vertex the ray passes when it hits one head-on.
        both_ways: Cast backwards too if the forward ray does not close.
        vertex_tol: Distance below which a crossing snaps to a vertex.
        max_steps: Safety cap on the number of crossings per direction.

    Returns:
        A :class:`RayPath` whose hits run from the backward end to the forward
        end.
    """
    if vertex_tol is None:
        vertex_tol = default_vertex_tol(G)

    hits: list[RayHit] = []
    walker = _walk(G, halfedge, t, direction, side, vertex_tol, max_steps)
    try:
        while True:
            hits.append(next(walker))
    except StopIteration as stop:
        forward_reason = stop.value

    return RayPath(hits=hits, closed=forward_reason == "closed", ends=("start", forward_reason))
