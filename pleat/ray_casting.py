"""Cast rays through a crease pattern, transmitting at every crease.

A ray carries only its current point, direction, and face, so every step is
local: no global geometry query is ever made.  Crossing a crease transmits the
direction by ``d - 2(d.u)u``; hitting a vertex is resolved by an angular fan
walk in which the epsilon offset cancels out.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .cutting import pointinpolygon
from .half import Face, HalfEdge, Vertex
from .overlap import line_segment_intersections

if TYPE_CHECKING:
    from .half import EuclideanPositionHEG


class DegenerateRayError(ValueError):
    """The ray hit a configuration this module deliberately does not resolve.

    This is raised for a *degenerate input or configuration*: a direction along
    the start edge, a zero-length direction, an arrival along a crease, a fan
    that wraps its vertex, a self-crossing trajectory.  It is **not** the
    channel for a ray that merely stopped early -- that is reported in
    :attr:`RayPath.ends`, whose non-terminal values are ``"stalled"`` and
    ``"max_steps"``.  A cast that returns has raised nothing; a cast that raises
    returns nothing.  The two never overlap, which is why the end value is
    ``"stalled"`` rather than ``"degenerate"``.
    """


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

    Raises:
        DegenerateRayError: if *u* has zero length, which has no direction to
            transmit across and would otherwise return ``nan`` silently.
    """
    u = np.asarray(u, dtype=float)
    norm = float(np.linalg.norm(u))
    if norm == 0.0:
        raise DegenerateRayError("cannot transmit across a zero-length crease")
    u = u / norm
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
        raise DegenerateRayError(f"ray arrives at {tuple(map(float, v['pos']))} along a crease")

    d = np.asarray(d, dtype=float)
    crossed: list[HalfEdge] = []
    sign = 1.0
    degree = v.order()

    while angle_tol < theta < np.pi - angle_tol:
        if len(crossed) >= degree:
            raise DegenerateRayError(
                f"ray wraps the whole vertex at {tuple(map(float, v['pos']))}; resolving this needs the "
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


def sector_at_vertex(v: Vertex, d: np.ndarray, angle_tol: float = 1e-9) -> Face | None:
    """Return the face at *v* whose sector contains the direction *d*.

    This resolves a ray *departing* from a vertex, the mirror of
    :func:`fan_at_vertex`'s arrival: the sectors at *v* partition the
    directions, and the one holding *d* is the face the ray sets off into.  It
    is the sector, not the side of any one incident edge, that decides -- a node
    of degree *n* has *n* sectors, and the side of one edge only tells the two
    apart that touch it.

    The face of a sector is read off its clockwise-most edge: ``h.face`` lies to
    the left of ``h``, so it is the sector running counter-clockwise from *h*,
    and the sector holding *d* is the one whose *h* is the smallest positive
    turn away.  ``None`` when that face is off the paper, which is how a ray
    aimed out through a border node reports "border" at once.

    Args:
        v: The vertex the ray sets off from.
        d: Direction of travel; need not be normalised.
        angle_tol: Radians; how close *d* may come to an incident crease before
            it counts as running along it.

    Raises:
        DegenerateRayError: if *d* runs along an incident crease, where no
            sector holds it and picking one of the two either side would send
            the ray along an edge instead of across it.
    """
    best: Face | None = None
    best_angle = np.inf
    for h in v.outgoing_iter():
        angle = signed_angle(halfedge_direction(h), d)
        if abs(angle) < angle_tol:
            raise DegenerateRayError(f"ray sets off from {tuple(map(float, v['pos']))} along a crease")
        if angle < 0:
            angle += 2 * np.pi
        if angle < best_angle:
            best, best_angle = h.face, angle
    return best


@dataclass
class RayHit:
    """One point at which a ray meets the graph.

    ``halfedges`` lists the creases transmitted through here, in order; it has
    more than one entry only at a vertex fan, and may be empty when the ray
    merely grazes a corner.

    ``t`` is the parameter along the edge the ray *arrived* on, which is not
    always ``halfedges[0]``:

    * at a plain crossing the two coincide -- ``t`` runs along ``halfedges[0]``;
    * at a vertex hit ``t`` is the parameter of the crossing ``first_crossing``
      found on the edge it found it on (so ``t`` is within a tolerance of ``0``
      or ``1``, not exactly either), while ``halfedges`` is the fan's crossing
      list, whose first entry is a different edge in general -- and is empty
      when the fan crosses nothing;
    * on the starting hit ``t`` is the caller's parameter along the start edge.

    Use ``position`` rather than ``t`` when what is wanted is the point.
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

    Raises:
        DegenerateRayError: if *d* has zero length.  Normalising it would warn
            and then return ``nan``, and every comparison below would be False,
            so the ray would silently report "no way out of this face".
    """
    d = np.asarray(d, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise DegenerateRayError("ray has no direction: |d| == 0")
    d = d / norm
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
    """A cast ray: its crossings, whether it closed, and how each end stopped.

    Each entry of ``ends`` is one of:

    * ``"closed"`` -- the ray came back to its start on its original heading;
    * ``"border"`` -- it ran off the paper;
    * ``"max_steps"`` -- it was still going when the safety cap ran out;
    * ``"stalled"`` -- it stopped inside a face with no edge to leave through
      (a malformed face, or a corner narrower than ``vertex_tol``);
    * ``"start"`` -- ``ends[0]`` only: that end was never traced, because the
      cast was one-way.

    Only the first two mean the end was traced to completion, and more reasons
    may be added, so code that wants to know whether an end failed must test
    ``ends[i] not in ("closed", "border")`` rather than looking for
    ``"max_steps"`` specifically.  None of these is a
    :class:`DegenerateRayError`: a stopped ray is returned, a degenerate one is
    raised, and the two channels do not overlap.

    Because the backward half retraces the trajectory's own crossing of the
    start point -- ``transmit(-d, E)`` from inside an edge, the reversed fan
    passage at a node -- the two halves are one reversible line: whichever end
    closes, the other has traced the same curve, so ``ends`` is always
    ``("closed", "closed")`` or two non-closing reasons.  ``closed`` and
    ``"closed" in ends`` therefore agree.
    """

    hits: list[RayHit]
    closed: bool
    ends: tuple[str, str]


def default_vertex_tol(G: "EuclideanPositionHEG") -> float:
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

    The heading compared is ``direction_out``, not ``direction_in``.  The start
    hit does not transmit, so its ``direction_out`` *is* ``d0``: the direction
    the ray departed in.  Returning to the start means arriving from the far
    side of the start edge and transmitting across it, so the direction that
    has to match ``d0`` for the trajectory to be one closed curve is the one it
    departs on again -- ``direction_out``.  The arriving direction is
    ``transmit(d0, start_edge)``, which equals ``d0`` only when the loop crosses
    the start edge perpendicularly.
    """
    return bool(
        np.linalg.norm(hit.position - start) <= vertex_tol
        # unit directions, so this is a tolerance of ~4e-5 radians -- far above
        # the drift a few thousand transmissions accumulate, far below the
        # angle between two distinct headings at one point.  Measured on the
        # closed-loop fixture rotated by 0.3/1.0/2.345 rad, `1 - dot` stayed
        # between 0 and 2.2e-16; the binding tolerance for a long loop is the
        # position check above (4.6e-16 of drift over 8 steps against a
        # `vertex_tol` of 1.03e-9), never this one.
        and np.dot(hit.direction_out, d0) > 1 - 1e-9
    )


def _start_vertex(start_he: HalfEdge, start_t: float, vertex_tol: float) -> Vertex | None:
    """Return the endpoint of *start_he* the start point snaps to, or ``None``.

    ``t = 0`` and ``t = 1`` are documented as meaning the endpoints, so the
    start is snapped to a vertex on the same rule every other crossing is -- a
    *distance* of at most ``vertex_tol`` from an end.  Doing it here rather
    than in :func:`add_ray_creases` is what keeps ``RayHit.vertex`` honest
    ("set iff the hit is exactly at a vertex"): the start hit is the one hit
    that never passes through :func:`first_crossing`, so it is the one hit
    whose parameter is the caller's raw ``t`` and can land anywhere.  Without
    this the materializer inserts a *second* vertex on top of an existing one,
    giving a zero-length edge and two zero-area faces on a graph that still
    passes ``check_consistency``.
    """
    edge_len = float(np.linalg.norm(halfedge_direction(start_he)))
    if start_t * edge_len <= vertex_tol:
        return start_he.orig
    if (1 - start_t) * edge_len <= vertex_tol:
        return start_he.dest
    return None


#: `_walk` resolves the start face itself unless it is handed one; `None` is a
#: meaningful face (off the paper), so "not given" needs a value of its own.
_RESOLVE = object()


def _walk(start_he, start_t, direction, side, vertex_tol, max_steps, angle_tol, start_face=_RESOLVE):
    """Yield ``RayHit``s from the start point onwards, ending with a stop reason.

    The generator's return value (via ``StopIteration.value``) is the reason:
    ``'closed'``, ``'border'``, ``'stalled'``, or ``'max_steps'``.  A path
    that finishes never reports ``'max_steps'``: the border test is made on
    entry and after every step, so it costs no iteration of its own.

    ``start_face`` overrides the face the ray sets off into.  Only the backward
    half of a node start passes it: there the departing direction and the face
    both come out of one fan call, and re-deriving the face from the direction
    would be a second, weaker answer to a question already settled (a fan that
    grazes leaves along a direction its own exit sector does not contain).
    """
    d = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise DegenerateRayError("ray has no direction: |d| == 0")
    # `_closes` reads the dot product of two of these as an angle, and
    # `first_crossing` reads the ray parameter as a distance: both need unit `d`
    d = d / norm
    d0 = d

    start_edge = halfedge_direction(start_he)
    edge_len = float(np.linalg.norm(start_edge))

    start_vertex = _start_vertex(start_he, start_t, vertex_tol)
    p = start = start_vertex["pos"] if start_vertex is not None else _point_on(start_he, start_t)

    if start_face is not _RESOLVE:
        face = start_face
    elif start_vertex is not None:
        # At a node the start edge is one of several, and which side of *it* `d`
        # points only tells apart the two sectors touching it: every other
        # direction would be sent into a face it points out of and stall.  The
        # sector holding `d` is what decides, and it does not care which
        # incident half-edge the caller named the node with.
        face = sector_at_vertex(start_vertex, d, angle_tol)
    else:
        # `h.face` is the face to the left of `h`, so a ray from strictly inside
        # the edge sets off into `start_he.face` exactly when `d` points left
        towards_left = cross2(start_edge, d)
        if abs(towards_left) < 1e-12 * edge_len:
            # neither side is the one the ray goes into; picking one silently would
            # send the ray along an edge it is supposed to be crossing
            raise DegenerateRayError("ray sets off along its own start edge")
        face = start_he.face if towards_left > 0 else start_he.rev.face

    yield RayHit(
        halfedges=[start_he],
        t=start_t,
        position=p,
        vertex=start_vertex,
        direction_in=d,
        direction_out=d,
        face=face,
    )

    if face is None:
        return "border"

    for _ in range(max_steps):
        found = first_crossing(face, p, d, vertex_tol)
        if found is None:
            # inside a face with no way out: not the paper's edge, but a
            # configuration this caster cannot step through
            return "stalled"
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
            crossed, d, face = fan_at_vertex(vertex, d, face, side=side, angle_tol=angle_tol)
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
        if face is None:
            return "border"
    return "max_steps"


def _reoriented(backward: list[RayHit], k: int) -> RayHit:
    """Return backward-pass hit ``backward[k]`` re-expressed in the forward sense.

    The backward pass records its fields in the sense it was travelling, which
    is the opposite of the order the spliced ``hits`` runs in.  Reversing the
    order alone would leave a list that looks like one trajectory but is two
    glued back to back.

    Directions negate *and* swap.  ``face`` becomes ``backward[k - 1].face``:
    the backward pass at ``k`` went ``face[k-1] -> hit k -> face[k]``, so the
    same hit traversed forwards is entered from the other side and leaves into
    ``face[k-1]``.  ``halfedges``, ``t``, ``position`` and ``vertex`` are
    orientation-independent and carry over unchanged.

    A copy is returned rather than the hit being mutated in place, so casting
    twice from the same start yields the same thing twice.
    """
    # `backward[0]` is the shared start point, which the splice drops in favour
    # of the forward pass's copy; it is therefore never re-oriented, and
    # `k - 1` is always a real index.
    assert k >= 1, "the backward pass's start hit is dropped, not re-oriented"
    hit = backward[k]
    return replace(
        hit,
        direction_in=-hit.direction_out,
        direction_out=-hit.direction_in,
        face=backward[k - 1].face,
    )


def cast_ray(
    G: "EuclideanPositionHEG",
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    both_ways: bool = True,
    vertex_tol: float | None = None,
    max_steps: int = 10_000,
    angle_tol: float = 1e-9,
) -> RayPath:
    """Cast a ray from a point on an edge, transmitting at every crease it meets.

    Args:
        G: The crease pattern.
        halfedge: The edge the ray starts on.
        t: Parameter along *halfedge*; ``0`` and ``1`` mean its endpoints, and a
            *t* within ``vertex_tol`` of an end snaps to that vertex, so
            starting at a node is this same entry point rather than another one.
            From strictly inside the edge the ray sets off into the face on the
            side of *halfedge* that *direction* points to; from a node it sets
            off into the sector that holds *direction*
            (:func:`sector_at_vertex`), which is every direction at the node and
            not only the two sectors touching *halfedge* -- so which incident
            half-edge names the node does not affect the ray.
        direction: Initial direction of travel.
        side: Which side of a vertex the ray passes when it hits one head-on.
        both_ways: Cast backwards too if the forward ray does not close.  The
            backward half retraces the trajectory's crossing of the start
            point, which is again the two cases above: from strictly inside
            *halfedge* it departs along ``-transmit(direction, halfedge)``;
            from a node it departs along the reverse of the fan passage through
            that node -- the fan applied to the reversed ray, which mentions no
            start edge, so this half too is the same whichever incident
            half-edge names the node.  At a degree-2 node, a point that is
            effectively mid-edge, the fan gives exactly
            ``-transmit(direction, E)`` back and the two rules coincide.  A
            direction that leaves the paper at a border node has no passage to
            reverse, and that end reports ``"border"`` at once.
        vertex_tol: Distance below which a crossing snaps to a vertex.
        max_steps: Safety cap on the number of crossings per direction.
        angle_tol: Radians; how close the fan walk's accumulated angle may come
            to ``0`` or ``pi`` before the crease counts as not met (see
            :func:`fan_at_vertex`).

    Returns:
        A :class:`RayPath` whose hits run from the backward end to the forward
        end.  Each entry of its ``ends`` is ``"closed"``, ``"border"``,
        ``"max_steps"``, ``"stalled"``, or -- for ``ends[0]`` on a one-way
        cast -- ``"start"``; see :class:`RayPath`, and note that testing for
        failure means ``not in ("closed", "border")``.

    Raises:
        DegenerateRayError: if *direction* has zero length; if it runs along
            *halfedge* from strictly inside it, so that neither side is the one
            the ray sets off into; if it runs along any incident crease of a
            start node, so that no sector holds it; or if the ray meets a vertex
            configuration :func:`fan_at_vertex` cannot resolve.
    """
    if vertex_tol is None:
        vertex_tol = default_vertex_tol(G)

    def run(d, side_, start_face=_RESOLVE):
        collected: list[RayHit] = []
        walker = _walk(halfedge, t, d, side_, vertex_tol, max_steps, angle_tol, start_face)
        try:
            while True:
                collected.append(next(walker))
        except StopIteration as stop:
            return collected, stop.value

    forward, forward_reason = run(direction, side)
    if forward_reason == "closed":
        # a closed loop is its own continuation: the forward pass already
        # traced everything a backward one could reach, so both ends are closed
        return RayPath(hits=forward, closed=True, ends=("closed", "closed"))
    if not both_ways:
        return RayPath(hits=forward, closed=False, ends=("start", forward_reason))

    # The side is mirrored for both cases below: the backward pass runs down the
    # same offset line the other way round, so what was +eps to the left
    # travelling forwards is -eps, on the right, travelling back.
    back_side = "right" if side == "left" else "left"
    d = np.asarray(direction, dtype=float)
    start_vertex = _start_vertex(halfedge, t, vertex_tol)

    if start_vertex is None:
        # From strictly inside the edge the backward heading is `transmit(-d, E)`,
        # not `-d`.  The start point lies *on* the start edge `E`, so the full
        # trajectory crosses `E` there: it arrives from the far side along
        # `transmit(d, E)` and departs along `d`.  Retracing that arrival means
        # leaving along `-transmit(d, E)`.  This is forced, not cosmetic:
        # materializing makes the start point a degree-4 vertex -- the two halves
        # of `E` plus the two rim segments -- and only the transmitted heading
        # makes that vertex satisfy Kawasaki.  It is also what makes the early
        # return above correct: the two halves are then one line, so a forward
        # pass that closed really has traced the whole trajectory.
        back_direction, back_face = -transmit(d, halfedge_direction(halfedge)), _RESOLVE
    else:
        # At a node there is no canonical `E` -- just whichever incident half-edge
        # the caller named the node with -- and `transmit(., E)` would make the
        # whole backward half depend on that arbitrary choice.  A ray through a
        # vertex is instead the degenerate case the fan rule exists for: it passes
        # an infinitesimal distance to one side, transmitting through a specific
        # set of creases.  So the trajectory through the node is a fan passage,
        # and retracing it is that same passage reversed: the reversed ray
        # *arrives* at the node along `-d`, out of the sector holding `d` (since
        # `-(-d) = d` points back into the face it came through), and the fan
        # hands back the heading and face it leaves on.  No start edge anywhere in
        # that, which is the point.  At a degree-2 node -- a point that is
        # effectively mid-edge -- the fan crosses the one other crease and gives
        # exactly `-transmit(d, E)` back, so this is the same rule as above and
        # not a second one.
        sector = sector_at_vertex(start_vertex, d, angle_tol)
        if sector is None:
            # `d` leaves the paper, so the forward half is already "border" at the
            # start hit and the reversed ray would have to arrive from off the
            # paper: there is no passage to reverse, and the backward end is the
            # border too.  `_walk` reports that from `start_face=None` alone, so
            # the heading below is never used for anything.
            back_direction, back_face = -d, None
        else:
            _, back_direction, back_face = fan_at_vertex(start_vertex, -d, sector, back_side, angle_tol)

    backward, backward_reason = run(back_direction, back_side, back_face)
    # Both passes emit the shared start point first.  Keep the *forward* copy:
    # its directions and its face are already in the trajectory's sense, while
    # the backward one's point the wrong way and sit on the wrong side of the
    # start edge.  The rest of the backward hits are re-oriented as they are
    # spliced, so the whole list reads as one trajectory.
    spliced = [_reoriented(backward, k) for k in range(len(backward) - 1, 0, -1)]
    # `backward_reason` cannot be `"closed"` here: the two halves are one
    # reversible line (that is what `transmit(-d, E)` above buys), so a backward
    # pass that came round would mean the forward pass had too, and the early
    # return above would already have fired.  Measured: 0 of 9975 casts.
    return RayPath(hits=spliced + forward, closed=False, ends=(backward_reason, forward_reason))


RAY_CREASE = "ray_crease"


def _canonical(h: HalfEdge) -> HalfEdge:
    """Return one representative of the undirected edge, the same for both halves.

    Hits reference whichever half the ray happened to cross, so grouping them
    needs a key that does not depend on that choice.  ``id()`` would serve while
    the objects are alive, but the graph's own ``"id"`` attribute is stable,
    readable and reproducible across runs, so use that instead.
    """
    return h if h["id"] < h.rev["id"] else h.rev


def _face_containing_segment(a: Vertex, b: Vertex) -> Face | None:
    """Return the face the segment ``a -> b`` runs through.

    Usually there is exactly one common face.  Two vertices bound two faces
    exactly when they are a 2-cut of the subdivision -- two distinct boundary
    paths between them, one on each side -- which a planar graph that is not
    3-connected is free to contain, and which subdividing faces along ray chords
    is free to create.  So the midpoint decides between them.

    No sweep against this module has produced the case (0 of ~19k segments
    across ~2200 casts), and the ``existing``-edge reuse in phase 2 intercepts
    the *one-path* version of it, where the two vertices are adjacent.  Neither
    is an argument that it cannot happen: reuse only fires when there is an edge
    ``a -> b``, and a 2-cut joined by two paths of length above one has none.
    Unobserved is not impossible, so the fallback stays.
    """
    candidates = list(a.common_faces_iter(b))
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    midpoint = 0.5 * (a["pos"] + b["pos"])
    for f in candidates:
        polygon = np.stack([w["pos"] for w in f.vertex_iter()])
        if pointinpolygon(midpoint[0], midpoint[1], polygon):
            return f
    # only reachable if the midpoint sits on a shared boundary, where the two
    # candidates are equally (in)valid
    return candidates[0]


def _touches(point: np.ndarray, segment: np.ndarray, tol: float) -> bool:
    """Return True if *point* is within *tol* of an endpoint of *segment*."""
    return bool(min(np.linalg.norm(point - segment[0]), np.linalg.norm(point - segment[1])) <= tol)


def _reject_self_crossing(path: RayPath, vertex_tol: float) -> None:
    """Raise if the cast trajectory crosses itself, before anything is materialized.

    The ray is cast to completion on the untouched pattern, so its whole
    trajectory is known as the polyline through the hit positions.  Whether it
    crosses itself is therefore purely geometric, and can be decided before the
    graph is touched -- which is the point: :func:`add_ray_creases` mutates in
    two phases, and discovering the crossing halfway through phase 2 would leave
    *G* subdivided, part-creased and impossible to unwind.

    Segments that merely *share an endpoint* touch, they do not cross:
    consecutive ones do so by construction, and a closed loop or a lasso brings
    two segments that are far apart in the list back to a common point the same
    way.  So the test is not index adjacency alone -- an intersection is ignored
    when it lands on an endpoint of *both* segments, whichever pair they are.
    """
    points = [np.asarray(hit.position, dtype=float) for hit in path.hits]
    segments = [np.stack((a, b)) for a, b in zip(points, points[1:]) if np.linalg.norm(b - a) > vertex_tol]
    # ponytail: O(n^2) sweep, ~n^2/2 segment tests.  A rim is a few hundred
    # segments at the very most -- the longest over a 180-cast `rosette(7)`
    # sweep is 6 -- so this is at worst tens of thousands of jitted intersection
    # tests against a cast that is itself O(n).  `get_potential_intersections`
    # in `pleat.overlap` is the sweep line to reach for if rims ever get long.
    #
    # The one input that makes n large is a path that ran to `max_steps`, where
    # this becomes ~5e7 tests and presents as a hang rather than a failure.
    # Rejecting a non-terminal `ends[1]` here would cap it, but `max_steps` is a
    # documented truncation knob that several callers and tests use on purpose
    # to keep a half-traced path in play, and `open_sink` -- the caller for which
    # an incomplete rim is meaningless -- already rejects it right after the
    # cast.  The loop tests carry a `timeout` marker so the regression fails
    # instead of hanging.
    for i, s1 in enumerate(segments):
        for s2 in segments[i + 2 :]:
            for point in line_segment_intersections(s1, s2, vertex_tol):
                if _touches(point, s1, vertex_tol) and _touches(point, s2, vertex_tol):
                    continue
                raise DegenerateRayError(
                    f"ray crosses itself at {tuple(map(float, point))}: the segment "
                    f"{tuple(map(float, s1[0]))} -> {tuple(map(float, s1[1]))} meets "
                    f"{tuple(map(float, s2[0]))} -> {tuple(map(float, s2[1]))}"
                )


def add_ray_creases(
    G: "EuclideanPositionHEG",
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    both_ways: bool = True,
    vertex_tol: float | None = None,
    max_steps: int = 10_000,
    angle_tol: float = 1e-9,
) -> tuple[list[HalfEdge], RayPath]:
    """Cast a ray and add it to *G* as new vertices and creases.

    The ray is cast to completion on the original pattern, then materialized:
    its own creases must not deflect it, and a materialized segment would break
    :func:`fan_at_vertex`'s "inside a face" precondition, since the ray would
    lie exactly on the boundary between two sub-faces rather than inside one.

    The price is that a hit's ``(halfedge, t)`` goes stale as soon as an earlier
    hit splits the same edge, so materialization runs in two phases: first every
    vertex, edge by edge and in order along each edge, then one crease between
    each consecutive pair of vertices.

    Args:
        G: The crease pattern; modified in place.
        halfedge: The edge the ray starts on.
        t: Parameter along *halfedge*.
        direction: Initial direction of travel.
        side: Which side of a vertex the ray passes when it hits one head-on.
        both_ways: Cast backwards too if the forward ray does not close.
        vertex_tol: Distance below which a crossing snaps to an existing vertex.
        max_steps: Safety cap on the number of crossings per direction.
        angle_tol: Radians; the fan walk's angular tolerance, see
            :func:`fan_at_vertex`.

    Returns:
        ``(rim, path)`` -- the new half-edges in traversal order, each tagged
        with :data:`RAY_CREASE`, and the :class:`RayPath` that was cast.

    Raises:
        DegenerateRayError: from the cast itself, or if the ray crosses its own
            earlier path -- the earlier chord has already split the face, so the
            later segment has an endpoint on each side of it and cannot be laid.
            An origami sink rim is a simple curve, so a self-crossing ray is not
            a sink and rejecting it loses nothing real; splitting at
            self-intersections is deliberately deferred.  The crossing is
            detected on the cast trajectory before materialization begins, so
            *G* is left untouched when this is raised.
    """
    if vertex_tol is None:
        vertex_tol = default_vertex_tol(G)

    path = cast_ray(
        G,
        halfedge,
        t,
        direction,
        side=side,
        both_ways=both_ways,
        vertex_tol=vertex_tol,
        max_steps=max_steps,
        angle_tol=angle_tol,
    )
    # Before anything is materialized: a self-crossing trajectory cannot be laid,
    # and finding that out mid-phase-2 would leave `G` half-modified.
    _reject_self_crossing(path, vertex_tol)

    # ---- phase 1: vertices -------------------------------------------------
    vertices: dict[int, Vertex] = {}  # index into path.hits -> vertex
    by_edge: dict[HalfEdge, list[tuple[float, int]]] = {}
    for i, hit in enumerate(path.hits):
        if hit.vertex is not None:
            vertices[i] = hit.vertex  # the ray landed on a vertex that exists already
            continue
        # `hit.halfedges` is non-empty here: a fan that crosses nothing is a
        # graze, and a graze is a vertex hit, caught by the branch above.
        h = _canonical(hit.halfedges[0])
        # Order along the edge by distance from the canonical origin rather than
        # by `hit.t`: `t` is measured on whichever half the hit referenced, so
        # two hits recorded on opposite halves would otherwise be sorted against
        # each other backwards -- and the walk below depends on that order.
        # Nothing observes this today, because every mixed-halves ray also
        # crosses itself and `_reject_self_crossing` turns it away first.  It is
        # correct by construction and free, and deferring "split the rim at its
        # self-intersections instead of rejecting it" is what keeps it dormant:
        # doing that work makes mixed-halves rays materializable and this key
        # load-bearing again.
        by_edge.setdefault(h, []).append((float(np.linalg.norm(hit.position - h.orig["pos"])), i))

    for canonical, entries in by_edge.items():
        cur = canonical
        for _, i in sorted(entries):
            position = path.hits[i].position
            # Only the `orig` end needs a snap.  `_walk` snaps on the same
            # `vertex_tol` -- the start hit included, which is the one hit whose
            # parameter is the caller's and which used not to be snapped -- so
            # anything that close to an endpoint of the *original* edge arrives
            # here already reported as a vertex hit.  The only coincidence left
            # is with a vertex this loop inserted itself, and since the loop
            # walks forward along the edge that vertex is always `cur.orig`.
            if np.linalg.norm(position - cur.orig["pos"]) <= vertex_tol:
                vertices[i] = cur.orig
                continue
            h2, v = G.subdivide_edge(cur)
            v["pos"] = position  # exact; no parameter arithmetic
            vertices[i] = v
            cur = h2  # the remaining hits on this edge lie on the tail

    # ---- phase 2: creases --------------------------------------------------
    rim: list[HalfEdge] = []
    chain = [vertices[i] for i in sorted(vertices)]
    for a, b in zip(chain, chain[1:]):
        if a is b:  # consecutive hits that snapped to the same vertex
            continue
        # A self-overlapping trajectory runs the same chord twice and so retraces
        # a segment it has already creased.  Laying a second edge over the first
        # would leave a zero-area face behind, so reuse the edge that is there.
        # (Before closure detection worked this was reached mainly by periodic
        # rays being reported as `max_steps` and repeating their whole period;
        # it is rarer now, and needs a pattern already subdivided by earlier
        # rays, but it still happens -- removing this corrupts the graph on
        # roughly one in ten multi-ray patterns.)
        existing = next((h for h in list(a.outgoing_iter()) if h.dest is b), None)
        if existing is not None:
            existing[RAY_CREASE] = existing.rev[RAY_CREASE] = True
            rim.append(existing)
            continue
        face = _face_containing_segment(a, b)
        if face is None:
            # No common face means the ray crossed its own earlier path inside
            # this face: that chord already split the face, so `a` and `b` now
            # sit on opposite sides of it.  Laying the segment would need a
            # vertex at the self-intersection and nothing creates one.  Skipping
            # it silently leaves a gap in the rim, and callers flood-fill
            # bounded by the rim, so a gap leaks across the whole sheet.
            # `_reject_self_crossing` should have caught this on the trajectory
            # before any of this ran; reaching it means it missed one.
            raise DegenerateRayError(
                f"ray crosses itself: no face contains the segment "
                f"{tuple(map(float, a['pos']))} -> {tuple(map(float, b['pos']))}"
            )
        h12, _ = G.subdivide_face(face, a, b, **{RAY_CREASE: True})
        rim.append(h12)

    # The general check, against the specific one above: it also catches the two
    # half-rays failing to meet at the start point, which a graph that is not a
    # proper crease pattern can produce.  The rim is documented as "half-edges in
    # traversal order", so a discontinuous one breaks this function's own
    # contract and must never be returned.  It may legitimately be a path rather
    # than a cycle (a border-to-border ray), so `rim[-1].dest is rim[0].orig` is
    # deliberately *not* required.
    for h1, h2 in zip(rim, rim[1:]):
        if h1.dest is not h2.orig:
            raise DegenerateRayError(
                f"ray traces a discontinuous rim: "
                f"{tuple(map(float, h1.dest['pos']))} != {tuple(map(float, h2.orig['pos']))}"
            )

    G.recompute_lengths_and_angles()
    return rim, path
