"""Sink folds on crease patterns.

An open sink pushes a folded point through the model.  In the crease pattern
this is a rim traced by a ray (see :mod:`pleat.ray_casting`), every crease
strictly inside the rim inverted, and the rim itself assigned *uniformly*
mountain or valley -- which of the two is inferred by running the vertex-wise
test of :mod:`pleat.flat_foldable` at each rim node.

A closed sink -- the same geometry with a different rim assignment -- is
deliberately out of scope.
"""

from __future__ import annotations

import logging

import numpy as np

from .flat_foldable import local_assignment_valid
from .half import Face, HalfEdge, Vertex
from .overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from .ray_casting import add_ray_creases

logger = logging.getLogger(__name__)

#: A folded crease position closer than this to its neighbour makes the vertex
#: genuinely ambiguous rather than wrong; it is logged, never failed on.
DEGENERATE_MARGIN = 1e-6


class InvalidSinkError(ValueError):
    """The requested sink fold does not yield a valid crease pattern."""


def _flood(seeds, blocked: set[HalfEdge]):
    """Yield the faces reachable from *seeds*, one at a time, never crossing *blocked*.

    The paper border stops the fill as well: a half-edge whose reverse has no
    face leads off the sheet.  Yielding one face at a time is what lets two
    fills be raced against each other in :func:`_interior_faces`.
    """
    seen: set[Face] = set()
    stack = [f for f in seeds if f is not None]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        yield f
        # `halfedge_iter` is a cyclic iterator; materialise before touching it
        for h in list(f.halfedge_iter()):
            if h not in blocked and h.rev.face is not None and h.rev.face not in seen:
                stack.append(h.rev.face)


def _rim_is_ccw(rim: list[HalfEdge]) -> bool:
    """Return True if the closed *rim* runs counter-clockwise.

    The shoelace sum over the rim's origins.  This is exact and needs no angle
    arithmetic; the rim is a simple polygon because ``add_ray_creases`` rejects
    a self-crossing ray before materialising it.
    """
    p = np.stack([h.orig["pos"] for h in rim])
    x, y = p[:, 0], p[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def _interior_faces(rim: list[HalfEdge], closed: bool) -> set[Face]:
    """Return the faces enclosed by *rim*, bounded by the rim and the paper border.

    For a closed rim the inside is determined exactly.  ``h.face`` is the face
    to the *left* of ``h``, so a counter-clockwise rim encloses ``h.face`` and a
    clockwise one encloses ``h.rev.face``.  Being exact matters: a sink that
    encloses most of the sheet has a bigger inside than outside, so anything
    that picks the smaller side would take the wrong one.

    An open rim runs border to border and has no canonical inside, so the two
    fills are raced against each other one face at a time and the side that
    finishes first -- the smaller one -- is taken as the interior.  That is a
    convention, not a derivation.  Racing costs no more than the smaller fill
    even when the other side is huge.
    """
    blocked = {h for h in rim} | {h.rev for h in rim}

    if closed:
        ccw = _rim_is_ccw(rim)
        seeds = [h.face if ccw else h.rev.face for h in rim]
        return set(_flood(seeds, blocked))

    left = _flood([h.face for h in rim], blocked)
    right = _flood([h.rev.face for h in rim], blocked)
    seen_left: set[Face] = set()
    seen_right: set[Face] = set()
    while True:
        f = next(left, None)
        if f is None:
            return seen_left
        seen_left.add(f)
        f = next(right, None)
        if f is None:
            return seen_right
        seen_right.add(f)


def _rim_nodes(rim: list[HalfEdge]) -> list[Vertex]:
    """Return the interior vertices along *rim*, in traversal order, without repeats.

    A closed rim visits each of its nodes twice and an open rim ends on the
    paper border, where no local flat-foldability condition applies -- so
    border vertices are dropped, and the rest are de-duplicated while keeping
    the order, so that diagnostics do not depend on set iteration order.
    """
    nodes: dict[Vertex, None] = {}
    for h in rim:
        for v in (h.orig, h.dest):
            if not v.on_border():
                nodes[v] = None
    return list(nodes)


def open_sink(
    G,
    halfedge: HalfEdge,
    t: float,
    direction,
    side: str = "left",
    strict: bool = True,
    **cast_kwargs,
) -> list[HalfEdge]:
    """Add an open sink fold to *G*, starting from a point on *halfedge*.

    Casts a ray to trace the rim, adds it to the graph, inverts every crease
    strictly inside it, and infers whether the rim is mountain or valley.  The
    rim of an open sink is uniform, so that last step is a two-candidate test
    rather than a search: the whole rim is set to ``MOUNTAIN`` and every
    interior rim node checked, then the whole rim flipped to ``VALLEY`` and
    checked again.

    The degree-4 case needs no special handling.  At a node the rim creates, the
    radial crease is split into ``c_in`` (inverted, being inside) and ``c_out``,
    so ``c_in == -c_out``; Maekawa then forces the two rim edges to agree and
    big-little-big picks out the radial half bounding the smallest sector.

    Args:
        G: The crease pattern; modified in place.
        halfedge: The edge the rim starts on.
        t: Parameter along *halfedge*; ``0`` and ``1`` mean its endpoints.
        direction: Initial direction of the rim.
        side: Which side of a vertex the ray passes when it hits one head-on.
        strict: If True, raise when a crease assignment the sink needs is
            missing, or when neither uniform rim assignment folds flat.  If
            False, log and carry on -- the geometry is identical either way, so
            this makes ``open_sink`` usable on a pattern whose creases have not
            been assigned yet.
        **cast_kwargs: Forwarded to :func:`pleat.ray_casting.add_ray_creases`
            (``both_ways``, ``vertex_tol``, ``max_steps``).

    Returns:
        The rim half-edges in traversal order.  A closed rim is a cycle; a ray
        that ran off the sheet in both directions gives a rim that is a path,
        which is a perfectly good sink.

    Raises:
        InvalidSinkError: if the ray did not terminate cleanly, or -- when
            *strict* -- a needed crease assignment is missing or no uniform rim
            assignment folds flat.  *G* has already been modified when this is
            raised: the rim is materialised, and the interior may already be
            inverted.
        pleat.ray_casting.DegenerateRayError: propagated from the cast, most
            often because the ray crossed its own path.  It is deliberately not
            wrapped: it is also a ``ValueError``, it carries the geometry of the
            failure in its message, and unlike ``InvalidSinkError`` it is raised
            before *G* is touched -- so a caller that has to tell "no sink here"
            from "the pattern is now half-sunk" can.
    """
    rim, path = add_ray_creases(G, halfedge, t, direction, side=side, **cast_kwargs)
    # `closed` and `border` are traced-to-completion; `start` is what the
    # backward end reports on a one-way cast and is fine too.  Anything else
    # (`max_steps`, `degenerate`) is a failure.  Testing for `"max_steps"`
    # specifically would let `"degenerate"` through as a success.
    bad = [reason for reason in path.ends if reason not in ("closed", "border", "start")]
    if bad:
        raise InvalidSinkError(f"the sink rim did not terminate cleanly: {bad}")

    interior = _interior_faces(rim, path.closed)
    rim_edges = {h for h in rim} | {h.rev for h in rim}
    # A crease lies *strictly* inside when both its faces do.  Requiring both
    # ends does three things at once: it drops the rim (whose far side is
    # outside), it drops the paper border (whose far side is no face at all,
    # and which is not a crease anyway), and it reaches each half-edge exactly
    # once -- from its own face -- so the two halves of an edge cannot end up
    # disagreeing about their assignment.
    inner = [h for f in interior for h in list(f.halfedge_iter()) if h not in rim_edges and h.rev.face in interior]

    nodes = _rim_nodes(rim)
    # Step 3 reads every crease at every rim node, not just the ones inside, so
    # both sets have to be assigned for the verdict to mean anything.  The rim's
    # own creases are excluded: assigning them is what step 3 is for.
    needed = set(inner) | {h for v in nodes for h in list(v.outgoing_iter()) if h not in rim_edges}
    missing = [h for h in needed if CREASE_ASSIGNMENT not in h.attributes]
    if missing:
        message = f"{len(missing)} creases the sink needs carry no crease assignment"
        if strict:
            raise InvalidSinkError(f"{message}; pass strict=False to sink anyway")
        logger.warning("%s; leaving them alone and skipping the nodes they touch", message)

    for h in inner:
        if CREASE_ASSIGNMENT in h.attributes:
            h[CREASE_ASSIGNMENT] = -h[CREASE_ASSIGNMENT]

    failures: list[tuple[Vertex, float]] = []
    for candidate in (MOUNTAIN, VALLEY):
        for h in rim:
            h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = candidate
        failures = []
        for v in nodes:
            if not all(CREASE_ASSIGNMENT in h.attributes for h in list(v.outgoing_iter())):
                continue  # only reachable with strict=False
            valid, margin = local_assignment_valid(v)
            if not valid:
                failures.append((v, margin))
            elif margin <= DEGENERATE_MARGIN:
                # a symmetric vertex legitimately lands here: the verdict stands,
                # it is just not robust to the tie-breaking
                logger.info("sink rim node at %s is ambiguous (margin %.3e)", v["pos"], margin)
        if not failures:
            return rim

    message = "no uniform rim assignment folds flat; as VALLEY these nodes fail: " + ", ".join(
        f"{tuple(map(float, v['pos']))} (margin {margin:.3e})" for v, margin in failures
    )
    if strict:
        raise InvalidSinkError(message)
    logger.warning("%s; leaving the rim MOUNTAIN", message)
    for h in rim:
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = MOUNTAIN
    return rim
