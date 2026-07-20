"""Sink folds on crease patterns.

A sink pushes a folded point through the model.  In the crease pattern this is
a rim traced by a ray (see :mod:`pleat.ray_casting`) plus a rule for
reassigning the creases it encloses.  Both variants live here and share their
whole geometric half:

* :func:`open_sink` inverts *every* crease strictly inside the rim, and the rim
  comes out uniformly mountain or valley.
* :func:`closed_sink` inverts only the **two** creases that end up outermost
  when the enclosed vertex is folded, and the rim switches at every node except
  those two.  It needs exactly one vertex inside a fully closed rim.

Which of the two is the same loop closed the other way round: the rim switches
mountain/valley at either all of its nodes or all but two.  In both cases the
absolute value of the rim is inferred by running the vertex-wise test of
:mod:`pleat.flat_foldable` over the candidates.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np

from .flat_foldable import folded_crease_angles, local_assignment_valid
from .half import Face, HalfEdge, Vertex
from .overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from .ray_casting import add_ray_creases

if TYPE_CHECKING:
    from .half import EuclideanPositionHEG

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


def _trace_rim(
    G: "EuclideanPositionHEG",
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str,
    strict: bool,
    require_closed: bool,
    cast_kwargs: dict,
) -> tuple[list[HalfEdge], set[Face], list[HalfEdge], list[Vertex]]:
    """Cast, materialise and vet the rim shared by both sink variants.

    Returns ``(rim, interior, inner, nodes)``: the rim half-edges in traversal
    order, the faces it encloses, the half-edges strictly inside it, and its
    interior nodes.

    *require_closed* tightens the end check for a closed sink, whose rim has to
    be a cycle; an open sink also accepts a rim running border to border.
    """
    # A one-way cast leaves the backward end untraced (`ends[0] == "start"`), so
    # the rim is a path with a loose end in the middle of the sheet: it does not
    # separate the paper, the interior fill leaks around the dangling end, and
    # the sink inverts the whole model.  A global M/V flip satisfies every local
    # condition, so nothing downstream catches it -- rejecting the argument
    # outright is the only place it can be caught.
    if "both_ways" in cast_kwargs:
        raise TypeError("a sink always casts both ways: a one-way rim has a loose end and does not separate the paper")
    rim, path = add_ray_creases(G, halfedge, t, direction, side=side, both_ways=True, **cast_kwargs)
    # Only `closed` and `border` are traced to completion.  Anything else
    # (`max_steps`, `stalled`, `start`) is a failure.  Testing for
    # `"max_steps"` specifically would let the others through as a success.
    allowed = ("closed",) if require_closed else ("closed", "border")
    bad = [reason for reason in path.ends if reason not in allowed]
    if bad:
        raise InvalidSinkError(f"the sink rim did not terminate cleanly: {bad}")
    # The two half-rays are one reversible line, so an end that closed means the
    # whole trajectory closed.  A lasso -- a cycle with a tail, `"closed"` in
    # `ends` without `closed` -- would slip past the check above and then be
    # treated as an *open* rim below, where the race picks a "smaller side" of a
    # curve that does not separate the paper.  It cannot happen; fail loudly if
    # it ever does rather than inverting an arbitrary region.
    assert path.closed or path.ends == ("border", "border"), f"cycle with a tail: {path.ends}"

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
    # The rim assignment is inferred by reading every crease at every rim node,
    # not just the ones inside, so both sets have to be assigned for the verdict
    # to mean anything.  The rim's own creases are excluded: assigning them is
    # what the inference is for.
    needed = set(inner) | {h for v in nodes for h in list(v.outgoing_iter()) if h not in rim_edges}
    missing = [h for h in needed if CREASE_ASSIGNMENT not in h.attributes]
    if missing:
        message = f"{len(missing)} creases the sink needs carry no crease assignment"
        if strict:
            raise InvalidSinkError(f"{message}; pass strict=False to sink anyway")
        logger.warning("%s; leaving them alone and skipping the nodes they touch", message)
    return rim, interior, inner, nodes


def _failing_nodes(vertices) -> list[tuple[Vertex, float]]:
    """Return the *vertices* whose crease assignment does not fold flat, with their margins.

    Vertices carrying an incomplete assignment are skipped -- only reachable
    with ``strict=False``, which is exactly the case that asked for it.
    """
    failures: list[tuple[Vertex, float]] = []
    for v in vertices:
        if not all(CREASE_ASSIGNMENT in h.attributes for h in list(v.outgoing_iter())):
            continue
        valid, margin = local_assignment_valid(v)
        if not valid:
            failures.append((v, margin))
        elif margin <= DEGENERATE_MARGIN:
            # a symmetric vertex legitimately lands here: the verdict stands,
            # it is just not robust to the tie-breaking
            logger.info("sink node at %s is ambiguous (margin %.3e)", v["pos"], margin)
    return failures


def _describe(failures: list[tuple[Vertex, float]]) -> str:
    return ", ".join(f"{tuple(map(float, v['pos']))} (margin {margin:.3e})" for v, margin in failures)


def open_sink(
    G: "EuclideanPositionHEG",
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    strict: bool = True,
    **cast_kwargs: object,
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
            (``vertex_tol``, ``angle_tol``, ``max_steps``).  ``both_ways`` is
            *not* accepted:
            a sink rim has to be traced in both directions or it does not
            separate the paper, so passing it raises ``TypeError``.

    Returns:
        The rim half-edges in traversal order.  A closed rim is a cycle; a ray
        that ran off the sheet in both directions gives a rim that is a path,
        which is a perfectly good sink.

    Raises:
        InvalidSinkError: if the ray did not terminate cleanly, or -- when
            *strict* -- a needed crease assignment is missing or no uniform rim
            assignment folds flat.  *G* has already been modified when this is
            raised: the rim is materialised, and the interior may already be
            inverted.  There is no undo, so a caller that needs to recover
            should sink into a copy of *G* and keep the original until the call
            has returned.
        TypeError: if ``both_ways`` is passed; see *cast_kwargs*.
        pleat.ray_casting.DegenerateRayError: propagated from the cast, most
            often because the ray crossed its own path.  It is deliberately not
            wrapped: it is also a ``ValueError``, it carries the geometry of the
            failure in its message, and unlike ``InvalidSinkError`` it is raised
            before *G* is touched -- so a caller that has to tell "no sink here"
            from "the pattern is now half-sunk" can.
    """
    rim, _interior, inner, nodes = _trace_rim(G, halfedge, t, direction, side, strict, False, cast_kwargs)

    for h in inner:
        if CREASE_ASSIGNMENT in h.attributes:
            h[CREASE_ASSIGNMENT] = -h[CREASE_ASSIGNMENT]

    failures: list[tuple[Vertex, float]] = []
    # A fully symmetric rim -- every node a symmetric degree-4 crossing -- admits
    # both candidates, and then the order below is the whole answer.  MOUNTAIN
    # first is the deliberate tie-break; a test pins it.
    for candidate in (MOUNTAIN, VALLEY):
        for h in rim:
            h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = candidate
        failures = _failing_nodes(nodes)
        if not failures:
            return rim

    message = "no uniform rim assignment folds flat; as VALLEY these nodes fail: " + _describe(failures)
    # Reset before deciding what to do about it, so the rim is left MOUNTAIN
    # either way rather than keeping whatever the last candidate happened to be.
    for h in rim:
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = MOUNTAIN
    if strict:
        raise InvalidSinkError(message)
    logger.warning("%s; leaving the rim MOUNTAIN", message)
    return rim


def _outer_creases(v: Vertex, tol: float = 1e-8) -> tuple[list[HalfEdge], list[HalfEdge]]:
    """Return the creases at *v* that land outermost when *v* is folded.

    Folding *v* maps its creases to the angles ``psi`` of
    :func:`pleat.flat_foldable.folded_crease_angles`; the folded cone spans
    ``[min psi, max psi]`` and the creases on those two boundary rays are the
    outer folds of the wedge.  ``psi`` is built from sector angles only, so this
    is geometry -- it never reads a crease assignment.

    ``folded_crease_angles`` indexes over ``v.incoming_iter()``, so ``psi[k]``
    belongs to the crease of the ``k``-th *incoming* half-edge; its ``rev`` is
    the outgoing half at *v*, which is the one a sink needs.  Getting that
    alignment off by one picks a different pair and is invisible downstream, so
    a test pins it against a vertex whose sectors all differ.

    Returns:
        ``(outermost, innermost)`` -- the creases at the maximum and at the
        minimum of ``psi``, as *outgoing* half-edges at *v*.  A symmetric vertex
        ties: a preliminary-base apex (90/90/90/90) puts two creases at each
        end, and the choice between them is not determined by geometry.  Both
        lists therefore report every tie rather than a coin flip.
    """
    psi = folded_crease_angles(v)
    creases = [h.rev for h in v.incoming_iter()]
    outermost = [c for c, p in zip(creases, psi) if p >= psi.max() - tol]
    innermost = [c for c, p in zip(creases, psi) if p <= psi.min() + tol]
    return outermost, innermost


def closed_sink(
    G: "EuclideanPositionHEG",
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    strict: bool = True,
    **cast_kwargs: object,
) -> list[HalfEdge]:
    """Add a closed sink fold to *G*, starting from a point on *halfedge*.

    Same rim as :func:`open_sink`, different crease assignment.  Where an open
    sink inverts every crease inside the rim and comes out with a uniform rim,
    a closed sink inverts only **two** -- the creases that end up outermost when
    the enclosed vertex is folded (:func:`_outer_creases`) -- and the rim
    switches mountain/valley at every node it passes *except* those two.

    That is what makes the loop close.  Every rim node is degree 4
    (``rim_in, c_in, rim_out, c_out``): where the radial crease is untouched its
    two halves agree and contribute ``+-2``, so the rim has to switch for
    Maekawa; where it is reversed they cancel, so the rim must *not* switch.
    The rim therefore switches at ``n - 2`` of its ``n`` nodes, and ``n`` is
    even, so it returns to the value it started with.  Inverting all ``n``
    closes the same loop the other way -- that is the open sink.

    Scope for this iteration: exactly one vertex strictly inside a rim that is a
    closed cycle.  The single-vertex restriction is what makes every crease
    inside the rim a radial half from that vertex, which is what lets the rule
    be stated per crease at all.

    Two things geometry does not fix, so both are searched and validated with
    :func:`pleat.flat_foldable.local_assignment_valid` at the sunk vertex *and*
    at every rim node:

    * the rim's absolute value -- Maekawa gives ``+-2`` either way, so only
      big-little-big decides between ``MOUNTAIN`` and ``VALLEY``;
    * ties at the extremes of ``psi`` -- a symmetric vertex offers several
      candidate pairs.  Every one is tried, and the ambiguity is logged rather
      than presented as a derivation.

    The check at the sunk vertex is not a formality: the new sum there is
    ``sum(m) - 2 * (m_p + m_q)``, which stays ``+-2`` only for some pairs.

    Args:
        G: The crease pattern; modified in place.
        halfedge: The edge the rim starts on.
        t: Parameter along *halfedge*; ``0`` and ``1`` mean its endpoints.
        direction: Initial direction of the rim.
        side: Which side of a vertex the ray passes when it hits one head-on.
        strict: If True, raise when a crease assignment the sink needs is
            missing, or when no candidate folds flat.  If False, log and carry
            on; the geometry is identical either way.
        **cast_kwargs: Forwarded to :func:`pleat.ray_casting.add_ray_creases`
            (``vertex_tol``, ``angle_tol``, ``max_steps``).  ``both_ways`` is
            *not* accepted; see :func:`open_sink`.

    Returns:
        The rim half-edges in traversal order, as a cycle.

    Raises:
        InvalidSinkError: if the rim is not a closed cycle, does not enclose
            exactly one vertex, or -- when *strict* -- a needed crease
            assignment is missing or no candidate folds flat.  *G* has already
            been modified when this is raised; see :func:`open_sink`.
        TypeError: if ``both_ways`` is passed.
        pleat.ray_casting.DegenerateRayError: propagated from the cast.
    """
    rim, interior, inner, nodes = _trace_rim(G, halfedge, t, direction, side, strict, True, cast_kwargs)

    inside = {w for f in interior for w in f.vertex_iter()} - {w for h in rim for w in (h.orig, h.dest)}
    if len(inside) != 1:
        raise InvalidSinkError(
            f"a closed sink needs exactly one vertex inside the rim, found {len(inside)}: "
            + ", ".join(str(tuple(map(float, w["pos"]))) for w in inside)
        )
    (v,) = inside

    outermost, innermost = _outer_creases(v)
    if len(outermost) > 1 or len(innermost) > 1:
        logger.info(
            "the folded cone at %s is symmetric (%d creases outermost, %d innermost), so which pair to reverse is "
            "ambiguous; trying them in turn",
            v["pos"],
            len(outermost),
            len(innermost),
        )
    # The rim crosses each radial crease exactly once, at the node it ends on.
    node_crease = {h.dest: h for h in v.outgoing_iter()}
    originals = {h: h[CREASE_ASSIGNMENT] for h in inner if CREASE_ASSIGNMENT in h.attributes}

    def apply(p: HalfEdge, q: HalfEdge, start: int) -> None:
        flipped = {p, q, p.rev, q.rev}
        # Only these two reverse.  Inverting every inner crease is the *open*
        # sink: it would make `c_in` and `c_out` cancel at every node while the
        # rim still switched there, leaving Maekawa at 0 instead of +-2.
        for h, assignment in originals.items():
            h[CREASE_ASSIGNMENT] = -assignment if h in flipped else assignment
        value = start
        for h in rim:
            h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = value
            if node_crease.get(h.dest) not in flipped:
                value = -value

    candidates = [(p, q) for p, q in itertools.product(outermost, innermost) if p is not q]
    best: list[tuple[Vertex, float]] | None = None
    for p, q in candidates:
        for start in (MOUNTAIN, VALLEY):
            apply(p, q, start)
            failures = _failing_nodes([v] + nodes)
            if not failures:
                return rim
            # Report the candidate that got furthest rather than whichever was
            # tried last: a vertex that fails for *every* candidate is the
            # actual obstruction, and the sunk vertex is often exactly that.
            if best is None or len(failures) < len(best):
                best = failures

    assert best is not None
    message = "no rim assignment folds flat for any of the {} candidate pairs; at best these fail: {}".format(
        len(candidates), _describe(best)
    )
    # Leave the graph in the state the first candidate produces, so a failed
    # strict=False call is reproducible rather than showing whichever candidate
    # happened to be tried last.
    apply(*candidates[0], MOUNTAIN)
    if strict:
        raise InvalidSinkError(message)
    logger.warning("%s; leaving the first candidate in place", message)
    return rim
