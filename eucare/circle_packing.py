"""Discrete circle packing for half-edge graphs.

Implements Collins-Stephenson radii iteration on a triangulated half-edge
graph, producing a circle packing whose combinatorics match the input. Two
public entry points:

- :func:`pack_euclidean`: euclidean packing with prescribed boundary radii.
- :func:`pack_hyperbolic`: maximal circle packing in the Poincaré disk.

Both functions require triangulated, simply-connected (disk topology) input
and raise :class:`ValueError` otherwise. Numerical non-convergence raises
:class:`ConvergenceError`.

The output is a fresh :class:`EuclideanPositionHEG` (copy of the input by
default; pass ``copy_graph=False`` to mutate in place) with:

- ``v['pos']``: euclidean center of the circle (complex for hyperbolic
  output, 2D real array for euclidean output).
- ``v['radius']``: euclidean radius of the circle.

The "natural euclidean" mode (maximal packing transferred to euclidean
geometry) is obtained by composition::

    G_natural = pack_hyperbolic(G).convert_to_euclidean()
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Mapping

import numpy as np

from scipy.optimize import brentq

from eucare.geometries import EuclideanGeometry, PoincareDiskModel
from eucare.half import EuclideanPositionHEG, HalfEdge, Vertex


class ConvergenceError(RuntimeError):
    """Raised when the radii iteration fails to reach the requested tolerance."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_triangulated_disk(G) -> None:
    """Raise ValueError unless G is a triangulated, simply-connected disk."""
    for f in G.faces:
        if f.order() != 3:
            raise ValueError(
                "pack_*: input graph must be triangulated. "
                f"Found face of order {f.order()}. "
                "Triangulate first, e.g. via "
                "kis_graph()(G, faces=[f for f in G.faces if f.order() != 3])."
            )
    try:
        cycle = G.boundary_cycle()
    except LookupError as e:
        raise ValueError("pack_*: input graph has no boundary (closed surface).") from e
    if len(cycle) != len(G.border_edges()):
        raise ValueError(
            "pack_*: input graph has multiple boundary components; " "only simply-connected disks are supported."
        )


# ---------------------------------------------------------------------------
# Boundary radii resolution
# ---------------------------------------------------------------------------


def _resolve_boundary_radii(
    spec: float | Mapping[Vertex, float] | Callable[[Vertex], float],
    boundary_vertices: list[Vertex],
) -> dict[Vertex, float]:
    """Materialize a boundary_radii spec into a dict[Vertex, float]."""
    result: dict[Vertex, float] = {}
    if isinstance(spec, Mapping):
        for v in boundary_vertices:
            if v not in spec:
                raise ValueError(f"boundary_radii dict is missing entry for boundary vertex {v}.")
            r = float(spec[v])
            if r <= 0:
                raise ValueError(f"boundary_radii for vertex {v} must be positive; got {r}.")
            result[v] = r
        return result
    if callable(spec):
        for v in boundary_vertices:
            r = float(spec(v))
            if r <= 0:
                raise ValueError(f"boundary_radii({v}) must be positive; got {r}.")
            result[v] = r
        return result
    r = float(spec)
    if r <= 0:
        raise ValueError(f"boundary_radii must be positive; got {r}.")
    for v in boundary_vertices:
        result[v] = r
    return result


# ---------------------------------------------------------------------------
# Flower / triangle enumeration
# ---------------------------------------------------------------------------


def _incident_triangles(v: Vertex) -> list[tuple[Vertex, Vertex]]:
    """Return CCW-ordered list of (u, w) pairs forming triangle faces (v, u, w) incident to v.

    Caller has already validated all faces are triangles.
    """
    pairs: list[tuple[Vertex, Vertex]] = []
    for h in v.outgoing_iter():
        if h.face is None:
            continue
        u = h.dest
        w = h.nex.dest
        pairs.append((u, w))
    return pairs


# ---------------------------------------------------------------------------
# Euclidean angle formula and CS update
# ---------------------------------------------------------------------------


def _euclidean_angle_sum(r_v: float, neighbor_pairs: list[tuple[float, float]]) -> float:
    """Sum of angles at v over incident triangles, in the euclidean metric.

    Each pair (r_u, r_w) represents an incident triangle (v, u, w) with
    circles of radii (r_v, r_u, r_w) pairwise tangent.

    Angle at v: alpha = 2*arcsin(sqrt(r_u * r_w / ((r_v + r_u) * (r_v + r_w)))).
    """
    total = 0.0
    for r_u, r_w in neighbor_pairs:
        ratio = (r_u * r_w) / ((r_v + r_u) * (r_v + r_w))
        # numerical guard
        ratio = max(0.0, min(1.0, ratio))
        total += 2.0 * float(np.arcsin(np.sqrt(ratio)))
    return total


def _bowers_stephenson_update(
    r_v: float,
    neighbor_pairs: list[tuple[float, float]],
    target: float = 2 * np.pi,
) -> float:
    """One Bowers-Stephenson update step for r_v with the uniform-neighbor heuristic."""
    n = len(neighbor_pairs)
    theta = _euclidean_angle_sum(r_v, neighbor_pairs)
    # uniform-neighbor radius reproducing current angle sum
    s = float(np.sin(theta / (2 * n)))
    if s >= 1.0:
        # Degenerate; fall back to a small step toward smaller r_v.
        return r_v * 0.5
    r_hat = r_v * s / (1.0 - s)
    # radius that would give target angle sum with uniform neighbors
    s_target = float(np.sin(target / (2 * n)))
    return r_hat * (1.0 - s_target) / s_target


# ---------------------------------------------------------------------------
# Layout (BFS over faces)
# ---------------------------------------------------------------------------


def _layout_euclidean(
    G,
    radii: dict[Vertex, float],
    alpha: Vertex,
    beta: Vertex,
) -> dict[Vertex, np.ndarray]:
    """BFS layout: place alpha at origin, beta on +x axis, propagate face-by-face.

    Returns a dict {Vertex: np.array([x, y])}.
    """
    pos: dict[Vertex, np.ndarray] = {}
    pos[alpha] = np.array([0.0, 0.0])
    d_ab = radii[alpha] + radii[beta]
    pos[beta] = np.array([d_ab, 0.0])

    # Find the half-edge alpha -> beta to seed face traversal.
    seed_h: HalfEdge | None = None
    for h in alpha.outgoing_iter():
        if h.dest is beta:
            seed_h = h
            break
    if seed_h is None:
        raise ValueError(f"beta ({beta}) is not a neighbor of alpha ({alpha}).")

    # BFS over half-edges. Invariant: when we pop h, orig and dest are already placed,
    # and we need to place the third vertex of h.face (if not yet placed).
    visited_faces: set = set()
    queue: deque[HalfEdge] = deque()

    def enqueue_face_edges(h: HalfEdge) -> None:
        """If h.face is a triangle, enqueue it for processing."""
        if h.face is None or h.face in visited_faces:
            return
        queue.append(h)

    # Seed: triangles on both sides of the alpha-beta edge.
    enqueue_face_edges(seed_h)
    enqueue_face_edges(seed_h.rev)

    while queue:
        h = queue.popleft()
        f = h.face
        if f is None or f in visited_faces:
            continue
        visited_faces.add(f)
        a = h.orig
        b = h.dest
        c = h.nex.dest
        if c not in pos:
            pa = pos[a]
            pb = pos[b]
            ra = radii[a]
            rb = radii[b]
            rc = radii[c]
            # Triangle (a, b, c) is CCW (since face is CCW); c lies on the left of ab.
            # Side lengths: |ab|=ra+rb, |ac|=ra+rc, |bc|=rb+rc.
            d_ab = float(np.linalg.norm(pb - pa))
            d_ac = ra + rc
            d_bc = rb + rc
            # Angle at a between ab and ac (interior triangle angle):
            # by law of cosines: cos(angle_a) = (d_ab^2 + d_ac^2 - d_bc^2) / (2*d_ab*d_ac)
            cos_a = (d_ab * d_ab + d_ac * d_ac - d_bc * d_bc) / (2 * d_ab * d_ac)
            cos_a = max(-1.0, min(1.0, cos_a))
            sin_a = float(np.sqrt(max(0.0, 1.0 - cos_a * cos_a)))
            # Unit vector from a to b.
            u_ab = (pb - pa) / d_ab
            # Rotate by +angle_a (CCW) to get unit vector from a to c.
            # Rotation matrix [[c, -s], [s, c]] applied to u_ab.
            u_ac = np.array(
                [
                    cos_a * u_ab[0] - sin_a * u_ab[1],
                    sin_a * u_ab[0] + cos_a * u_ab[1],
                ]
            )
            pos[c] = pa + d_ac * u_ac
        # Enqueue adjacent faces via the other two edges of this triangle.
        for other_h in (h.nex, h.nex.nex):
            enqueue_face_edges(other_h.rev)

    if len(pos) != len(list(G.vertices)):
        missing = [v for v in G.vertices if v not in pos]
        raise RuntimeError(
            f"Layout did not reach all vertices ({len(missing)} unplaced). " "Graph may be disconnected."
        )
    return pos


# ---------------------------------------------------------------------------
# Alpha / beta selection
# ---------------------------------------------------------------------------


def _choose_alpha(G) -> Vertex:
    """Pick an interior vertex with maximal graph-distance to the boundary."""
    boundary = set(G.border_vertices())
    if not boundary:
        # No boundary — pick any vertex (won't happen due to validation, but defensive).
        return next(iter(G.vertices))
    # BFS from all boundary vertices simultaneously
    dist: dict[Vertex, int] = {v: 0 for v in boundary}
    queue: deque[Vertex] = deque(boundary)
    while queue:
        v = queue.popleft()
        for h in v.outgoing_iter():
            w = h.dest
            if w not in dist:
                dist[w] = dist[v] + 1
                queue.append(w)
    # Pick the interior vertex with max distance
    interior = [v for v in G.vertices if v not in boundary]
    if not interior:
        # Tiny graph, only boundary verts; pick any
        return next(iter(G.vertices))
    return max(interior, key=lambda v: dist.get(v, 0))


def _choose_beta(alpha: Vertex) -> Vertex:
    """Pick alpha's first neighbor (in flower CCW order) that is incident to a face."""
    for h in alpha.outgoing_iter():
        if h.face is not None:
            return h.dest
    raise ValueError(f"alpha vertex {alpha} has no incident triangle face.")


# ---------------------------------------------------------------------------
# Public entry point: pack_euclidean
# ---------------------------------------------------------------------------


def pack_euclidean(
    G,
    boundary_radii: float | Mapping[Vertex, float] | Callable[[Vertex], float] = 1.0,
    *,
    alpha: Vertex | None = None,
    beta: Vertex | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    copy_graph: bool = True,
) -> EuclideanPositionHEG:
    """Compute a euclidean circle packing with prescribed boundary radii.

    Args:
        G: Triangulated, simply-connected disk EuclideanPositionHEG.
        boundary_radii: Per-vertex boundary radii. Accepts a positive scalar
            (uniform), a Mapping[Vertex, float], or a callable Vertex -> float.
            Defaults to 1.0 (uniform boundary).
        alpha: Optional anchor vertex (placed at origin). Defaults to the
            interior vertex with maximum graph-distance to the boundary.
        beta: Optional second anchor (placed on +x axis through alpha).
            Defaults to alpha's first flower neighbor.
        tol: Maximum permitted angle defect ``|angle_sum - 2*pi|`` over
            interior vertices.
        max_iter: Maximum Collins-Stephenson iterations.
        copy_graph: If True (default), the input graph is copied; otherwise
            the input graph is mutated in place.

    Returns:
        EuclideanPositionHEG with EuclideanGeometry, ``v['pos']`` as 2D real
        array centers, and ``v['radius']`` as positive floats.

    Raises:
        ValueError: input not triangulated, not a simply-connected disk, or
            invalid boundary_radii.
        ConvergenceError: iteration did not reach ``tol`` within ``max_iter``.
    """
    _validate_triangulated_disk(G)

    # Working graph
    P = G.copy() if copy_graph else G

    # Boundary set + radii
    boundary_verts = P.border_vertices()
    radii_spec = _resolve_boundary_radii(boundary_radii, boundary_verts)
    boundary_set = set(boundary_verts)

    # Initial radii: boundary as spec, interior as mean of boundary
    boundary_mean = float(np.mean(list(radii_spec.values()))) if radii_spec else 1.0
    radii: dict[Vertex, float] = {}
    for v in P.vertices:
        if v in boundary_set:
            radii[v] = radii_spec[v]
        else:
            radii[v] = boundary_mean

    # Pre-cache incident-triangle neighbor lists for interior vertices
    interior_verts = [v for v in P.vertices if v not in boundary_set]
    flower_neighbors: dict[Vertex, list[tuple[Vertex, Vertex]]] = {v: _incident_triangles(v) for v in interior_verts}

    # Collins-Stephenson iteration
    converged = False
    final_defect = float("inf")
    for _iteration in range(max_iter):
        max_defect = 0.0
        for v in interior_verts:
            pairs = flower_neighbors[v]
            neighbor_radii = [(radii[u], radii[w]) for (u, w) in pairs]
            theta = _euclidean_angle_sum(radii[v], neighbor_radii)
            defect = abs(theta - 2 * np.pi)
            if defect > max_defect:
                max_defect = defect
            radii[v] = _bowers_stephenson_update(radii[v], neighbor_radii)
        if max_defect < tol:
            converged = True
            final_defect = max_defect
            break
        final_defect = max_defect
    if not converged:
        raise ConvergenceError(
            f"pack_euclidean: did not converge in {max_iter} iterations; "
            f"final max angle defect = {final_defect:.3e}"
        )

    # Choose anchors
    if alpha is None:
        alpha = _choose_alpha(P)
    if beta is None:
        beta = _choose_beta(alpha)

    # Layout
    pos = _layout_euclidean(P, radii, alpha, beta)

    # Write back onto graph
    for v in P.vertices:
        v["pos"] = pos[v]
        v["radius"] = radii[v]
    P.geometry = EuclideanGeometry
    P.recompute_lengths_and_angles()
    return P


# ---------------------------------------------------------------------------
# Hyperbolic angle formula (x-radius parametrization)
# ---------------------------------------------------------------------------


def _hyperbolic_angle_sum(x_v: float, neighbor_pairs: list[tuple[float, float]]) -> float:
    """Sum of hyperbolic angles at v over incident triangles, in x-radius parametrization.

    For each incident triangle (v, u, w) with x-radii (x_v, x_u, x_w) and circles
    pairwise tangent in the hyperbolic metric, the angle at v is::

        sin^2(alpha/2) = (1 - x_v) * x_u * x_w
                        / ((1 - (1-x_v)(1-x_w)) * (1 - (1-x_v)(1-x_u)))

    This expression is the natural limit of the Euclidean formula and
    extends consistently to horocycles (x = 1).
    """
    a_v = 1.0 - x_v
    total = 0.0
    for x_u, x_w in neighbor_pairs:
        a_u = 1.0 - x_u
        a_w = 1.0 - x_w
        denom = (1.0 - a_v * a_w) * (1.0 - a_v * a_u)
        num = a_v * x_u * x_w
        if denom <= 0.0:
            ratio = 1.0
        else:
            ratio = num / denom
        ratio = max(0.0, min(1.0, ratio))
        total += 2.0 * float(np.arcsin(np.sqrt(ratio)))
    return total


def _solve_x_for_target_angle(neighbor_pairs: list[tuple[float, float]], target: float) -> float:
    """Solve for x_v in (0, 1) such that the hyperbolic angle sum at v equals target.

    Angle sum is monotonically decreasing in x_v, ranging from N*pi at x_v=0+
    to 0 at x_v=1 (horocycle). For target=2*pi we need at least 3 incident
    triangles (otherwise no solution exists).
    """

    def f(x: float) -> float:
        return _hyperbolic_angle_sum(x, neighbor_pairs) - target

    lo, hi = 1e-14, 1.0 - 1e-14
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo * f_hi > 0:
        # No bracket — degenerate case. Return whichever endpoint is closer to target.
        return lo if abs(f_lo) < abs(f_hi) else hi
    return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-14))


# ---------------------------------------------------------------------------
# Hyperbolic layout (Poincare disk via Mobius constructions)
# ---------------------------------------------------------------------------


def _hyperbolic_angle_at_anchor(x_anchor: float, x_arm: float, x_target: float) -> float:
    """Angle at the anchor in a triangle (anchor, arm, target) of tangent hyperbolic disks."""
    pair_at_anchor = [(x_arm, x_target)]
    # _hyperbolic_angle_sum returns sum over a list; with one pair, it's just that angle.
    return _hyperbolic_angle_sum(x_anchor, pair_at_anchor)


def _layout_hyperbolic(
    G,
    x_radii: dict[Vertex, float],
    alpha: Vertex,
    beta: Vertex,
) -> dict[Vertex, complex]:
    """Place hyperbolic centers in the Poincare disk via BFS over triangle faces.

    Returns a dict {Vertex: complex} of hyperbolic centers inside the unit disk.
    Boundary x-radii must be < 1 (horocycles not supported in v1).
    """
    # Convert x-radii to hyperbolic radii.
    h_radii: dict[Vertex, float] = {}
    for v in G.vertices:
        x = x_radii[v]
        if x >= 1.0 - 1e-15:
            raise NotImplementedError(
                "pack_hyperbolic: horocycle boundary (x_radius >= 1) is not yet "
                "supported. Use boundary_x_radii close to but less than 1.0."
            )
        h_radii[v] = -0.5 * float(np.log(1.0 - x))

    pos: dict[Vertex, complex] = {}
    pos[alpha] = complex(0.0, 0.0)
    d_alpha_beta = h_radii[alpha] + h_radii[beta]
    pos[beta] = complex(np.tanh(d_alpha_beta / 2.0), 0.0)

    # Find seed half-edge alpha -> beta
    seed_h: HalfEdge | None = None
    for h in alpha.outgoing_iter():
        if h.dest is beta:
            seed_h = h
            break
    if seed_h is None:
        raise ValueError(f"beta ({beta}) is not a neighbor of alpha ({alpha}).")

    visited_faces: set = set()
    queue: deque[HalfEdge] = deque()

    def enqueue(h: HalfEdge) -> None:
        if h.face is None or h.face in visited_faces:
            return
        queue.append(h)

    enqueue(seed_h)
    enqueue(seed_h.rev)

    while queue:
        h = queue.popleft()
        f = h.face
        if f is None or f in visited_faces:
            continue
        visited_faces.add(f)
        a = h.orig
        b = h.dest
        c = h.nex.dest
        if c not in pos:
            angle_at_a = _hyperbolic_angle_at_anchor(x_radii[a], x_radii[b], x_radii[c])
            length_ac = h_radii[a] + h_radii[c]
            # construct_next_poly_point(p_arm, p_anchor, angle, length) → place point
            # such that angle(p_arm, p_anchor, c) == angle and dist(p_anchor, c) == length.
            pos[c] = complex(PoincareDiskModel.construct_next_poly_point(pos[b], pos[a], angle_at_a, length_ac))
        for h_other in (h.nex, h.nex.nex):
            enqueue(h_other.rev)

    if len(pos) != len(list(G.vertices)):
        missing = [v for v in G.vertices if v not in pos]
        raise RuntimeError(f"Hyperbolic layout did not reach all vertices ({len(missing)} unplaced).")
    return pos


# ---------------------------------------------------------------------------
# Public entry point: pack_hyperbolic
# ---------------------------------------------------------------------------


def pack_hyperbolic(
    G,
    boundary_x_radii: float | Mapping[Vertex, float] | Callable[[Vertex], float] = 0.5,
    *,
    alpha: Vertex | None = None,
    beta: Vertex | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    copy_graph: bool = True,
) -> EuclideanPositionHEG:
    """Compute a hyperbolic circle packing in the Poincare disk.

    Boundary vertices are assigned the prescribed x-radii (where x = 1 - exp(-2h),
    h being the hyperbolic radius); interior radii are iterated via
    Collins-Stephenson to satisfy hyperbolic angle sum = 2*pi.

    Note: v1 supports finite boundary x-radii only (x < 1). True maximal
    packings (with boundary horocycles, x = 1) require additional Mobius
    handling at the unit-circle boundary and are deferred to a follow-up.
    Boundary x-radii close to 1 approximate maximal behavior.

    Args:
        G: Triangulated, simply-connected disk EuclideanPositionHEG.
        boundary_x_radii: Per-vertex boundary x-radii in (0, 1). Accepts a
            scalar (uniform), Mapping, or callable. Defaults to 0.5.
        alpha: Optional anchor vertex (placed at Poincare disk origin).
            Defaults to interior vertex with max graph-distance to boundary.
        beta: Optional second anchor (placed on +x axis through alpha).
        tol: Max angle defect over interior vertices.
        max_iter: Max Collins-Stephenson iterations.
        copy_graph: If True (default), return a new EHEG; otherwise mutate.

    Returns:
        EuclideanPositionHEG with PoincareDiskModel geometry, ``v['pos']``
        as complex hyperbolic centers in the open unit disk, and
        ``v['radius']`` as x-radii in (0, 1).
    """
    _validate_triangulated_disk(G)

    P = G.copy() if copy_graph else G

    boundary_verts = P.border_vertices()
    radii_spec = _resolve_boundary_radii(boundary_x_radii, boundary_verts)
    for v, x in radii_spec.items():
        if not (0.0 < x < 1.0):
            raise ValueError(
                f"pack_hyperbolic v1: boundary x-radius must be in (0, 1); "
                f"got {x} for vertex {v}. Horocycles (x=1) not yet supported."
            )
    boundary_set = set(boundary_verts)

    # Init: boundary at spec, interior uniformly at 0.5 (mid-range)
    x_radii: dict[Vertex, float] = {}
    for v in P.vertices:
        if v in boundary_set:
            x_radii[v] = radii_spec[v]
        else:
            x_radii[v] = 0.5

    interior_verts = [v for v in P.vertices if v not in boundary_set]
    flower_neighbors: dict[Vertex, list[tuple[Vertex, Vertex]]] = {v: _incident_triangles(v) for v in interior_verts}

    converged = False
    final_defect = float("inf")
    for _iteration in range(max_iter):
        max_defect = 0.0
        for v in interior_verts:
            pairs = flower_neighbors[v]
            neighbor_xs = [(x_radii[u], x_radii[w]) for (u, w) in pairs]
            theta = _hyperbolic_angle_sum(x_radii[v], neighbor_xs)
            defect = abs(theta - 2 * np.pi)
            if defect > max_defect:
                max_defect = defect
            x_radii[v] = _solve_x_for_target_angle(neighbor_xs, 2 * np.pi)
        if max_defect < tol:
            converged = True
            final_defect = max_defect
            break
        final_defect = max_defect
    if not converged:
        raise ConvergenceError(
            f"pack_hyperbolic: did not converge in {max_iter} iterations; "
            f"final max angle defect = {final_defect:.3e}"
        )

    if alpha is None:
        alpha = _choose_alpha(P)
    if beta is None:
        beta = _choose_beta(alpha)

    pos = _layout_hyperbolic(P, x_radii, alpha, beta)

    for v in P.vertices:
        v["pos"] = pos[v]
        v["radius"] = x_radii[v]
    P.geometry = PoincareDiskModel
    P.recompute_lengths_and_angles()
    return P
