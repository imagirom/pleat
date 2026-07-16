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

import math
from collections import deque
from typing import Callable, Mapping

import numpy as np

from pleat.geometries import EuclideanGeometry, PoincareDiskModel
from pleat.half import EuclideanPositionHEG, HalfEdge, Vertex


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
# Boundary angle resolution & inference
# ---------------------------------------------------------------------------


def _resolve_boundary_angles(
    spec: float | Mapping[Vertex, float] | Callable[[Vertex], float],
    boundary_vertices: list[Vertex],
) -> dict[Vertex, float]:
    """Materialize a boundary_angles spec into dict[Vertex, float]; validate (0, 2π)."""
    result: dict[Vertex, float] = {}
    if isinstance(spec, Mapping):
        for v in boundary_vertices:
            if v not in spec:
                raise ValueError(f"boundary_angles dict is missing entry for boundary vertex {v}.")
            theta = float(spec[v])
            if not (0.0 < theta < 2.0 * np.pi):
                raise ValueError(f"boundary_angles[{v}] must be in (0, 2π); got {theta}.")
            result[v] = theta
        return result
    if callable(spec):
        for v in boundary_vertices:
            theta = float(spec(v))
            if not (0.0 < theta < 2.0 * np.pi):
                raise ValueError(f"boundary_angles({v}) must be in (0, 2π); got {theta}.")
            result[v] = theta
        return result
    theta = float(spec)
    if not (0.0 < theta < 2.0 * np.pi):
        raise ValueError(f"boundary_angles must be in (0, 2π); got {theta}.")
    for v in boundary_vertices:
        result[v] = theta
    return result


def _position_as_2d(p) -> np.ndarray:
    """Coerce a vertex position (real array or complex) into a 2D float array."""
    if isinstance(p, complex) or np.iscomplexobj(p):
        z = complex(p)
        return np.array([z.real, z.imag], dtype=float)
    return np.asarray(p, dtype=float)


def boundary_angles_from_positions(G: EuclideanPositionHEG) -> dict[Vertex, float]:
    """Sum the incident-triangle angles at each boundary vertex from ``v['pos']``.

    For a flat triangulated tiling these angles automatically satisfy
    ``Σ_bdy θ = π(F − 2 V_int)`` (Gauss-Bonnet), making the result a valid
    input for :func:`pack_euclidean` in angle-prescribed mode. Suitable for
    matching a packing's combinatorics-shape to an existing flat tiling.

    Args:
        G: Half-edge graph with ``v['pos']`` populated on boundary vertices
            (and their interior triangle neighbors).

    Returns:
        Mapping from each boundary vertex to its total incident-triangle angle.
    """
    result: dict[Vertex, float] = {}
    for v in G.border_vertices():
        p_v = _position_as_2d(v["pos"])
        total = 0.0
        for h in v.outgoing_iter():
            if h.face is None:
                continue
            p_u = _position_as_2d(h.dest["pos"])
            p_w = _position_as_2d(h.nex.dest["pos"])
            vu = p_u - p_v
            vw = p_w - p_v
            cos_a = float(np.dot(vu, vw) / (np.linalg.norm(vu) * np.linalg.norm(vw)))
            cos_a = max(-1.0, min(1.0, cos_a))
            total += float(np.arccos(cos_a))
        result[v] = total
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
# Vectorized flower indexing
# ---------------------------------------------------------------------------


def _build_flower_arrays(
    vertices: list[Vertex],
    update_verts: list[Vertex],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack incident-triangle data for the to-be-iterated vertices into padded arrays.

    Returns:
        update_idx: (n_update,) int array indexing into ``vertices`` for the
            vertices that will be iterated.
        U, W: (n_update, max_deg) int arrays where row j lists the neighbor
            indices of the j-th update vertex, padded with 0 past its degree.
        valid: (n_update, max_deg) bool array marking real triangle slots.
        degrees: (n_update,) int array of the per-vertex triangle counts.
    """
    idx_of = {v: i for i, v in enumerate(vertices)}
    flowers = [_incident_triangles(v) for v in update_verts]
    degrees_list = [len(f) for f in flowers]
    if not degrees_list:
        empty = np.zeros((0, 0), dtype=np.int64)
        return (
            np.zeros(0, dtype=np.int64),
            empty,
            empty,
            np.zeros((0, 0), dtype=bool),
            np.zeros(0, dtype=np.int64),
        )
    max_deg = max(degrees_list)
    n = len(update_verts)
    U = np.zeros((n, max_deg), dtype=np.int64)
    W = np.zeros((n, max_deg), dtype=np.int64)
    valid = np.zeros((n, max_deg), dtype=bool)
    for j, pairs in enumerate(flowers):
        for k, (u, w) in enumerate(pairs):
            U[j, k] = idx_of[u]
            W[j, k] = idx_of[w]
            valid[j, k] = True
    update_idx = np.fromiter((idx_of[v] for v in update_verts), dtype=np.int64, count=n)
    degrees = np.asarray(degrees_list, dtype=np.int64)
    return update_idx, U, W, valid, degrees


def _euclidean_angle_sums_vec(
    r: np.ndarray,
    update_idx: np.ndarray,
    U: np.ndarray,
    W: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Vectorized euclidean angle sum at each update vertex.

    For each triangle (v, u, w): angle at v = 2 arcsin(sqrt(r_u r_w / ((r_v+r_u)(r_v+r_w)))).
    """
    r_v = r[update_idx][:, None]
    r_u = r[U]
    r_w = r[W]
    denom = (r_v + r_u) * (r_v + r_w)
    # padding slots have denom = 4 r_v^2 (nonzero), but valid mask zeros them out below.
    ratio = (r_u * r_w) / denom
    np.clip(ratio, 0.0, 1.0, out=ratio)
    angles = 2.0 * np.arcsin(np.sqrt(ratio))
    angles *= valid
    return angles.sum(axis=1)


def _bowers_stephenson_euclidean_update_vec(
    r_v: np.ndarray,
    theta: np.ndarray,
    degrees: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Vectorized euclidean Bowers-Stephenson update.

    r_v: (n,) current radii of update vertices.
    theta: (n,) current angle sums.
    degrees: (n,) flower sizes.
    targets: (n,) target angle sums (typically 2π for interior).
    """
    s = np.sin(theta / (2.0 * degrees))
    # Degenerate: s >= 1 means total angle ≥ N·π, can only happen with broken init.
    # Fall back to halving r_v (matches scalar fallback).
    safe = s < 1.0
    s_safe = np.where(safe, s, 0.5)
    r_hat = r_v * s_safe / (1.0 - s_safe)
    s_target = np.sin(targets / (2.0 * degrees))
    new_r = r_hat * (1.0 - s_target) / s_target
    return np.where(safe, new_r, r_v * 0.5)


def _hyperbolic_angle_sums_vec(
    x: np.ndarray,
    update_idx: np.ndarray,
    U: np.ndarray,
    W: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Vectorized hyperbolic angle sum at each update vertex (x-radius parametrisation)."""
    x_v = x[update_idx][:, None]
    a_v = 1.0 - x_v
    x_u = x[U]
    x_w = x[W]
    a_u = 1.0 - x_u
    a_w = 1.0 - x_w
    denom = (1.0 - a_v * a_w) * (1.0 - a_v * a_u)
    num = a_v * x_u * x_w
    # denom > 0 wherever the configuration is non-degenerate; numerator ≥ 0.
    ratio = np.where(denom > 0.0, num / np.where(denom > 0.0, denom, 1.0), 1.0)
    np.clip(ratio, 0.0, 1.0, out=ratio)
    angles = 2.0 * np.arcsin(np.sqrt(ratio))
    angles *= valid
    return angles.sum(axis=1)


def _bowers_stephenson_hyperbolic_update_vec(
    x_v: np.ndarray,
    theta: np.ndarray,
    degrees: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Vectorized hyperbolic Bowers-Stephenson update in x-radius parametrisation.

    Mirrors the euclidean Bowers-Stephenson trick: invert the uniform-neighbor
    formula to recover an effective horocyclic offset ``u_hat = 1 - x_hat``
    from the current angle sum, then solve for the new ``x_v`` that hits the
    target angle sum against those uniform neighbours.

    For one triangle (v, u, u) with all-uniform u-radius and angle α at v::

        sin(α/2) = sqrt(1 − x_v) · (1 − u_hat) / (1 − (1 − x_v) u_hat)

    Step 1 solves this for u_hat given current α; step 2 solves a quadratic in
    ``w = sqrt(1 − x_v_new)`` to hit the target angle.
    """
    n = degrees.astype(float)
    # Numerical floor: x_v ∈ (eps, 1-eps), so v = sqrt(1 - x_v) > 0.
    v_sqrt = np.sqrt(np.clip(1.0 - x_v, 0.0, 1.0))

    s_cur = np.sin(theta / (2.0 * n))
    # Step 1: u_hat = (v - s) / (v (1 - s v))
    # Fallback when v - s < 0 (theta too large for uniform-neighbor consistency)
    # or denominator non-positive: clamp u_hat to a small positive value, which
    # forces neighbors toward horocyclic — the BS update then pulls x_v down.
    denom1 = v_sqrt * (1.0 - s_cur * v_sqrt)
    u_hat = np.where((denom1 > 1e-300) & (v_sqrt > s_cur), (v_sqrt - s_cur) / np.where(denom1 > 0, denom1, 1.0), 0.0)
    np.clip(u_hat, 0.0, 1.0 - 1e-14, out=u_hat)

    # Step 2: solve s_t (1 - a_v u_hat) = sqrt(a_v) (1 - u_hat) for a_v.
    # Quadratic in w = sqrt(a_v):  (s_t u_hat) w^2 + (1 - u_hat) w - s_t = 0.
    # Stable positive root: w = 2 s_t / ((1 - u_hat) + sqrt((1 - u_hat)^2 + 4 s_t^2 u_hat)).
    s_t = np.sin(targets / (2.0 * n))
    one_minus_u = 1.0 - u_hat
    disc = np.sqrt(one_minus_u * one_minus_u + 4.0 * s_t * s_t * u_hat)
    w_new = 2.0 * s_t / (one_minus_u + disc)
    a_new = w_new * w_new
    x_new = 1.0 - a_new
    # Clamp to the open interval used downstream by the layout / solver.
    np.clip(x_new, 1e-14, 1.0 - 1e-14, out=x_new)
    return x_new


# ---------------------------------------------------------------------------
# Stephenson superstep acceleration
# ---------------------------------------------------------------------------

# Maximum bad-cut retries inside a single BS pass before giving up on the
# superstep for that outer iteration. Matches CirclePack's default.
_MAX_BAD_CUTS = 10


def _iterate_with_superstep(
    r_arr: np.ndarray,
    update_idx: np.ndarray,
    U_arr: np.ndarray,
    W_arr: np.ndarray,
    valid_arr: np.ndarray,
    degrees_arr: np.ndarray,
    targets_arr: np.ndarray,
    angle_sum_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    bs_update_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    tol: float,
    max_iter: int,
    max_value: float = float("inf"),
) -> tuple[int, float, bool]:
    """Bowers-Stephenson iteration with Collins-Stephenson superstep acceleration.

    Faithful Jacobi-vectorised port of CirclePack's ``EuclPacker.continueRiffle``
    (Collins & Stephenson, *Comput. Geom.* 25 (2003) 233-256, §3): each outer
    step does a BS pass, then over-relaxes radii along the direction of the
    last pass, then does another BS pass — accepting or reverting based on
    actual vs. predicted error reduction.

    ``r_arr`` is mutated in place. Boundary radii (entries outside
    ``update_idx``) are never touched. ``max_value`` caps the superstep
    extrapolation so that updated radii stay in ``(0, max_value)`` — set to
    ``1.0`` for the hyperbolic x-radius parametrisation (the default
    ``inf`` is appropriate for the unbounded euclidean radii). Returns
    ``(iterations, max_defect, converged)`` where ``max_defect`` is the
    angle-defect L∞ norm on the final iterate.
    """
    if update_idx.size == 0:
        return 0, 0.0, True

    theta = angle_sum_fn(r_arr, update_idx, U_arr, W_arr, valid_arr)
    max_defect = float(np.max(np.abs(theta - targets_arr)))
    if max_defect < tol:
        return 0, max_defect, True

    # === Bootstrap pass (analogue of CirclePack startRiffle) ===
    accumErr2 = float(np.sqrt(np.sum((theta - targets_arr) ** 2)))
    r_arr[update_idx] = bs_update_fn(r_arr[update_idx], theta, degrees_arr, targets_arr)

    m = 1.0
    sct = 1
    key = 1  # current superstep type

    iter_count = 1
    while iter_count < max_iter:
        # Convergence check on whatever state the previous step left us in.
        theta = angle_sum_fn(r_arr, update_idx, U_arr, W_arr, valid_arr)
        max_defect = float(np.max(np.abs(theta - targets_arr)))
        if max_defect < tol:
            return iter_count, max_defect, True

        # === Pass 1 (with bad-cut retries) ===
        R1 = r_arr[update_idx].copy()
        c1 = float(np.sqrt(np.sum((theta - targets_arr) ** 2)))
        r_arr[update_idx] = bs_update_fn(r_arr[update_idx], theta, degrees_arr, targets_arr)
        iter_count += 1
        factor = c1 / accumErr2 if accumErr2 > 0.0 else 0.0

        num_bad_cuts = 0
        while factor >= 1.0 and iter_count < max_iter:
            accumErr2 = c1
            key = 1
            num_bad_cuts += 1
            if num_bad_cuts > _MAX_BAD_CUTS:
                break
            theta = angle_sum_fn(r_arr, update_idx, U_arr, W_arr, valid_arr)
            c1 = float(np.sqrt(np.sum((theta - targets_arr) ** 2)))
            r_arr[update_idx] = bs_update_fn(r_arr[update_idx], theta, degrees_arr, targets_arr)
            iter_count += 1
            factor = c1 / accumErr2 if accumErr2 > 0.0 else 0.0
            # If c1 is already below tol, the next outer-loop convergence check will pick it up.

        if num_bad_cuts > _MAX_BAD_CUTS or factor >= 1.0:
            # Skip superstep this iteration and try again next outer pass.
            accumErr2 = c1
            continue

        # === Superstep extrapolation ===
        R2 = r_arr[update_idx].copy()
        diff = R2 - R1
        # Cap lambda so the extrapolated radii stay inside (0, max_value): if
        # diff < 0, R2 + λ·diff > 0 requires λ < -R2/diff; if diff > 0 and
        # max_value is finite, R2 + λ·diff < max_value requires
        # λ < (max_value - R2)/diff. CirclePack additionally halves the cap
        # for safety.
        lmax = 1.0e4
        shrinking = diff < 0.0
        if shrinking.any():
            lmax = min(lmax, float(np.min(-R2[shrinking] / diff[shrinking])))
        if np.isfinite(max_value):
            growing = diff > 0.0
            if growing.any():
                lmax = min(lmax, float(np.min((max_value - R2[growing]) / diff[growing])))
        lmax = lmax / 2.0

        if key == 1:
            lambda_ = m * factor
            mmax = 0.75 / (1.0 - factor)
            mm = (1.0 + 0.8 / (sct + 1)) * m
            m = mmax if mmax < mm else mm
        else:
            # CirclePack's Type-2 branch reduces (in this codebase) to lambda = factor.
            lambda_ = factor

        if lambda_ > lmax:
            lambda_ = lmax
        if lambda_ < 0.0:
            lambda_ = 0.0

        r_arr[update_idx] = R2 + lambda_ * diff
        sct += 1

        # === Pass 2 on extrapolated radii ===
        theta = angle_sum_fn(r_arr, update_idx, U_arr, W_arr, valid_arr)
        new_accumErr2 = float(np.sqrt(np.sum((theta - targets_arr) ** 2)))
        r_arr[update_idx] = bs_update_fn(r_arr[update_idx], theta, degrees_arr, targets_arr)
        iter_count += 1

        # Acceptance: predicted improvement = factor^lambda_; actual = new_accumErr2/c1.
        pred = factor**lambda_
        act = new_accumErr2 / c1 if c1 > 0.0 else 0.0

        if act < 1.0:
            if act > pred:
                # Some progress, but under-delivered: drop multiplier and flip key 1 → 2.
                m = 1.0
                sct = 0
                if key == 1:
                    key = 2
            accumErr2 = new_accumErr2
        else:
            # No improvement from the extrapolation — roll radii back to R2.
            m = 1.0
            sct = 0
            r_arr[update_idx] = R2
            accumErr2 = c1
            if key == 2:
                key = 1

    # Out of iterations: report the latest state's defect.
    theta = angle_sum_fn(r_arr, update_idx, U_arr, W_arr, valid_arr)
    max_defect = float(np.max(np.abs(theta - targets_arr)))
    return iter_count, max_defect, max_defect < tol


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
    G: EuclideanPositionHEG,
    boundary_radii: float | Mapping[Vertex, float] | Callable[[Vertex], float] | None = None,
    *,
    boundary_angles: float | Mapping[Vertex, float] | Callable[[Vertex], float] | str | None = None,
    alpha: Vertex | None = None,
    beta: Vertex | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    copy_graph: bool = True,
) -> EuclideanPositionHEG:
    """Compute a euclidean circle packing with prescribed boundary data.

    The boundary can be pinned in one of two ways (mutually exclusive):

    - ``boundary_radii``: fix each boundary vertex's radius; iterate only
      interior vertices toward angle sum 2π. This is the classical
      euclidean boundary-value problem.
    - ``boundary_angles``: fix each boundary vertex's total incident-triangle
      angle; iterate every vertex (boundary and interior). For a flat input
      tiling pass ``"from_positions"`` to read the angles off the input
      positions. The angles must satisfy Gauss-Bonnet:
      ``Σ_bdy θ = π(F − 2 V_int)``.

    If neither is given, defaults to ``boundary_radii=1.0``.

    Args:
        G: Triangulated, simply-connected disk EuclideanPositionHEG.
        boundary_radii: Per-vertex boundary radii. Accepts a positive scalar
            (uniform), Mapping[Vertex, float], or callable Vertex -> float.
        boundary_angles: Per-vertex boundary angle sums in (0, 2π). Accepts
            a scalar (uniform), Mapping, callable, or the string
            ``"from_positions"`` to infer angles from ``G``'s positions.
        alpha: Optional anchor vertex (placed at origin). Defaults to the
            interior vertex with maximum graph-distance to the boundary.
        beta: Optional second anchor (placed on +x axis through alpha).
        tol: Max permitted angle defect over iterated vertices.
        max_iter: Maximum Collins-Stephenson iterations.
        copy_graph: If True (default), the input graph is copied.

    Returns:
        EuclideanPositionHEG with EuclideanGeometry, ``v['pos']`` as 2D real
        array centers, and ``v['radius']`` as positive floats.

    Raises:
        ValueError: input not triangulated/disk; both modes specified;
            boundary_angles violate Gauss-Bonnet; invalid spec values.
        ConvergenceError: iteration did not reach ``tol`` within ``max_iter``.
    """
    _validate_triangulated_disk(G)

    if boundary_radii is not None and boundary_angles is not None:
        raise ValueError("pack_euclidean: specify either `boundary_radii` or `boundary_angles`, not both.")
    if boundary_radii is None and boundary_angles is None:
        boundary_radii = 1.0

    P = G.copy() if copy_graph else G
    boundary_verts = P.border_vertices()
    boundary_set = set(boundary_verts)
    interior_verts = [v for v in P.vertices if v not in boundary_set]

    targets: dict[Vertex, float]
    update_verts: list[Vertex]
    radii: dict[Vertex, float]

    if boundary_angles is not None:
        if isinstance(boundary_angles, str):
            if boundary_angles == "from_positions":
                boundary_angles = boundary_angles_from_positions(P)
            else:
                raise ValueError(f"Unknown boundary_angles string: {boundary_angles!r}. " f"Expected 'from_positions'.")
        angle_spec = _resolve_boundary_angles(boundary_angles, boundary_verts)
        n_faces = len(P.faces)
        n_int = len(interior_verts)
        expected_sum = float(np.pi * (n_faces - 2 * n_int))
        actual_sum = float(sum(angle_spec.values()))
        if not np.isclose(actual_sum, expected_sum, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"pack_euclidean: boundary_angles sum to {actual_sum:.10f}, but "
                f"Gauss-Bonnet requires Σ θ = π(F − 2·V_int) = π·({n_faces} − 2·{n_int}) "
                f"= {expected_sum:.10f}. Defect = {actual_sum - expected_sum:.3e}."
            )
        targets = {v: 2.0 * np.pi for v in interior_verts}
        targets.update(angle_spec)
        update_verts = list(targets.keys())
        radii = {v: 1.0 for v in P.vertices}
    else:
        radii_spec = _resolve_boundary_radii(boundary_radii, boundary_verts)
        targets = {v: 2.0 * np.pi for v in interior_verts}
        update_verts = interior_verts
        boundary_mean = float(np.mean(list(radii_spec.values()))) if radii_spec else 1.0
        radii = {}
        for v in P.vertices:
            if v in boundary_set:
                radii[v] = radii_spec[v]
            else:
                radii[v] = boundary_mean

    all_vertices = list(P.vertices)
    idx_of = {v: i for i, v in enumerate(all_vertices)}
    r_arr = np.fromiter((radii[v] for v in all_vertices), dtype=float, count=len(all_vertices))
    update_idx, U_arr, W_arr, valid_arr, degrees_arr = _build_flower_arrays(all_vertices, update_verts)
    targets_arr = np.fromiter((targets[v] for v in update_verts), dtype=float, count=len(update_verts))

    _, final_defect, converged = _iterate_with_superstep(
        r_arr,
        update_idx,
        U_arr,
        W_arr,
        valid_arr,
        degrees_arr,
        targets_arr,
        _euclidean_angle_sums_vec,
        _bowers_stephenson_euclidean_update_vec,
        tol,
        max_iter,
    )
    if not converged:
        raise ConvergenceError(
            f"pack_euclidean: did not converge in {max_iter} iterations; "
            f"final max angle defect = {final_defect:.3e}"
        )
    radii = {v: float(r_arr[i]) for v, i in idx_of.items()}

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


# ---------------------------------------------------------------------------
# Hyperbolic layout in euclidean Poincaré-disk coordinates
# ---------------------------------------------------------------------------


def _x_radius_from_euclidean(c: complex, r_e: float) -> float:
    """Recover the x-radius of a Poincaré-disk circle from (euc center, euc radius).

    Inverse of the standard (z, x) → (c, r_e) map. Derived from
    ``h = arctanh(2 r_e / (1 - |c|^2 + r_e^2))`` and ``x = 1 - exp(-2h)``::

        x = 4 r_e / ((1 + r_e)^2 - |c|^2)

    For a horocycle (|c| = 1 - r_e), this returns 1.0 exactly.
    """
    abs_c2 = float(c.real * c.real + c.imag * c.imag)
    return 4.0 * r_e / ((1.0 + r_e) ** 2 - abs_c2)


def _eucl_radius_at_origin(x: float) -> float:
    """Euclidean radius of a Poincaré-disk circle with x-radius ``x`` centered at origin."""
    if x >= 1.0:
        raise ValueError("Cannot place a horocycle at the Poincaré disk origin.")
    s = np.sqrt(1.0 - x)
    return float((1.0 - s) / (1.0 + s))


def _eucl_radius_tangent_to_origin(x: float, r_origin: float) -> float:
    """Euclidean radius of a circle with x-radius ``x`` placed tangent to a circle at origin.

    The new circle's euclidean center is at distance ``r_origin + r_new`` from the origin.
    Solved in closed form from x = 4r / ((1+r)^2 - |c|^2) with |c| = r_origin + r.
    For x = 1 (horocycle): r = (1 - r_origin) / 2.
    """
    if x >= 1.0:
        return (1.0 - r_origin) / 2.0
    # x (1 - r_origin^2) = (4 - 2 x (1 - r_origin)) r
    num = x * (1.0 - r_origin * r_origin)
    den = 4.0 - 2.0 * x * (1.0 - r_origin)
    return float(num / den)


def _place_third_circle(c_a: complex, r_a: float, c_b: complex, r_b: float, x_c: float) -> tuple[complex, float]:
    """Solve for (c_C, r_C) given two placed circles and the target x-radius.

    Constraints: euclidean tangency to A and B, and the Poincaré-disk coupling
    between (c_C, r_C) and x_C. Returns the CCW-side solution (i.e., c_C is
    to the left of the directed segment c_A -> c_B).

    For horocycles (``x_c = 1.0``), the coupling reduces to ``|c_C| + r_C = 1``.

    Closed-form solve: parametrise ``c_C`` by ``r_C`` via the euclidean law of
    cosines for the triangle ``(c_A, c_B, c_C)`` and substitute into the
    x-radius equation ``|c_C|² = (1 + r_C)² − 4 r_C / x_c``. With
    ``sin α = 2√T / ((r_A+r_B)(r_A+r_C))`` where ``T = r_A r_B r_C (r_A+r_B+r_C)``
    (Heron's identity for a tangent-circle triangle), this collapses to a
    quadratic in ``r_C``. The CCW-side root is selected by the sign of the
    auxiliary linear function ``P(r_C)``.
    """
    delta = c_b - c_a
    d_ab = abs(delta)
    if d_ab == 0.0:
        raise ValueError("_place_third_circle: c_a and c_b coincide.")
    # rot = c_a · conj(delta) / d_ab = A + i B
    rot = c_a * delta.conjugate() / d_ab
    A = rot.real
    B = rot.imag
    abs_c_a2 = c_a.real * c_a.real + c_a.imag * c_a.imag
    rab = r_a + r_b

    # P(r) = P0 + P1 r  is the residual of |c_C|^2 = (1+r)^2 - 4 r / x_c with the
    # α-independent terms collected and the (sin α)-coefficient moved to the
    # other side of the equation. Then P(r) = -4 B √T(r); squaring gives a
    # quadratic.
    p0 = rab * (abs_c_a2 + r_a * r_a - 1.0 + 2.0 * A * r_a)
    p1 = 2.0 * rab * (r_a - 1.0) + 4.0 * rab / x_c + 2.0 * A * (r_a - r_b)
    aa = p1 * p1 - 16.0 * B * B * r_a * r_b
    bb = 2.0 * p0 * p1 - 16.0 * B * B * r_a * r_b * rab
    cc = p0 * p0

    if abs(aa) < 1e-300:
        if abs(bb) < 1e-300:
            raise ValueError("_place_third_circle: degenerate quadratic (a = b = 0).")
        r_c = -cc / bb
    else:
        disc = bb * bb - 4.0 * aa * cc
        if disc < 0.0:
            # Tiny negative from FP noise — clamp; large negative means real
            # geometric infeasibility.
            if disc < -1e-12 * max(1.0, abs(bb * bb)):
                raise ValueError(f"_place_third_circle: no real solution (disc={disc:.3e}, x_c={x_c}).")
            disc = 0.0
        sq = np.sqrt(disc)
        r1 = (-bb + sq) / (2.0 * aa)
        r2 = (-bb - sq) / (2.0 * aa)
        # Pick the positive root whose P(r) has sign matching the CCW branch
        # (P = -4 B √T, so sign(P) = -sign(B)).
        candidates: list[float] = []
        for r in (r1, r2):
            if r <= 1e-15:
                continue
            if B == 0.0:
                candidates.append(r)
                continue
            pval = p0 + p1 * r
            if pval * (-B) >= -1e-12 * max(1.0, abs(pval)):
                candidates.append(r)
        if not candidates:
            # Sign disambiguation failed: fall back to any positive root.
            candidates = [r for r in (r1, r2) if r > 0.0]
        if not candidates:
            raise ValueError(f"_place_third_circle: no positive root (roots: {r1}, {r2}, x_c={x_c}).")
        r_c = min(candidates)

    u = delta / d_ab

    # Polish r_c with one Newton step on the original constraint. The squared
    # quadratic above carries ~1e-8 cancellation error in some configurations;
    # one Newton step using a finite-difference derivative recovers near
    # machine precision at a cost of two extra residual evaluations.
    def _residual(r: float) -> float:
        p_ = r_a + r
        T_ = r_a * r_b * r * (r_a + r_b + r)
        if T_ < 0.0:
            T_ = 0.0
        cos_a = (r_a * (r_a + r_b + r) - r_b * r) / (rab * p_)
        if cos_a > 1.0:
            cos_a = 1.0
        elif cos_a < -1.0:
            cos_a = -1.0
        sin_a = 2.0 * math.sqrt(T_) / (rab * p_)
        c_local = c_a + p_ * u * complex(cos_a, sin_a)
        abs_c2 = c_local.real * c_local.real + c_local.imag * c_local.imag
        if x_c >= 1.0:
            return math.sqrt(abs_c2) + r - 1.0
        denom = (1.0 + r) * (1.0 + r) - abs_c2
        if denom <= 0.0:
            return 1.0 - x_c
        return 4.0 * r / denom - x_c

    f_val = _residual(r_c)
    h_step = max(1e-12, r_c * 1e-7)
    df = (_residual(r_c + h_step) - _residual(r_c - h_step)) / (2.0 * h_step)
    if df != 0.0 and math.isfinite(df):
        step = f_val / df
        if math.isfinite(step) and abs(step) < r_c:
            r_c -= step

    # Recover c_c via law of cosines. Heron's identity for tangent-circle
    # triangles gives sin α directly, avoiding sqrt(1 − cos² α) cancellation
    # when α is small.
    p = r_a + r_c
    T = r_a * r_b * r_c * (r_a + r_b + r_c)
    cos_alpha = (r_a * (r_a + r_b + r_c) - r_b * r_c) / (rab * p)
    if cos_alpha > 1.0:
        cos_alpha = 1.0
    elif cos_alpha < -1.0:
        cos_alpha = -1.0
    sin_alpha = 2.0 * math.sqrt(max(0.0, T)) / (rab * p)
    c_c = c_a + p * u * complex(cos_alpha, sin_alpha)
    return c_c, r_c


def _layout_hyperbolic(
    G,
    x_radii: dict[Vertex, float],
    alpha: Vertex,
    beta: Vertex,
) -> tuple[dict[Vertex, complex], dict[Vertex, float]]:
    """Place each circle's euclidean (center, radius) in the Poincaré disk.

    Returns ``(centers, radii)`` mappings from Vertex to complex euclidean
    centers and float euclidean radii. Handles both interior (0 < x < 1) and
    horocycle (x = 1) circles uniformly.
    """
    centers: dict[Vertex, complex] = {}
    radii: dict[Vertex, float] = {}

    if x_radii[alpha] >= 1.0:
        raise ValueError("alpha must be an interior vertex (x_radius < 1).")
    centers[alpha] = complex(0.0, 0.0)
    radii[alpha] = _eucl_radius_at_origin(x_radii[alpha])

    r_beta = _eucl_radius_tangent_to_origin(x_radii[beta], radii[alpha])
    centers[beta] = complex(radii[alpha] + r_beta, 0.0)
    radii[beta] = r_beta

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
        if c not in centers:
            c_c, r_c = _place_third_circle(centers[a], radii[a], centers[b], radii[b], x_radii[c])
            centers[c] = c_c
            radii[c] = r_c
        for h_other in (h.nex, h.nex.nex):
            enqueue(h_other.rev)

    if len(centers) != len(list(G.vertices)):
        missing = [v for v in G.vertices if v not in centers]
        raise RuntimeError(f"Hyperbolic layout did not reach all vertices ({len(missing)} unplaced).")
    return centers, radii


# ---------------------------------------------------------------------------
# Public entry point: pack_hyperbolic
# ---------------------------------------------------------------------------


def pack_hyperbolic(
    G: EuclideanPositionHEG,
    boundary_x_radii: float | Mapping[Vertex, float] | Callable[[Vertex], float] = 1.0,
    *,
    alpha: Vertex | None = None,
    beta: Vertex | None = None,
    tol: float = 1e-10,
    max_iter: int = 10_000,
    copy_graph: bool = True,
) -> EuclideanPositionHEG:
    """Compute a hyperbolic circle packing in the Poincaré disk.

    Boundary vertices are assigned the prescribed x-radii (where x = 1 - exp(-2h),
    h being the hyperbolic radius); interior x-radii are iterated via
    Collins-Stephenson to satisfy hyperbolic angle sum = 2*pi. Setting
    ``boundary_x_radii = 1.0`` produces a maximal packing (boundary horocycles).

    The output stores the **euclidean** (Poincaré-disk) representation::

        v['pos']    — euclidean center of the circle (complex, inside unit disk)
        v['radius'] — euclidean radius of the circle (float)

    Two hyperbolic disks are hyperbolically tangent iff their euclidean
    representations are euclidean tangent, so adjacent circles satisfy
    ``|c_u − c_v| = r_u + r_v``. The intrinsic x-radius of each vertex is
    recoverable via :func:`_x_radius_from_euclidean` (and equals 1 for
    boundary horocycles).

    Args:
        G: Triangulated, simply-connected disk EuclideanPositionHEG.
        boundary_x_radii: Per-vertex boundary x-radii in (0, 1]. Accepts a
            scalar (uniform), Mapping, or callable. Defaults to 0.5; pass
            ``1.0`` for the maximal packing.
        alpha: Optional anchor vertex (placed at Poincaré disk origin).
            Must be interior. Defaults to the interior vertex with max
            graph-distance to the boundary.
        beta: Optional second anchor (placed on +x axis through alpha).
        tol: Max angle defect over interior vertices.
        max_iter: Max Collins-Stephenson iterations.
        copy_graph: If True (default), return a new EHEG; otherwise mutate.

    Returns:
        EuclideanPositionHEG with PoincareDiskModel geometry.
    """
    _validate_triangulated_disk(G)

    P = G.copy() if copy_graph else G

    boundary_verts = P.border_vertices()
    radii_spec = _resolve_boundary_radii(boundary_x_radii, boundary_verts)
    for v, x in radii_spec.items():
        if not (0.0 < x <= 1.0):
            raise ValueError(f"pack_hyperbolic: boundary x-radius must be in (0, 1]; got {x} for vertex {v}.")
    boundary_set = set(boundary_verts)

    # Init: boundary at spec, interior uniformly at 0.5 (mid-range)
    x_radii: dict[Vertex, float] = {}
    for v in P.vertices:
        if v in boundary_set:
            x_radii[v] = radii_spec[v]
        else:
            x_radii[v] = 0.5

    interior_verts = [v for v in P.vertices if v not in boundary_set]
    all_vertices = list(P.vertices)
    idx_of = {v: i for i, v in enumerate(all_vertices)}
    x_arr = np.fromiter((x_radii[v] for v in all_vertices), dtype=float, count=len(all_vertices))
    update_idx, U_arr, W_arr, valid_arr, degrees_arr = _build_flower_arrays(all_vertices, interior_verts)
    targets_arr = np.full(len(interior_verts), 2.0 * np.pi, dtype=float)

    _, final_defect, converged = _iterate_with_superstep(
        x_arr,
        update_idx,
        U_arr,
        W_arr,
        valid_arr,
        degrees_arr,
        targets_arr,
        _hyperbolic_angle_sums_vec,
        _bowers_stephenson_hyperbolic_update_vec,
        tol,
        max_iter,
        max_value=1.0,
    )
    if not converged:
        raise ConvergenceError(
            f"pack_hyperbolic: did not converge in {max_iter} iterations; "
            f"final max angle defect = {final_defect:.3e}"
        )
    x_radii = {v: float(x_arr[i]) for v, i in idx_of.items()}

    if alpha is None:
        alpha = _choose_alpha(P)
    if beta is None:
        beta = _choose_beta(alpha)

    centers, eucl_radii = _layout_hyperbolic(P, x_radii, alpha, beta)

    for v in P.vertices:
        v["pos"] = centers[v]
        v["radius"] = eucl_radii[v]
    P.geometry = PoincareDiskModel
    P.recompute_lengths_and_angles()
    return P
