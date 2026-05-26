"""Alternating-flagstone origami crease patterns.

This module is the high-level pipeline around
:func:`eucare.conway.alternating_flagstone_graph`. The Conway operator alone
produces a *topological* crease pattern: it does not know that, for the
result to fold flat, every flagstone must be a similar (rotated, uniformly
scaled) copy of the original face, and that the crease lengths around each
star vertex must match.

The pipeline implemented here is:

1. :func:`build_structure` applies ``alternating_flagstone_graph`` to a
   tiling ``G`` and walks the resulting graph to identify the per-face
   "flagstones", the per-vertex "stars", and the radial / diagonal creases
   that connect them. Initial mountain/valley assignments are also placed
   here.
2. :func:`optimize_alternating_flagstone` adjusts a single global
   ``scale``, a per-flagstone rotation and 2D offset, and a 2D position
   for every star, so that the diagonal-crease lengths and star-arm
   lengths satisfy the foldability constraints. This step requires
   PyTorch (the optional ``torch`` extra); the rest of the module does
   not.
3. :func:`connection_length_metric` reports how close the optimised CP
   is to a foldable solution.
4. :func:`extend_border` mirrors the outermost flagstone vertices across
   each border edge to produce a CP whose boundary closes properly.
5. :func:`cut_twist_centres` removes a small polygon around every star
   vertex, producing a CP suitable for physical folding (or for
   :func:`eucare.overlap.fold_wireframe`).
6. :func:`subdivide_ridges_for_curved_fold` subdivides each ridge crease
   so that, when folded, it traces an approximate circular arc — the
   input format expected by https://origamisimulator.org/.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .base import angle_to_axis, line_intersection, unit_vector
from .conway import alternating_flagstone_graph
from .cutting import cut_out_poly
from .half import Face, GeometricHEG, HalfEdge, Vertex
from .overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY, color_creases
from .rendering import inset_poly

if TYPE_CHECKING:
    from .half import EuclideanPositionHEG


@dataclass
class AlternatingFlagstoneStructure:
    """Bookkeeping for an alternating-flagstone CP.

    Attributes:
        original: The input tiling (a copy of the user's graph; mutating it
            will not affect the user-supplied original).
        CP: The crease pattern produced by
            :func:`eucare.conway.alternating_flagstone_graph`. Vertices
            originating from a vertex of *original* carry a ``pre_conway``
            attribute that points to that vertex; faces originating from a
            face of *original* carry the analogous attribute.
        flagstone_faces: The faces of *CP* that came from a face of *original*,
            in the same order as ``original_faces``.
        original_faces: For each entry of ``flagstone_faces``, the
            corresponding face of *original*.
        star_vertices: The vertices of *CP* that came from a vertex of
            *original*.
        flagstone_corners: The "free" flagstone-corner vertices of *CP*
            (one per (face, vertex) pair of *original*).
        f_to_flagstone: Maps a face of *original* to its flagstone face in *CP*.
        v_to_star: Maps a vertex of *original* to its star vertex in *CP*.
        corner_to_original: Maps ``(corner, flagstone_face)`` to the
            ``(vertex, face)`` of *original* it came from.
        original_to_corner: The inverse of ``corner_to_original``.
        star_groups: For each star vertex, the set of flagstone-corner
            vertices that should be equidistant from it.
    """

    original: "EuclideanPositionHEG"
    CP: "EuclideanPositionHEG"
    flagstone_faces: list = field(default_factory=list)
    original_faces: list = field(default_factory=list)
    star_vertices: list = field(default_factory=list)
    flagstone_corners: list = field(default_factory=list)
    f_to_flagstone: dict = field(default_factory=dict)
    v_to_star: dict = field(default_factory=dict)
    corner_to_original: dict = field(default_factory=dict)
    original_to_corner: dict = field(default_factory=dict)
    star_groups: dict = field(default_factory=dict)


def build_structure(
    G: "EuclideanPositionHEG",
    t: float = 0.5,
    *,
    assign_initial_creases: bool = True,
) -> AlternatingFlagstoneStructure:
    """Apply the alternating-flagstone Conway operator and build lookup tables.

    Args:
        G: The input tiling. Will be copied; the user's graph is not mutated.
        t: Parameter forwarded to
            :func:`eucare.conway.alternating_flagstone_graph` controlling how
            far the flagstones shrink topologically.
        assign_initial_creases: If True, set every flagstone-perimeter
            half-edge to MOUNTAIN and every other half-edge to VALLEY,
            then call :func:`eucare.overlap.color_creases`.
    """
    # Keep `original` as an untouched reference graph, then build a second
    # copy that we feed to the Conway operator (which mutates / consumes it).
    # The `pre_conway` attributes that the operator stamps onto CP point
    # into this intermediate copy, which `v_map`/`f_map` map back to
    # `original`.
    original = cast("EuclideanPositionHEG", G.copy())
    pre_CP, (v_map, _, f_map) = original.copy(return_mappings=True)
    v_map_rev = {v: k for k, v in v_map.items()}
    f_map_rev = {f: k for k, f in f_map.items()}

    # Drop any pre_conway attributes that the input graph might carry from a
    # prior Conway operator. The Conway operator stamps a fresh ``pre_conway``
    # attribute on every new element, and the leftover attributes from a prior
    # call would otherwise be re-mapped through the wrong ``obj_map`` (or worse,
    # collide on the same Vertex/Face objects). See issue surfaced in the
    # Alternating Flagstones tutorial when chaining ``dual_graph`` with this.
    for objs in (pre_CP.vertices, pre_CP.halfedges, pre_CP.faces):
        for obj in objs:
            if "pre_conway" in obj.attributes:
                del obj["pre_conway"]

    CP = cast("EuclideanPositionHEG", alternating_flagstone_graph(t=t)(cast(GeometricHEG, pre_CP)))

    # CP-iteration order is the single authoritative ordering. ``original_faces``
    # is built parallel to ``flagstone_faces`` so that index ``i`` in either list
    # always refers to the same flagstone — this keeps the optimiser's
    # per-face tensors and the per-corner ``flagstone_corners`` list in sync.
    flagstone_faces = [f for f in CP.faces if isinstance(f.attributes.get("pre_conway", None), Face)]
    original_faces = [f_map_rev[f["pre_conway"]] for f in flagstone_faces]
    f_to_flagstone = dict(zip(original_faces, flagstone_faces))

    star_vertices_all = [v for v in CP.vertices if isinstance(v.attributes.get("pre_conway", None), Vertex)]
    v_to_star = {v_map_rev[v["pre_conway"]]: v for v in star_vertices_all}
    star_vertices = list(star_vertices_all)

    # walk every flagstone to find its corners and the star they sit next to
    flagstone_corners: list = []
    corner_to_original: dict = {}
    star_groups: dict = defaultdict(set)
    for f0, f in zip(original_faces, flagstone_faces):
        for v0 in f0.vertex_iter():
            star = v_to_star[v0]
            for h in f.halfedge_iter():
                if h.rev.nex.dest is star:
                    h = h.nex
                    flagstone_corners.append(h.orig)
                    corner_to_original[(h.orig, f)] = (v0, f0)
                    star_groups[star].add(h.orig)
                    break

    original_to_corner = {v: k for k, v in corner_to_original.items()}

    structure = AlternatingFlagstoneStructure(
        original=original,
        CP=CP,
        flagstone_faces=flagstone_faces,
        original_faces=original_faces,
        star_vertices=star_vertices,
        flagstone_corners=flagstone_corners,
        f_to_flagstone=f_to_flagstone,
        v_to_star=v_to_star,
        corner_to_original=corner_to_original,
        original_to_corner=original_to_corner,
        star_groups=dict(star_groups),
    )

    if assign_initial_creases:
        _assign_initial_creases(structure)

    return structure


def _assign_initial_creases(structure: AlternatingFlagstoneStructure) -> None:
    """Mountain on flagstone perimeters, valley on every other crease."""
    CP = structure.CP
    for h in CP.halfedges:
        h[CREASE_ASSIGNMENT] = VALLEY

    for f in structure.flagstone_faces:
        for h in f.halfedge_iter():
            ridge = h.rev.nex
            if ridge.rev.on_border():
                continue
            ridge[CREASE_ASSIGNMENT] = ridge.rev[CREASE_ASSIGNMENT] = MOUNTAIN

    color_creases(CP)


# ---------------------------------------------------------------------------
# Metric


def _edge_length(h: HalfEdge) -> float:
    return float(np.linalg.norm(h.orig["pos"] - h.dest["pos"]))


def connection_length_metric(structure: AlternatingFlagstoneStructure) -> dict:
    """Quantify how close the CP is to a foldable alternating-flagstone solution.

    Returns a dict with keys:

    - ``max_relative_length_error``: max relative discrepancy between a
      diagonal-crease length and the corresponding flagstone-side length.
    - ``max_angle_error_rad`` / ``max_angle_error_deg``: max mismatch
      between the angle of the original-tiling edge and the angle of the
      matching ridge crease.
    - ``summary``: a human-readable, multi-line string suitable for
      printing or for the ``extra_info`` argument of
      :func:`eucare.overlap.save_results`.
    """
    structure.CP.recompute_lengths_and_angles()
    rel_errors = []
    angle_errors = []
    for f0 in structure.original_faces:
        f = structure.f_to_flagstone[f0]
        for h0 in f0.halfedge_iter():
            if h0.rev.on_border():
                continue
            corner, _ = structure.original_to_corner[(h0.orig, f0)]
            h = next(h for h in f.halfedge_iter() if h.orig is corner).rev
            ridge = h.nex.rev
            connection = ridge.nex.rev

            length = _edge_length(connection)
            target = _edge_length(h)
            rel_errors.append((length - target) / target)

            angle_errors.append(h["in_angle"] - ridge["in_angle"])

    rel_max = float(np.max(np.abs(rel_errors))) if rel_errors else 0.0
    ang_max_rad = float(np.max(np.abs(angle_errors))) if angle_errors else 0.0
    ang_max_deg = ang_max_rad * 180 / np.pi

    summary = (
        f"Maximum relative error in connection length is {rel_max:4.2%}.\n"
        f"Maximum mismatch between angles is {ang_max_deg:4.2f}°."
    )
    return {
        "max_relative_length_error": rel_max,
        "max_angle_error_rad": ang_max_rad,
        "max_angle_error_deg": ang_max_deg,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Optimisation (requires torch)


_TORCH_HINT = (
    "optimize_alternating_flagstone requires PyTorch. "
    "Install the optional 'torch' extra: `pip install eucare[torch]` "
    "(or `pip install torch einops`)."
)


def optimize_alternating_flagstone(
    structure: AlternatingFlagstoneStructure,
    *,
    n_steps: int = 10000,
    initial_scale: float = 0.4,
    lr_offset: float = 1e-3,
    lr_angle: float = 1e-2,
    lr_star: float = 1e-2,
    ramp_step: int = 150,
    ramp_factor: float = 10.0,
    momentum_after_ramp: float = 0.9,
    tol: float = 1e-9,
    progress: bool = False,
) -> list[float]:
    """Optimise flagstone rotations / offsets / star positions for foldability.

    Variables (all per-flagstone, plus per-star):

    - ``scale`` (scalar, shared): uniform shrink factor of every flagstone.
    - ``angle_i`` (scalar, per flagstone): rotation around the original
      face midpoint.
    - ``offset_i`` (2-vector, per flagstone): post-rotation translation.
    - ``star_j`` (2-vector, per star vertex): position of the star.

    Loss = (sum of squared errors of diagonal-crease lengths) +
           (sum of squared deviations of star-arm lengths from their mean).

    Mutates ``structure.CP`` in place. Returns the per-step training loss
    (useful for plotting / convergence diagnostics).

    Raises:
        ModuleNotFoundError: if PyTorch is not installed.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on environments without torch
        raise ModuleNotFoundError(_TORCH_HINT) from exc

    CP = structure.CP

    flagstone_faces = structure.flagstone_faces
    original_faces = structure.original_faces
    star_vertices = structure.star_vertices
    corners = structure.flagstone_corners
    star_groups = structure.star_groups
    original_to_corner = structure.original_to_corner

    # Per-flagstone reference coordinates: the *original* face's vertex positions
    # in `f0.vertex_iter()` cyclic order. The forward pass shrinks them by
    # `scale` toward the face midpoint, so `scale=0.4` corresponds to flagstones
    # that are 40% of the original face — independent of the topological `t`
    # passed to ``alternating_flagstone_graph``.
    face_coords0 = [torch.tensor(np.stack([v0["pos"] for v0 in f0.vertex_iter()])).float() for f0 in original_faces]
    rotation_centers = torch.from_numpy(np.stack([f0.midpoint() for f0 in original_faces])).float()
    initial_star_points = torch.from_numpy(np.stack([v["pos"] for v in star_vertices])).float()

    # Build the connection / target indices.
    #   connection: indices of the two corner vertices forming a diagonal crease.
    #   target_length: indices of the two flagstone-side endpoints whose
    #                  current distance is the *target* length.
    corner_index = {v: i for i, v in enumerate(corners)}
    connections_list: list[list[int]] = []
    target_length_indices_list: list[list[int]] = []
    for h in structure.original.halfedges_representing_edges():
        if h.on_border() or h.rev.on_border():
            continue
        va, _ = original_to_corner[(h.orig, h.face)]
        vb, _ = original_to_corner[(h.dest, h.rev.face)]
        vc, _ = original_to_corner[(h.dest, h.face)]
        connections_list.append([corner_index[va], corner_index[vb]])
        target_length_indices_list.append([corner_index[va], corner_index[vc]])
    connections = torch.LongTensor(connections_list)
    target_length_indices = torch.LongTensor(target_length_indices_list)

    star_group_indices = [[corner_index[c] for c in star_groups[star]] for star in star_vertices]

    # Parameters
    n_faces = len(flagstone_faces)
    scale = torch.nn.Parameter(torch.tensor(initial_scale).float())
    angles = torch.nn.Parameter(torch.zeros(n_faces).float())
    offsets = torch.nn.Parameter(torch.zeros(n_faces, 2).float())
    star_points = torch.nn.Parameter(initial_star_points.clone())

    def rot_mat(a: Any) -> Any:
        s, c = torch.sin(a), torch.cos(a)
        return torch.stack([c, -s, s, c], dim=-1).view(-1, 2, 2)

    def forward() -> tuple[Any, Any]:
        rms = rot_mat(angles)
        out = []
        for coords, center, R, off in zip(face_coords0, rotation_centers, rms, offsets):
            out.append(((coords - center) @ R * scale) + center + off)
        return torch.cat(out), star_points

    def loss_fn(face_coords: Any, star_coords: Any) -> Any:
        # connection loss
        a = face_coords[connections]
        l = (a[:, 1] - a[:, 0]).pow(2).sum(-1).sqrt()
        b = face_coords[target_length_indices]
        target = (b[:, 1] - b[:, 0]).pow(2).sum(-1).sqrt()
        loss = (l - target).pow(2).sum()
        # star loss
        for group, center in zip(star_group_indices, star_coords):
            arm = (face_coords[group] - center[None]).pow(2).sum(-1).sqrt()
            loss = loss + (arm - arm.mean()).pow(2).sum()
        return loss

    optimizer = torch.optim.SGD(
        [
            {"params": [offsets], "lr": lr_offset},
            {"params": [angles], "lr": lr_angle},
            {"params": [star_points], "lr": lr_star},
        ],
        lr=0,
        momentum=0.0,
    )

    # writeable view onto the CP positions in the order (corners ..., stars ...)
    position_view = np.stack([v["pos"] for v in corners + star_vertices])
    for v, p in zip(corners + star_vertices, position_view):
        v["pos"] = p

    iterator: Any = range(n_steps)
    if progress:
        try:  # pragma: no cover
            from tqdm.auto import tqdm

            iterator = tqdm(iterator)
        except ModuleNotFoundError:
            pass

    training_curve: list[float] = []
    for i in iterator:
        optimizer.zero_grad()
        face_coords, star_coords = forward()
        loss = loss_fn(face_coords, star_coords)
        loss.backward()
        optimizer.step()
        training_curve.append(float(loss.item()))

        if loss.item() < tol:
            break

        if progress and i % 10 == 0:
            iterator.set_postfix(loss=loss.item())

        if i == ramp_step:
            for g in optimizer.param_groups:
                g["lr"] *= ramp_factor
                g["momentum"] = momentum_after_ramp

    # commit final positions
    with torch.no_grad():
        face_coords, star_coords = forward()
    final = np.concatenate([face_coords.detach().numpy(), star_coords.detach().numpy()]).astype(np.float64)
    position_view[:] = final
    for v, p in zip(corners + star_vertices, final):
        v["pos"] = np.asarray(p, dtype=np.float64)

    CP.recompute_lengths_and_angles()
    return training_curve


# ---------------------------------------------------------------------------
# Border processing


def _translation_mat(t: NDArray) -> NDArray:
    m = np.eye(3)
    m[:2, 2] = t
    return m


def _rotation_mat(a: float | NDArray) -> NDArray:
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _mirror_mat(line: NDArray | list[NDArray]) -> NDArray:
    """3x3 affine that mirrors the plane across the line ``[p0, p1]``."""
    t1 = _translation_mat(-line[0])
    t2 = _translation_mat(line[0])
    angle = angle_to_axis(line[1] - line[0])
    r1 = _rotation_mat(-angle)
    r2 = _rotation_mat(angle)
    mx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    return cast(NDArray, t2 @ r2 @ mx @ r1 @ t1)


def _apply_affine(m: NDArray, v: NDArray) -> NDArray:
    return cast(NDArray, (m @ np.concatenate([v, [1]]))[:2])


def extend_border(CP: "EuclideanPositionHEG") -> "EuclideanPositionHEG":
    """Extend the boundary of an alternating-flagstone CP by mirror reflection.

    For every border edge whose adjacent ridge crease is *not* a mountain,
    the border edge is promoted to a mountain crease and a small triangle
    is added outside, with its apex placed by mirror-reflecting the
    inward-pointing ridge across the border edge. This is a *cosmetic*
    finishing pass that gives the CP a clean outer outline whose folded
    model closes properly.

    The input graph is not modified; a new graph is returned.

    .. note::
       The algorithm is geometrically simple but topologically fragile:
       it walks a snapshot of the original border, and a previous
       triangle insertion can absorb a later border edge into an interior
       face. Such edges are silently skipped. As a result the function is
       best-effort — for many tilings (notably triangular bases) it
       processes every border edge cleanly, but on others (square / hex
       bases) some edges remain unextended.
    """
    out = cast("EuclideanPositionHEG", CP.copy())
    for h in out.border_edges():
        if not h.on_border():
            # A previously-added triangle has already absorbed this halfedge.
            continue
        if h.rev.pre[CREASE_ASSIGNMENT] == MOUNTAIN:
            continue
        try:
            v = h.rev.nex.dest
            pos = _apply_affine(_mirror_mat([h.orig["pos"], h.dest["pos"]]), v["pos"])
            rev_face = h.rev.face
            assert rev_face is not None
            h_new, _ = out.subdivide_face(rev_face, h.orig, h.dest)
            h_new[CREASE_ASSIGNMENT] = h_new.rev[CREASE_ASSIGNMENT] = MOUNTAIN
            h_new_2, v_new = out.subdivide_edge(h, pos=pos)
            out.subdivide_face(cast(Face, None), h_new_2.nex.dest, v_new)
        except StopIteration:
            # Border zigzag: skip and continue.
            continue
        h["color_key"] = h.rev["color_key"] = (0, 1, 0)
    color_creases(out)
    out.check_consistency()
    return out


# ---------------------------------------------------------------------------
# Cutting twist centres for a foldable CP


def _edge_midpoint(h: HalfEdge) -> NDArray:
    return cast(NDArray, np.mean([v["pos"] for v in (h.orig, h.dest)], axis=0))


def cut_twist_centres(
    structure: AlternatingFlagstoneStructure,
    *,
    inset: float = -0.05,
) -> "EuclideanPositionHEG":
    """Return a copy of ``structure.CP`` with the centre of every "twist" cut out.

    For each star vertex, a small polygon enclosing the star and the
    incident diagonal creases is removed via
    :func:`eucare.cutting.cut_out_poly`. The remaining graph contains
    only the foldable creases (mountain on the flagstone perimeter,
    valley on the radial creases connecting flagstones to the now-missing
    star).

    Args:
        structure: A structure produced by :func:`build_structure`. Its
            ``CP`` is not modified.
        inset: Negative for outward inset (the default; matches the
            notebook). Forwarded to :func:`eucare.rendering.inset_poly`.
    """
    CP_for_folding_raw, (v_map, _, f_map) = structure.CP.copy(return_mappings=True)
    CP_for_folding = cast("EuclideanPositionHEG", CP_for_folding_raw)

    polys: list[NDArray] = []
    for star, group in structure.star_groups.items():
        star_c = v_map[star]
        if star_c not in CP_for_folding.vertices:
            continue
        group_c = {v_map[v] for v in group}

        if star_c.on_border():
            h = next(h1 for h1 in star_c.incoming_iter() if h1.on_border()).rev
            poly_pts: list[NDArray] = [h.orig["pos"], h.dest["pos"]]
            while True:
                h = h.pre.rev
                if h.on_border():
                    break
                h = h.pre.rev
                poly_pts.append(h.dest["pos"])
            poly_pts.append(_edge_midpoint(h))
            poly = np.stack(poly_pts)
        else:
            poly = np.stack([v["pos"] for v in star_c.vertex_iter() if v in group_c])
        polys.append(poly)

    # delete diagonal folds, set M/V on flagstone perimeter and ridges.
    to_delete = set()
    for f in structure.flagstone_faces:
        f = f_map[f]
        for h in f.halfedge_iter():
            h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = MOUNTAIN
            ridge = h.rev.nex
            if ridge.rev.on_border():
                continue
            ridge[CREASE_ASSIGNMENT] = ridge.rev[CREASE_ASSIGNMENT] = VALLEY
            to_delete.add(ridge.rev.nex)
    color_creases(CP_for_folding)
    CP_for_folding.delete_subset(to_delete)

    for poly in polys:
        cut_out_poly(CP_for_folding, np.asarray(inset_poly(list(poly), inset)), delete_outside=True)
    CP_for_folding.recompute_lengths_and_angles()
    return CP_for_folding


# ---------------------------------------------------------------------------
# Curved-fold subdivision (for Origami Simulator)


def _opposite_triangle_vertex(h: HalfEdge) -> Vertex | None:
    if h.on_border():
        return None
    assert h.face is not None
    if h.face.order() != 3:
        return None
    return h.nex.dest


def subdivide_ridges_for_curved_fold(
    CP: "EuclideanPositionHEG",
    n_subdivisions: int = 15,
) -> "EuclideanPositionHEG":
    """Subdivide ridge creases so they trace circular arcs when folded.

    A *ridge* is a half-edge whose two adjacent faces are triangles whose
    apex vertices are both star vertices (i.e. carry a ``pre_conway``
    attribute pointing to a vertex of the original tiling). Each ridge is
    replaced by ``n_subdivisions`` short edges, with subdivision points
    chosen so that, viewed from the apex, the polyline subtends equal
    angle steps — this yields a smooth circular arc when folded, suitable
    for import into Origami Simulator (https://origamisimulator.org/).

    The input graph is not modified; a new graph is returned.
    """
    if not isinstance(n_subdivisions, int) or n_subdivisions < 2:
        raise ValueError(f"n_subdivisions must be an int >= 2, got {n_subdivisions!r}")

    G = cast("EuclideanPositionHEG", CP.copy())

    to_subdivide = set()
    for h in G.halfedges:
        v1 = _opposite_triangle_vertex(h)
        v2 = _opposite_triangle_vertex(h.rev)
        if v1 is not None and v2 is not None and "pre_conway" in v1 and "pre_conway" in v2:
            h["color_key"] = (1, 1, 0)
            to_subdivide.add(h.pre)

    while to_subdivide:
        h = to_subdivide.pop()
        to_subdivide.discard(h.rev)
        v1 = _opposite_triangle_vertex(h)
        v2 = _opposite_triangle_vertex(h.rev)
        if v1 is None or v2 is None:
            continue

        angle0 = angle_to_axis(h.dest["pos"] - v1["pos"])
        angle1 = angle_to_axis(h.orig["pos"] - v1["pos"])
        delta_angle = (angle0 - angle1) % (2 * np.pi)
        pts = []
        for t in np.linspace(0, 1, n_subdivisions, endpoint=False)[1:]:
            a = angle0 - delta_angle * t
            pts.append(
                line_intersection(
                    np.array([h.orig["pos"], h.dest["pos"]]),
                    np.array([v1["pos"], v1["pos"] + unit_vector(a)]),
                )
            )
        new_vs = [G.subdivide_edge(h, pos=p)[1] for p in pts]

        assert h.face is not None
        for vi in new_vs:
            G.subdivide_face(h.face, v1, vi, color_key=(1, 1, 0))
        if not h.rev.on_border():
            rev_face = h.rev.face
            assert rev_face is not None
            for vi in new_vs:
                G.subdivide_face(rev_face, vi, v2, color_key=(1, 1, 0))

    return G
