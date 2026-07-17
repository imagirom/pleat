"""Parse a GomJau-Hogg code into a finite half-edge graph.

Pipeline:

1. :func:`polygon_placement` parses the first stage (e.g. ``"6-3-3"``) into a
   small seed graph by gluing regular polygons together.
2. :func:`apply_transform` interprets each later stage (e.g. ``"m30"``,
   ``"r(h2)"``, ``"r(c3)"``) as one or more affine transforms.
3. :func:`compile_gjh_graph` applies the transforms iteratively, expanding the
   graph until no new tiles fit within the requested bounding box.
"""

from __future__ import annotations

import itertools
import warnings

import networkx as nx
import numpy as np

import pleat
from pleat.base import angle_to_axis, signed_area
from pleat.conversions import EHEG_from_nx
from pleat.half import EuclideanPositionHEG, Face, Vertex
from pleat.overlap import group_closeby
from pleat.prototiles import RegularEuclideanTile

_EPS = 0.1


# --- Stage 1: polygon placement ---------------------------------------------


def seed_polygon(n: int) -> EuclideanPositionHEG:
    """Construct an isolated regular n-gon graph (oriented for use as a seed)."""
    G = EuclideanPositionHEG(other=RegularEuclideanTile(n).make_graph(add_positions=True)[0])
    if n == 3:
        # Triangles need to be re-oriented so their border edge lies on the negative x side,
        # matching the convention used by ``starting_border``.
        pos = G.get_position_view(return_vertices=False)
        pos *= -1
        pos -= pos.min(0, keepdims=True)
    return G


def _starting_border(G: EuclideanPositionHEG, seed_face: Face):
    """Return the first border half-edge to attach to (rightmost edge of the seed face)."""
    threshold = _EPS if seed_face.order() != 3 else _EPS  # kept symmetric to the notebook logic
    try:
        h = next(
            h.rev
            for h in seed_face.halfedge_iter()
            if h.rev.on_border() and max(h.orig["pos"][0], h.dest["pos"][0]) < threshold
        )
    except StopIteration as e:
        raise RuntimeError("Seed face has no border edge in the negative-x half-plane") from e

    h0 = h
    while max(h.orig["pos"][0], h.dest["pos"][0]) < _EPS:
        h = h.nex
        if h is h0:
            raise RuntimeError("No border edge found in the positive quadrant during seed walk")
    return h


def polygon_placement(code: str) -> EuclideanPositionHEG:
    """Parse the first stage of a GJH code (polygons separated by ``-`` and ``,``) into a graph.

    Args:
        code: First stage of a GJH code, e.g. ``"6"``, ``"6-3-3"``, ``"4-3-0,4"``.
            A ``0`` in a phase means "skip this attachment slot".

    Returns:
        A small Euclidean half-edge graph containing all placed polygons.
    """
    code = code.replace(" ", "")
    phases = [[int(n) for n in c.split(",")] for c in code.split("-")]

    if len(phases[0]) != 1:
        raise ValueError(f"Seed phase must consist of one polygon; got {phases[0]}")
    G = seed_polygon(phases[0][0])
    seed_face = next(iter(G.faces))

    for phase in phases[1:]:
        # Tag each existing border half-edge so we can later restrict attachment
        # to edges added in the most recent phase only.
        for h in (h for h in G.halfedges if h.on_border()):
            h["old"] = h.attributes.get("old", 0) + 1

        attach_at_list = [_starting_border(G, seed_face)]
        while True:
            attach_at_list.append(attach_at_list[-1].nex)
            if attach_at_list[-1] is attach_at_list[0]:
                break
        attach_at_list = attach_at_list[:-1]
        attach_at_list = [h for h in attach_at_list if h.attributes.get("old", 0) <= 1]

        polys = [seed_polygon(n) if n > 0 else None for n in phase]
        i = 0
        for poly in polys:
            try:
                while not (attach_at_list[i].on_border() and attach_at_list[i] in G.halfedges):
                    i += 1
                attach_at = attach_at_list[i]
            except IndexError as e:
                raise IndexError(
                    f"Not enough new edges to attach polygons {phase} "
                    f"(only {len(attach_at_list)} attachment points available)"
                ) from e
            i += 1
            if poly is None:
                continue
            G.glue_graph_e2e(poly, attach_at, next(h for h in poly.halfedges if h.on_border()))
    return G


# --- Stage 2+: affine transforms --------------------------------------------


def _unit_vector_from_y(alpha: float) -> np.ndarray:
    """Return a 2D unit vector at angle ``alpha`` measured *clockwise from the +y axis*.

    Distinct from :func:`pleat.base.unit_vector`, which measures counter-clockwise
    from the +x axis. The clockwise-from-+y convention matches the angle ordering
    used by :func:`_order_points` and the GJH transform code (e.g. ``"m30"`` mirrors
    across a line 30° clockwise from north).
    """
    return np.stack([np.sin(alpha), np.cos(alpha)])


def _translation_mat(t: np.ndarray) -> np.ndarray:
    m = np.eye(3)
    m[:2, 2] = t
    return m


def _rotation_mat(alpha: float) -> np.ndarray:
    s, c = np.sin(alpha), np.cos(alpha)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _mirror_mat_line(line: np.ndarray) -> np.ndarray:
    t1 = _translation_mat(-line[0])
    t2 = _translation_mat(line[0])
    angle = angle_to_axis(line[1] - line[0])
    r1 = _rotation_mat(-angle)
    r2 = _rotation_mat(angle)
    mx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    return t2 @ r2 @ mx @ r1 @ t1


def _mirror_mat_point(point: np.ndarray) -> np.ndarray:
    return _mirror_mat_line(np.stack([point, point + np.array([point[1], -point[0]])]))


def _rotation_mat_point(point: np.ndarray, angle: float = np.pi) -> np.ndarray:
    return _translation_mat(point) @ _rotation_mat(angle) @ _translation_mat(-point)


def _apply_affine(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.moveaxis(
        m @ np.moveaxis(np.concatenate([v, np.ones_like(v[..., :1])], axis=-1), -1, -2),
        -1,
        -2,
    )[..., :2]


def _order_points(points: np.ndarray) -> np.ndarray:
    """Return an index that sorts points clockwise from the positive y-axis, ties by distance."""
    eps_angle = np.pi / 1800
    angles = (-angle_to_axis(points) + np.pi / 2 + 2 * np.pi + eps_angle) % (2 * np.pi) - eps_angle
    norms = np.linalg.norm(points, axis=-1)
    is_origin = norms < 0.1
    normed = points / np.clip(norms[:, None], a_min=1e-6, a_max=np.inf)
    angle_groups = group_closeby(normed, eps=eps_angle * 2 * np.pi)
    _, index, inverse = np.unique(angle_groups, return_index=True, return_inverse=True)
    angles = angles[index[inverse]]
    return np.lexsort((norms, angles, is_origin))


def _remove_duplicates(G: EuclideanPositionHEG, eps: float = 1e-6, exclude_edges=()) -> EuclideanPositionHEG:
    """Merge near-coincident vertices and return a clean EuclideanPositionHEG."""
    vs = list(G.vertices)
    pos = np.stack([v["pos"] for v in vs])

    groups = group_closeby(pos, eps)
    _, index, inverse = np.unique(groups, return_index=True, return_inverse=True)
    node_mapping = {i: j for i, j in enumerate(index[inverse])}
    v_index = {v: node_mapping[i] for i, v in enumerate(vs)}

    nxG = nx.Graph()
    nxG.add_nodes_from(v_index.values())
    positions = {i: pos[i] for i in node_mapping.values()}
    excluded = set(exclude_edges).union({e.rev for e in exclude_edges})
    nxG.add_edges_from(
        [(v_index[e.orig], v_index[e.dest]) for e in G.halfedges_representing_edges() if e not in excluded]
    )

    G2 = EHEG_from_nx(nxG, positions)
    G2.recompute_lengths_and_angles()
    return G2


def apply_transform(G: EuclideanPositionHEG, code: str) -> list[np.ndarray]:
    """Parse a stage-2+ transform code (``"m30"``, ``"r(h2)"``, ``"r(c3)"``, ``"m"``, ``"r"``) into matrices.

    Args:
        G: The current graph; used to look up face centers, vertices, and edges
            referenced by ``c<i>`` / ``v<i>`` / ``h<i>`` origin specifiers.
        code: One transform stage. Possible forms:

            * ``m<deg>`` or ``r<deg>``: mirror or rotate at the origin by an angle in degrees
              (with subsequent doublings filled in automatically).
            * ``m`` / ``r`` (no angle): equivalent to ``m180`` / ``r180``.
            * ``m(<origin>)`` or ``r(<origin>)``: mirror/rotate about an origin specifier,
              where ``<origin>`` is ``c<i>`` (i-th face center), ``v<i>`` (i-th vertex),
              or ``h<i>`` (i-th edge midpoint / edge line).

    Returns:
        A list of one or more 3x3 affine matrices (homogeneous 2D transforms).
    """
    parts = code.split("(")
    mode = parts[0][0]
    if mode not in ("r", "m"):
        raise ValueError(f"Transform type must be 'r' or 'm'; got {mode!r}")
    angle_str = parts[0][1:]
    angle = np.pi / 180 * int(angle_str) if angle_str else None

    if len(parts) == 1:
        angle = np.pi if angle is None else angle
        if angle <= 0:
            raise ValueError(f"Invalid transform angle: {angle}")
        angles = [angle]
        while 2 * angles[-1] < 2 * np.pi:
            angles.append(angles[-1] * 2)
        if mode == "m":
            return [_mirror_mat_line(np.stack([np.zeros(2), _unit_vector_from_y(a)])) for a in angles]
        return [_rotation_mat(a) for a in angles]

    if angle is not None:
        raise ValueError(f"Specify either an angle or an origin, not both: {code!r}")
    origin_code = parts[1]
    if not origin_code.endswith(")"):
        raise ValueError(f"Unterminated origin specifier: {code!r}")
    origin_code = origin_code[:-1]
    origin_type, index_str = origin_code[0], origin_code[1:]
    idx = int(index_str) - 1

    if origin_type == "c":
        points = np.stack([f.midpoint() for f in G.faces])
        point = points[_order_points(points)[idx]]
        return [_mirror_mat_point(point) if mode == "m" else _rotation_mat_point(point)]
    if origin_type == "v":
        points = np.stack([v["pos"] for v in G.vertices])
        point = points[_order_points(points)[idx]]
        return [_mirror_mat_point(point) if mode == "m" else _rotation_mat_point(point)]
    if origin_type == "h":
        sides = np.stack([np.stack([h.orig["pos"], h.dest["pos"]]) for h in G.halfedges_representing_edges()])
        points = sides.mean(1)
        order = _order_points(points)
        if mode == "m":
            return [_mirror_mat_line(sides[order[idx]])]
        return [_rotation_mat_point(points[order[idx]])]
    raise ValueError(f"Origin type must be 'c', 'v', or 'h'; got {origin_type!r}")


# --- Tile tracking & expansion ----------------------------------------------


class _Tile:
    """Lightweight polygon-by-vertex-positions container used during expansion."""

    __slots__ = ("positions",)

    def __init__(self, positions: np.ndarray) -> None:
        self.positions = np.asarray(positions, dtype=np.float64)

    @classmethod
    def from_face(cls, f: Face) -> "_Tile":
        return cls(positions=np.array([v["pos"] for v in f.vertex_iter()]))

    def center(self) -> np.ndarray:
        return self.positions.mean(0)

    def copy(self) -> "_Tile":
        return _Tile(self.positions.copy())

    def transform(self, mat: np.ndarray) -> "_Tile":
        self.positions = _apply_affine(mat, self.positions)
        return self


def _polygon_graph(positions: np.ndarray) -> EuclideanPositionHEG:
    from pleat.half import CyclicHalfedgeGraph

    vs = [Vertex() for _ in range(len(positions))]
    G = CyclicHalfedgeGraph(vs)
    for v, p in zip(vs, positions):
        v["pos"] = p
    return EuclideanPositionHEG(other=G)


def _tiles_to_graph(tiles: list[_Tile]) -> EuclideanPositionHEG:
    Gs = [_polygon_graph(t.positions if signed_area(t.positions) > 0 else t.positions[::-1]) for t in tiles]
    G = Gs[0]
    for G2 in Gs[1:]:
        G.add_graph(G2)
    return _remove_duplicates(G, eps=1e-1)


def _add_transformed_tiles(
    tiles: list[_Tile],
    mat: np.ndarray,
    center_filter=None,
) -> list[_Tile]:
    center_filter = (lambda c: True) if center_filter is None else center_filter
    centers = np.stack([t.center() for t in tiles])
    transformed = _apply_affine(mat, centers)
    all_centers = np.concatenate([centers, transformed])
    groups = group_closeby(all_centers, 0.01)
    _, index, inverse = np.unique(groups, return_index=True, return_inverse=True)
    new_ind = index[inverse]
    new_ind = new_ind[new_ind >= len(tiles)] - len(tiles)
    return tiles + [tiles[i].copy().transform(mat) for i in new_ind if center_filter(transformed[i])]


def compile_gjh_graph(code: str, bbox_size: float = 20.0) -> EuclideanPositionHEG:
    """Compile a full GJH code into a finite tiled :class:`EuclideanPositionHEG`.

    Args:
        code: A full GJH code, e.g. ``"6-3-3/r60/r(h5)"``.
        bbox_size: Side length of the square bounding box (centred at the origin)
            within which to expand the tiling. Larger values produce more tiles.

    Returns:
        The expanded tiling as a Euclidean half-edge graph.
    """
    code = code.replace(" ", "")
    stages = code.split("/")
    G = polygon_placement(stages[0])
    tiles = [_Tile.from_face(f) for f in G.faces]

    mats: list[np.ndarray] = []
    for stage in stages[1:]:
        ms = apply_transform(G, stage)
        for m in ms:
            tiles = _add_transformed_tiles(tiles, m)
        mats.extend(ms)
        if len(tiles) > len(G.faces):
            G = _tiles_to_graph(tiles)

    _MAX_EXPANSION_ITERS = 1000
    for i in itertools.count():
        n_before = len(tiles)
        for m in mats:
            tiles = _add_transformed_tiles(tiles, m, center_filter=lambda c: np.max(np.abs(c)) < bbox_size / 2)
        if len(tiles) == n_before:
            break
        if i >= _MAX_EXPANSION_ITERS:
            warnings.warn(
                f"compile_gjh_graph hit the {_MAX_EXPANSION_ITERS}-iteration expansion cap "
                f"for code {code!r} at bbox_size={bbox_size}; tiling may be incomplete.",
                stacklevel=2,
            )
            break

    return _tiles_to_graph(tiles)
