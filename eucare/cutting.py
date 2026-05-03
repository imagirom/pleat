"""Cut half-edge graphs along halfplane boundaries or polygon regions.

The main entry point is :func:`cut_out_poly`, which inserts new vertices and
edges where a polygon crosses an existing graph and (optionally) deletes the
faces lying outside the polygon.  Class :class:`Halfplane` (and its
:class:`CuttingRegion` base) is used by :mod:`eucare.overlap` to model
folded-face stacking constraints.

Low-level numba-accelerated helpers (:func:`pointinpolygon`,
:func:`parallelpointinpolygon`, :func:`polygon_line_segment_intersections`,
:func:`get_potential_intersections_2`, :func:`get_ordered_crossings`) sit
beneath the public API and are reused by other modules.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numba
import numpy as np
from numba import jit, njit

from eucare.half import rotate_by

if TYPE_CHECKING:
    from eucare.half import EuclideanPositionHEG, HalfEdgeGraph
from eucare.overlap import group_closeby, intervals_overlapping, line_segment_intersections
from eucare.rendering import inset_poly

# ---------------------------------------------------------------------------
# Point-in-polygon helpers (from https://stackoverflow.com/a/48760556)
# ---------------------------------------------------------------------------


@jit(nopython=True)
def pointinpolygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Numba-jitted point-in-polygon test (ray casting).

    Args:
        x: Query x coordinate.
        y: Query y coordinate.
        poly: ``(n, 2)`` float64 array of polygon vertices.

    Returns:
        True iff ``(x, y)`` lies strictly inside *poly*.
    """
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised :func:`pointinpolygon` over a batch of query points."""
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i, 0], points[i, 1], polygon)
    return D


@numba.jit(nopython=True)
def polygon_line_segment_intersections(poly: np.ndarray, line_segment: np.ndarray, eps: float = 1e-12) -> list:
    """Return the list of intersections of *line_segment* with each edge of closed polygon *poly*."""
    intersections = []
    p1 = poly[-1]
    for i in range(len(poly)):
        p2 = poly[i]
        intersections.extend(line_segment_intersections(line_segment, np.stack((p1, p2)), eps))
        p1 = p2
    return intersections


def get_potential_intersections_2(segments1: np.ndarray, segments2: np.ndarray, epsilon: float = 1e-12) -> list:
    """Bounding-box prefilter: return ``(i, j)`` pairs whose AABBs overlap.

    Used as a pre-pass before the expensive exact intersection test.  Sweep-line
    over x; for each x-active segment in one set, intersect-test against
    x-active segments in the other set whose y-intervals overlap.
    """
    if len(segments2) == 0 or len(segments1) == 0:
        return []
    ns1 = len(segments1)
    segments = np.concatenate((segments1, segments2))

    # sort each segment by y coordinate
    segments.sort(axis=1)
    # move all segments start point by epsilon in x and y, to also register just non-intersections
    segments[:, 0, :] -= epsilon
    assert len(segments.shape) == 3 and segments.shape[1:] == (2, 2), f"{segments.shape}"
    x_coords = segments[:, :, 0]
    x_coords.sort(axis=1)

    # create tuples (position, index, is_start)
    points = [(s[0][0] - epsilon, i, 1) for i, s in enumerate(segments)] + [
        (s[1][0], i, 0) for i, s in enumerate(segments)
    ]
    points = np.array(points, dtype=tuple)

    # sort by first coordinate
    points = points[np.argsort(points[:, 0])]
    active_labels_1 = set()
    active_labels_2 = set()
    possibly_intersecting = list()
    for i, is_start in points[:, 1:]:

        if i < ns1:  # segment in segments1
            if is_start:
                for j in active_labels_2:
                    if intervals_overlapping(segments[i, :, 1], segments[j, :, 1]):
                        possibly_intersecting.append((i, j - ns1))
                active_labels_1.add(i)
            else:
                active_labels_1.remove(i)
        else:  # segment in segments1
            if is_start:
                for j in active_labels_1:
                    if intervals_overlapping(segments[i, :, 1], segments[j, :, 1]):
                        possibly_intersecting.append((j, i - ns1))
                active_labels_2.add(i)
            else:
                active_labels_2.remove(i)
    return possibly_intersecting


def get_ordered_crossings(segments1: np.ndarray, segments2: np.ndarray, eps: float = 1e-10) -> tuple:
    """Compute exact intersections between two segment sets and group near-duplicates.

    Args:
        segments1: ``(n1, 2, 2)`` array of polygon-side segments.
        segments2: ``(n2, 2, 2)`` array of graph-edge segments.
        eps: Tolerance for clustering near-coincident crossings.

    Returns:
        A 5-tuple ``(crossings, segments1_to_crossings, segments2_to_crossings,
        crossings_to_segments1, crossings_to_segments2)``. ``crossings`` is a
        ``(k, 2)`` array of unique crossing positions; ``segmentsX_to_crossings[i]``
        is the ordered list of crossing indices along segment ``i`` of set X;
        ``crossings_to_segmentsX[k]`` is the set of segment indices in set X
        contributing to crossing ``k``.
    """
    crossings = []
    # This will be a list mapping the index of a crossing to the pair (i, j) of indices of the corresponding edges.
    crossings_to_edges = []

    for i, j in get_potential_intersections_2(segments1, segments2, epsilon=eps):
        l1, l2 = segments1[i], segments2[j]
        intersections = line_segment_intersections(l1, l2, eps=eps)
        if not intersections:
            continue
        crossings.extend(intersections)
        for _ in range(len(intersections)):
            crossings_to_edges.append((i, j))
    crossings = np.array(crossings)
    if len(crossings) == 0:
        return (
            np.zeros(0, dtype=np.int32),
            [],
            [],
            [[] for _ in range(len(segments1))],
            [[] for _ in range(len(segments2))],
        )

        # ------ group closeby crossings ------
    clustering = group_closeby(crossings, eps)
    first_occurences = np.argmax(clustering[None] == np.arange(np.max(clustering) + 1)[:, None], axis=1)
    # filtered_crossings consists of one exemplar per cluster
    filtered_crossings = crossings[first_occurences]
    n_filtered_crossings = len(filtered_crossings)

    # construct an array mapping crossings to all involved edges,
    # and on mapping edges to all crossings they are involved in.
    filtered_crossings_to_segments1 = [set() for i in range(n_filtered_crossings)]
    filtered_crossings_to_segments2 = [set() for i in range(n_filtered_crossings)]
    segments1_to_crossings = [set() for i in range(len(segments1))]
    segments2_to_crossings = [set() for i in range(len(segments2))]
    for i, edge_ids in enumerate(crossings_to_edges):
        i_filtered = clustering[i]
        filtered_crossings_to_segments1[i_filtered].add(edge_ids[0])
        filtered_crossings_to_segments2[i_filtered].add(edge_ids[1])
        segments1_to_crossings[edge_ids[0]].add(clustering[i])
        segments2_to_crossings[edge_ids[1]].add(clustering[i])

    # ------ get crossing orders ------
    def order_crossings(segments, segments_to_crossings):
        segments_dirs = segments[:, 1] - segments[:, 0]
        segments_dirs /= np.linalg.norm(segments_dirs, axis=1, keepdims=True)
        for i, crossing_ids in enumerate(segments_to_crossings):
            if len(crossing_ids) < 2:
                segments_to_crossings[i] = list(crossing_ids)
                continue
            crossing_ids = np.array(list(crossing_ids))
            progression_along_edge = ((filtered_crossings[crossing_ids] - segments[i, :1]) * segments_dirs[i]).sum(-1)
            segments_to_crossings[i] = list(crossing_ids[np.argsort(progression_along_edge)])

    order_crossings(segments1, segments1_to_crossings)
    order_crossings(segments2, segments2_to_crossings)

    return (
        filtered_crossings,
        segments1_to_crossings,
        segments2_to_crossings,
        filtered_crossings_to_segments1,
        filtered_crossings_to_segments2,
    )


def delete_new_outside(G: "HalfEdgeGraph", border_key: str = "new_border") -> None:
    """Flood-fill faces reachable from a ``new_border`` half-edge and delete them.

    Args:
        G: Graph to mutate in place.
        border_key: Half-edge attribute marking the new boundary; cleared after deletion.
    """
    # floodfill the outside region and delete it
    hs = [h for h in G.halfedges if border_key in h.attributes]
    to_delete = {h.face for h in hs if h.face}
    border = to_delete.copy()
    while border:
        f = border.pop()
        for h in f.halfedge_iter():
            if border_key in h.attributes:
                continue
            f2 = h.rev.face
            if f2 and f2 not in to_delete:
                to_delete.add(f2)
                border.add(f2)
    G.delete_subset(to_delete)
    for h in hs:
        assert h.on_border(), f"{h}"
        del h[border_key]


def cut_out_poly(
    G: "EuclideanPositionHEG",
    poly: np.ndarray,
    delete_outside: bool = True,
    eps: float = 1e-10,
) -> "EuclideanPositionHEG":
    """Insert vertices and edges where *poly* crosses *G*, optionally deleting outside.

    Args:
        G: Graph to cut, mutated in place.
        poly: ``(n, 2)`` array of polygon vertices in order.
        delete_outside: If True, faces outside *poly* are deleted afterwards.
        eps: Tolerance for crossing clustering.

    Returns:
        The modified graph (also mutated in place).
    """
    # numba-jit'd ``line_segment_intersections`` requires a uniform float64
    # dtype across both segment arrays; cast defensively here so that callers
    # that produced positions via float32 paths (e.g. torch optimisers) work.
    poly_segments = np.stack(list(rotate_by(poly, (0, 1)))).astype(np.float64)
    es = list(G.halfedges_representing_edges())
    edge_segments = np.array([[e.orig["pos"], e.dest["pos"]] for e in es], dtype=np.float64)
    (
        crossing_positions,
        segments1_to_crossings,
        segments2_to_crossings,
        crossings_to_segments1,
        crossings_to_segments2,
    ) = get_ordered_crossings(poly_segments, edge_segments, eps=eps)

    crossing_to_vertices = defaultdict(set)
    vertices_to_connect = set()
    crossing_specification_to_vertex = defaultdict(set)

    for edge_index, crossing_indices in enumerate(segments2_to_crossings):
        e = es[edge_index]
        for i in reversed(crossing_indices):
            crossing = crossing_positions[i]

            if np.linalg.norm(e.orig["pos"] - crossing) < eps:
                v = e.orig
            elif np.linalg.norm(e.dest["pos"] - crossing) < eps:
                v = e.dest
            else:
                _, v = G.subdivide_edge(e, pos=crossing)
            crossing_to_vertices[i].add(v)
            # label the vertex
            v["poly_side"] = {
                poly_side: np.argwhere(np.array(segments1_to_crossings[poly_side]) == i)[0, 0]
                for poly_side in crossings_to_segments1[i]
            }
            for poly_side, j in v["poly_side"].items():
                crossing_specification_to_vertex[(poly_side, j)].add(v)
            vertices_to_connect.add(v)

    # for key, val in crossing_specification_to_vertex.items():
    #     print(key, len(val))

    for v in vertices_to_connect:
        # attempt to connect it to the next vertex in line
        for poly_side, crossing_ind_along_side in v["poly_side"].items():
            corners = []
            start = (poly_side, crossing_ind_along_side)
            # print(start)
            # try to connect via positive area face:
            while True:
                crossing_ind_along_side += 1
                if crossing_ind_along_side >= len(segments1_to_crossings[poly_side]):
                    # last point on this poly side, go to next side AND add the corner(s) inbetween
                    poly_side = (poly_side + 1) % len(poly)
                    crossing_ind_along_side = 0
                    corner = poly[poly_side]
                    if not np.linalg.norm(corner - v["pos"]) < eps:
                        corners.append(corner)
                # print('.', poly_side, crossing_ind_along_side)
                if (poly_side, crossing_ind_along_side) == start:
                    # if no_connection:
                    # print('no connection found')
                    break
                try:
                    f, v2 = next(
                        (f, v2)
                        for v2 in crossing_specification_to_vertex[(poly_side, crossing_ind_along_side)]
                        for f in v.common_faces_iter(v2)
                        if f.area() > 0
                        and (
                            not corners
                            or np.all(
                                parallelpointinpolygon(
                                    np.array(corners),
                                    np.asarray(inset_poly(np.stack([vf["pos"] for vf in f.vertex_iter()]), -eps)),
                                )
                            )
                        )
                    )
                    if v2 is v:
                        # print('identical vertex')
                        break
                    corners_this = [c for c in corners if np.linalg.norm(v2["pos"] - c) > eps]
                    v_out = next(h for h in f.halfedge_iter() if h.orig is v)
                    if not corners_this and v_out.dest is v2:
                        # print('yep')
                        v_out["new_border"] = True
                    elif not corners_this and v_out.pre.orig is v2:
                        # print('yep2')
                        v_out.pre.rev["new_border"] = True
                    else:
                        # print(f'connecting {start} and {(poly_side, crossing_ind_along_side)}, {len(corners_this)}')
                        G.subdivide_face(f, v, v2)
                        # add the corners by subdividing the new edge
                        new_edge = v_out.pre.rev
                        for corner in corners_this:
                            new_edge["new_border"] = True
                            G.subdivide_edge(new_edge, pos=corner)
                            new_edge = new_edge.nex
                        new_edge["new_border"] = True

                    break
                except StopIteration as e:
                    continue

        for poly_side, crossing_ind_along_side in v["poly_side"].items():
            corners = []
            start = (poly_side, crossing_ind_along_side)
            # try to connect via negative area face:
            while True:
                crossing_ind_along_side -= 1
                if crossing_ind_along_side < 0:
                    # last point on this poly side, go to next side AND add the corner(s) in between
                    corner = poly[poly_side]
                    poly_side = (poly_side - 1) % len(poly)
                    crossing_ind_along_side = len(segments1_to_crossings[poly_side])
                    if not np.linalg.norm(corner - v["pos"]) < eps:
                        corners.append(corner)
                # print('.', poly_side, crossing_ind_along_side)
                if (poly_side, crossing_ind_along_side) == start:
                    # if no_connection:
                    # print('no negative area connection found')
                    break
                try:
                    f, v2 = next(
                        (f, v2)
                        for v2 in crossing_specification_to_vertex[(poly_side, crossing_ind_along_side)]
                        for f in v.common_faces_iter(v2)
                        if f.area() < 0
                        and (
                            not corners
                            or np.all(
                                parallelpointinpolygon(
                                    np.array(corners),
                                    np.asarray(
                                        inset_poly(
                                            np.stack([vf["pos"] for vf in reversed(list(f.vertex_iter()))]), -eps
                                        )
                                    ),
                                )
                            )
                        )
                    )
                    if v2 is v:
                        # print('identical vertex')
                        break
                    corners_this = [c for c in corners if np.linalg.norm(v2["pos"] - c) > eps]
                    v_out = next(h for h in f.halfedge_iter() if h.orig is v)
                    if not corners_this and v_out.dest is v2:
                        # print('yep')
                        v_out["new_border"] = True
                    elif not corners_this and v_out.pre.orig is v2:
                        # print('yep2')
                        v_out.pre.rev["new_border"] = True
                    else:
                        # print(f'connecting {start} and {(poly_side, crossing_ind_along_side)}, {len(corners_this)}')
                        G.subdivide_face(f, v, v2)
                        # add the corners by subdividing the new edge
                        new_edge = v_out.pre.rev
                        for corner in corners_this:
                            new_edge["new_border"] = True
                            G.subdivide_edge(new_edge, pos=corner)
                            new_edge = new_edge.nex
                        new_edge["new_border"] = True

                    break
                except StopIteration:
                    continue

    if delete_outside:
        delete_new_outside(G, border_key="new_border")

    for v in G.vertices:
        if "signed_distance" in v.attributes:
            del v["signed_distance"]

    # iterate over vertices that need to be connected

    return G


class CuttingRegion:
    """Abstract closed region used to cut graphs (see :func:`cut_out_poly`).

    Subclasses implement :meth:`inside`, :meth:`intersections`, and optionally
    :meth:`corners` (any region corners that should be inserted as vertices).
    """

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask: True where each point lies inside the region."""
        raise NotImplementedError

    def intersections(self, line_segments: np.ndarray) -> np.ndarray:
        """Compute intersection points of *line_segments* with the region boundary."""
        raise NotImplementedError

    def corners(self) -> list:
        """Region corners that should be added as new graph vertices (none by default)."""
        return []


class Halfplane(CuttingRegion):
    """Open halfplane ``{x : (x - p) . v > 0}``, used as a cutting region."""

    def __init__(self, p: np.ndarray, v: np.ndarray) -> None:
        """Construct from a point on the boundary and an outward-pointing normal.

        Args:
            p: Any point on the boundary line.
            v: Normal vector pointing into the *kept* halfplane (normalised internally).
        """
        self.p = p
        self.v = v / np.linalg.norm(v)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from *points* to the boundary (negative = inside)."""
        return -((points - self.p) * self.v).sum(-1)

    def intersections(self, line_segments: np.ndarray) -> np.ndarray:
        """Intersections of the boundary line with each segment in *line_segments*.

        Args:
            line_segments: Shape ``(n_segments, 2, 2)``: ``(start, end), (x, y)``.
        """
        mixing_coefficients = np.abs(((line_segments - self.p) * self.v).sum(-1, keepdims=True))
        mixing_coefficients /= mixing_coefficients.sum(1, keepdims=True)
        mixing_coefficients = 1 - mixing_coefficients
        return (line_segments * mixing_coefficients).sum(1)
