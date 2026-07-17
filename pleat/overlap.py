"""Compute folded states of crease patterns: overlap graphs, stacking order, and crease assignments."""

from __future__ import annotations

import itertools
import logging
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from copy import copy
from functools import cmp_to_key
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pulp
from fastcluster import linkage_vector
from numba import jit
from scipy.cluster.hierarchy import fcluster
from tqdm.auto import tqdm

from .base import orientation
from .conversions import EHEG_from_nx
from .layout import angle_to_height, min_edge_length, optimize_rotation, rotate_graph
from .rendering import BORDER_COLOR, MOUNTAIN_COLOR, VALLEY_COLOR, CairoRenderer, SvgwriteRenderer, multi_show

if TYPE_CHECKING:
    from .half import EuclideanPositionHEG, Face

logger = logging.getLogger(__name__)


def intervals_overlapping(interval1: tuple[float, float], interval2: tuple[float, float]) -> bool:
    """Check if two intervals overlap (inclusive of endpoints)."""
    return not (interval1[0] > interval2[1]) and not (interval1[1] < interval2[0])


def get_potential_intersections(segments: np.ndarray, epsilon: float = 1e-12) -> list[tuple[int, int]]:
    """Return pairs of segment indices that may intersect, using a sweep-line approach on bounding boxes."""
    # list of start and end points, with index of corresponding segment and flag whether it is a start or an end point.
    segments = np.array(segments).copy()
    segments.sort(axis=1)  # sort each segment by y coordinate
    # move all segments start point by epsilon in x and y, to also register just non-intersections
    segments[:, 0, :] -= epsilon
    assert len(segments.shape) == 3, f"{segments.shape}"
    assert segments.shape[1:] == (2, 2), f"{segments.shape}"
    x_coords = segments[:, :, 0]
    x_coords.sort(axis=1)

    # create tuples (position, index, is_start)
    points = [(s[0][0] - epsilon, i, 1) for i, s in enumerate(segments)] + [
        (s[1][0], i, 0) for i, s in enumerate(segments)
    ]
    points = np.array(points, dtype=tuple)

    # sort by first coordinate
    points = points[np.argsort(points[:, 0])]
    active_labels = set()
    possibly_intersecting = list()
    for i, is_start in points[:, 1:]:
        if is_start:
            for j in active_labels:
                if intervals_overlapping(segments[i, :, 1], segments[j, :, 1]):
                    possibly_intersecting.append((i, j))
            active_labels.add(i)
        else:
            active_labels.remove(i)
    return possibly_intersecting


@jit(nopython=True)
def _on_segment(p, q, r):  # pragma: no cover - numba-jitted, invisible to coverage
    """Check if q lies on segment pr, assuming the three points are collinear."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])) and (min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


@jit(nopython=True)
def _det(a, b):  # pragma: no cover - numba-jitted, invisible to coverage
    """Compute the 2x2 determinant of vectors a and b."""
    return a[0] * b[1] - a[1] * b[0]


@jit(nopython=True)
def line_segment_intersections(s1, s2, eps=1e-12):  # pragma: no cover - numba-jitted, invisible to coverage
    """Return a list of intersection points between two line segments, handling collinear cases."""

    if s1.dtype is not np.float64:
        s1 = s1.astype(np.float64)
    if s2.dtype is not np.float64:
        s2 = s2.astype(np.float64)

    p1, q1 = s1[0], s1[1]
    p2, q2 = s2[0], s2[1]
    # FIXME: eps here is arbitrary; should be scaled with edge lengths
    l1, l2 = np.linalg.norm(p1 - q1), np.linalg.norm(p2 - q2)
    # In particular: if one edge length is very small, the orientation will always be almost zero
    o1 = orientation(
        np.stack((p1, q1, p2)), eps * l1
    )  # divide eps by l1, since area/base = height = distance to segment
    o2 = orientation(np.stack((p1, q1, q2)), eps * l1)
    o3 = orientation(np.stack((p2, q2, p1)), eps * l2)
    o4 = orientation(np.stack((p2, q2, q1)), eps * l2)

    # General case
    # result = []
    if o1 != o2 and o3 != o4:
        xdiff = (p1[0] - q1[0], p2[0] - q2[0])
        ydiff = (p1[1] - q1[1], p2[1] - q2[1])
        div = _det(xdiff, ydiff)
        if div == 0:
            # return [] # TODO: investigate when this happens..
            raise ValueError("lines do not intersect")
        d = (_det(p1, q1), _det(p2, q2))
        x = _det(d, xdiff) / div
        y = _det(d, ydiff) / div
        return [np.array((x, y))]
        # return [[x, y]]

    # Special Cases (or no intersection)
    result = []

    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and _on_segment(p1, p2, q1):
        result.append(p2)

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and _on_segment(p1, q2, q1):
        result.append(q2)

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and _on_segment(p2, p1, q2):
        result.append(p1)

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and _on_segment(p2, q1, q2):
        result.append(q1)

    return result


def fast_group_closeby(pts: np.ndarray, eps: float) -> np.ndarray:
    """Cluster nearby points using single-linkage clustering on cityblock distance."""
    Z = linkage_vector(pts, method="single", metric="cityblock")
    unsorted = fcluster(Z, eps, criterion="distance")
    result = []
    i = 0
    id_map = {}
    for idx in unsorted:
        if idx not in id_map:
            id_map[idx] = i
            i += 1
        result.append(id_map[idx])
    return np.array(result, dtype=np.int32)


def faster_group_closeby_nx(arr: np.ndarray, eps: float) -> np.ndarray:
    """Cluster nearby points via approximate single-linkage using grid-based connected components.

    Points less than eps apart receive the same label. Return an array mapping
    each input point to its cluster label.
    """
    if len(arr) == 0:
        return np.zeros(0, dtype=np.int32)
    scaled = arr / eps / 2
    G = nx.Graph()
    for offset in itertools.product([0, 0.5], repeat=arr.shape[-1]):
        _, index, inverse_ids, counts = np.unique(
            np.floor(scaled + offset), axis=0, return_counts=True, return_inverse=True, return_index=True
        )
        G.add_edges_from([(i, index[j]) for i, j in enumerate(inverse_ids) if counts[j] > 1])
    result = -np.ones(len(arr), dtype=np.int32)
    current_id = 0
    for component in nx.connected_components(G):
        result[np.array(list(component), dtype=np.int32)] = current_id
        current_id += 1
    individual_cluster_positions = result == -1
    result[individual_cluster_positions] = current_id + np.arange(np.sum(individual_cluster_positions))
    return result


group_closeby = faster_group_closeby_nx

"""
Plan for cleaner and more efficient overlap-graph creation:

0. Start with list of N line segments, as array of shape (N, 2, 2)
1. Find identical segments (reshape to (N, 4) , cluster with group_closeby())
2. Find crossings of unique segments with sweeping line algorithm
3. Group close crossings
4. Find segments very close to crossings; Add these to the crossings (This step is new!!)
5. Again, group close segments
6. Construct nx graph

7. return: the nx graph, a mapping from the indices of original segments to the edges in the graph

To use this for cleaning of e.g. svg-imported CPs, provide additional method that 'cleans' the CP:
Join all straight degree 2 vertices that

"""


def overlap_graph(G: "EuclideanPositionHEG", eps: float = 1e-10) -> "EuclideanPositionHEG":
    """Build the overlap graph of a folded crease pattern by intersecting all edges.

    Return an EuclideanPositionHEG whose faces carry 'original_faces' attributes
    indicating which original faces overlap at each region.
    """
    from .utils import VerboseTimer

    # TODO: first step: group points in graph
    # list of halfedges representing edges, on the border using the inside halfedge
    edges = [h.rev if h.on_border() else h for h in G.halfedges_representing_edges()]

    logger.info("number of edges: %d", len(edges))
    line_segments = np.array([[e.orig["pos"], e.dest["pos"]] for e in edges])

    timer = VerboseTimer()
    # ------ get all crossings ------
    # TODO implement sweeping line algorithm https://www.geeksforgeeks.org/given-a-set-of-line-segments-find-if-any-two-segments-intersect/

    # This will be a list of the positions of crossings between line segments. This includes the original vertices.
    crossings = []
    # This will be a list mapping the index of a crossing to the pair (i, j) of indices of the corresponding edges.
    crossings_to_edges = []

    # iterate over all pairs
    potential_intersections = get_potential_intersections(line_segments, epsilon=eps)
    timer.round("potential intersections")
    for i, j in potential_intersections:
        l1, l2 = line_segments[i], line_segments[j]
        intersections = line_segment_intersections(l1, l2, eps=eps)
        if not intersections:
            continue
        crossings.extend(intersections)
        for _ in range(len(intersections)):
            crossings_to_edges.append((i, j))
    crossings = np.array(crossings)
    timer.round("crossings")

    # ------ group closeby crossings ------

    logger.info("n crossings: %d. clustering them..", len(crossings))
    clustering = group_closeby(crossings, eps)
    timer.round("clustered crossings")
    logger.info("reduced n crossings: %d", np.max(clustering) + 1)

    # first_occurences = np.argmax(np.equal(clustering[None], np.arange(np.max(clustering) + 1)[:, None]), axis=1)
    first_occurences = np.array([np.argmax(clustering == i) for i in range(max(clustering) + 1)])  # memory efficient
    # filtered_crossings consists of one exemplar per cluster
    filtered_crossings = crossings[first_occurences]
    n_filtered_crossings = len(filtered_crossings)

    # construct an array mapping crossings to all involved edges,
    # and on mapping edges to all crossings they are involved in.
    filtered_crossings_to_edges = [set() for i in range(n_filtered_crossings)]  # TODO: get rid of
    edges_to_crossings = [set() for i in range(len(edges))]
    for i, edge_ids in enumerate(crossings_to_edges):
        filtered_crossings_to_edges[clustering[i]].update(edge_ids)
        for e in edge_ids:
            edges_to_crossings[e].add(clustering[i])
    timer.round("filtered crossings")

    # ------ get crossing orders, construct nx graph ------
    logger.info("getting crossing orders..")
    # nodes = np.arange(n_filtered_crossings)
    nx_edges = []
    edge_to_ordered_ids = []
    # for each edge (=line segment), order the crossings on it from its orig to dest.
    for i in tqdm(range(len(line_segments))):
        e = edges[i]
        crossing_ids = np.array(list(edges_to_crossings[i]))
        crossing_positions = filtered_crossings[crossing_ids]
        progression_along_edge = (
            (crossing_positions - e.orig["pos"][None]) * (e.dest["pos"] - e.orig["pos"])[None]
        ).sum(-1)
        order = np.argsort(progression_along_edge)
        ordered_ids = crossing_ids[order]
        edge_to_ordered_ids.append(ordered_ids)
        # add line segments between crossings along the edge to the nx graph
        nx_edges.append(np.stack([ordered_ids[:-1], ordered_ids[1:]], axis=-1))
    nx_edges = np.concatenate(nx_edges)
    timer.round("crossing orders")

    logger.info("constructing nx graph..")
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(nx_edges)
    nx_positions = {i: pos for i, pos in enumerate(filtered_crossings)}
    timer.round("nx graph constructed.")
    logger.info("order of nx graph: %d", nx_graph.order())
    logger.info("converting to EHEG..")

    overlap_G, v_lookup = EHEG_from_nx(nx_graph, nx_positions, return_v_lookup=True)
    overlap_G.recompute_lengths_and_angles()
    timer.round("EHEG constructed.")

    # ------ assign original vertices, edges ------
    overlap_G.check_consistency()
    logger.info("assigning original vertices and edges..")
    for v in overlap_G.vertices:
        v["original_vertices"] = set()
    for e in overlap_G.halfedges:
        e["original_edges"] = set()
        # to keep track of which faces adjacent to original edges were next to each other
        # this dict will be a map from folds (=sets {h, h.rev} for h in the original graph) aligned with a new halfedge
        # to the faces corresponding to that fold on that side of the halfedge.
        # So the values can have two or one elements, respectively if the fold is a true (180°) fold, or lying flat.
        e["original_face_groups"] = dict()
    for f in overlap_G.faces:
        f["original_faces"] = set()

    # get face orientations
    face_orientations = {}
    for f in G.faces:
        face_orientations[f] = f.area() > 0

    for i in tqdm(range(len(edges)), desc="assigning original vertices, edges, faces"):
        ordered_ids_0 = edge_to_ordered_ids[i]
        if len(ordered_ids_0) == 0:
            raise RuntimeError(f"orig and dest of every edge should be crossings, but edge {edges[i]} has no crossings")
        for e in (edges[i], edges[i].rev):
            ordered_ids = ordered_ids_0
            if e is edges[i].rev:
                # reverse order of crossings (=nodes in nx graph) for reversed edge
                ordered_ids = ordered_ids[::-1]
            # this for loop sets adds the orig of the original edge to the 'original_vertices' attribute
            # of the first crossing along the edge that is a vertex in the overlap graph. By doing this for every edge
            # and it's reverse, every original vertex is addressed.
            # TODO: figure out why exactly this is necessary, i.e. in which cases it can happen that the first crossing
            #  along the edge is no vertex in the overlap graph. Show an example!
            for idx in ordered_ids:
                if idx in v_lookup and "original_vertices" in v_lookup[idx]:
                    v_lookup[idx]["original_vertices"].add(e.orig)
                    break
            # to assign original edges and face_groups, revert the order of the edge if the face is oriented negatively.
            if not e.on_border() and not face_orientations[e.face]:
                ordered_ids = ordered_ids[::-1]
            # iterate along segments between crossings on the original edge
            for k, l in zip(ordered_ids[:-1], ordered_ids[1:]):
                if k not in v_lookup or l not in v_lookup:
                    continue  # this can happen if dangling edges were deleted
                # find the halfedge in the overlap graph between crossings k and l

                new_edge = next(h for h in v_lookup[k].outgoing_iter() if h.dest is v_lookup[l])
                new_edge["original_edges"].add(e)
                if not e.on_border():
                    if frozenset((e, e.rev)) not in new_edge["original_face_groups"]:
                        new_edge["original_face_groups"][frozenset((e, e.rev))] = set()
                    new_edge["original_face_groups"][frozenset((e, e.rev))].add(e.face)

    # convert the face groups from a dict to a list of sets, as the information which edge the individual original faces
    # were coming from is no longer required.
    for e in overlap_G.halfedges:
        e["original_face_groups"] = [frozenset(group) for group in e["original_face_groups"].values()]

    # ------ assign original faces ------
    logger.info("assigning original faces..")
    # start on the border where there are no original faces crossing the edge (as can and does happen in the middle of
    # the folded model).
    initial_edge = next(overlap_G.border_edge_iter()).rev
    assert not initial_edge.on_border()
    frontier = [(initial_edge, {e_orig.face for e_orig in initial_edge["original_edges"] if not e_orig.on_border()})]
    yet_to_assign = set(overlap_G.faces)
    while yet_to_assign:
        current_halfedge, original_faces = frontier.pop()
        assert not current_halfedge.on_border()
        current_face = current_halfedge.face
        if current_face not in yet_to_assign:
            continue
        current_face["original_faces"] = original_faces
        yet_to_assign.remove(current_face)
        for e in current_face.halfedge_iter():
            if not e.rev.on_border() and e.rev.face in yet_to_assign:
                frontier.append(
                    (
                        e.rev,
                        (
                            original_faces - {e_orig.face for e_orig in e["original_edges"] if not e_orig.on_border()}
                        ).union({e_orig.face for e_orig in e.rev["original_edges"] if not e_orig.on_border()}),
                    )
                )
    timer.round("complete")
    logger.info("done.")
    return overlap_G


MOUNTAIN = 1
VALLEY = -1
BORDER = 0
ORIGINAL_FACES = "original_faces"
FACES_OF_FOLDS_ON_EDGE = "original_face_groups"
LENGTH = "length"
SORTED_ORIGINAL_FACES = "sorted_original_faces"


def find_triplet_overlap_areas(G: "EuclideanPositionHEG") -> dict[frozenset, float]:
    """Compute the total overlap area for each triplet of original faces."""
    result = defaultdict(float)
    bar = tqdm(G.faces, desc="finding triplet overlap areas")
    for f in bar:
        area = f.area()
        for triplet in itertools.combinations(f[ORIGINAL_FACES], 3):
            result[frozenset(triplet)] += area
        bar.set_description(f"finding triplet overlap areas ({len(result)} so far)")
    return result


def find_fold_over_facet_lengths(G: "EuclideanPositionHEG") -> dict[tuple[frozenset, "Face"], float]:
    """Compute total edge length where a fold passes over a facet."""
    result = defaultdict(float)
    for e in tqdm(G.halfedges, desc="finding fold over facet lengths"):
        if e.on_border() or e.rev.on_border():
            continue
        over_both = e.face[ORIGINAL_FACES].intersection(e.rev.face[ORIGINAL_FACES])
        edge_length = e[LENGTH]
        for group in e[FACES_OF_FOLDS_ON_EDGE]:
            if len(group) != 2:
                continue
            for face in over_both:
                result[(frozenset(group), face)] += edge_length
    return result


def find_conincident_fold_lengths(G: "EuclideanPositionHEG") -> dict[frozenset, float]:
    """Compute total edge length where two folds coincide."""
    result = defaultdict(float)
    for e in tqdm(G.halfedges, desc="finding coincident fold lengths"):
        if e.on_border():
            continue
        folds_on_edge = [group for group in e[FACES_OF_FOLDS_ON_EDGE] if len(group) == 2]
        length = e[LENGTH]
        for fold1, fold2 in itertools.combinations(folds_on_edge, 2):
            result[frozenset((frozenset(fold1), frozenset(fold2)))] += length
            # e['color_key'] = (1, 0, 0)
    return dict(result)


def cache_all(cache=None):
    """Decorator factory that memoizes all calls in a shared cache dict."""
    cache = dict() if cache is None else cache

    def wrapper(func):
        def wrapped(*args):
            if args in cache:
                return cache[args]
            result = func(*args)
            cache[args] = result
            return result

        return wrapped

    return wrapper


SOLVER_ORDER = [pulp.CPLEX, pulp.GLPK, pulp.PULP_CBC_CMD]


def infer_additional_over_under_pairs(over_under_pairs: list, facet_triplets: set) -> list:
    """Transitively close over/under relations: if A over B and B over C within a triplet, infer A over C."""
    # over_dict[f] is set of all facets that f lies over.
    over_dict = defaultdict(set)
    for over, under in over_under_pairs:
        over_dict[over].add(under)

    def n_pairs():
        return sum(len(over_set) for over_set in over_dict.values())

    current = n_pairs()
    while True:
        for over, under_set in list(over_dict.items()):
            to_add = set(
                [
                    even_lower
                    for under in under_set
                    for even_lower in over_dict[under]
                    if frozenset((over, under, even_lower)) in facet_triplets
                ]
            )
            over_dict[over].update(to_add)
        if n_pairs() == current:
            break
        current = n_pairs()

    return [(over, under) for over, under_set in over_dict.items() for under in under_set]


def find_folded_face_order(
    G: "EuclideanPositionHEG",
    over_under_pairs=(),
    solver=None,
    double_fold_weight: float = 0,
    allow_slack: bool = True,
    problem_file: str | None = None,
    quiet: bool = False,
) -> dict:
    """Determine the stacking order of overlapping faces by solving an ILP.

    Assign 'sorted_original_faces' to each face of the overlap graph G and return
    a crease assignment dict mapping original half-edges to MOUNTAIN/VALLEY.
    """
    if problem_file is None:  # by default, use temporary file
        with tempfile.TemporaryDirectory() as directory_name:
            filename = os.path.join(directory_name, "Problem.lp")
            return find_folded_face_order(
                G,
                over_under_pairs,
                solver,
                double_fold_weight=double_fold_weight,
                allow_slack=allow_slack,
                problem_file=filename,
            )
    logger.info("Processing overlap graph with %d facets..", G.order)
    triplet_overlap_areas = find_triplet_overlap_areas(G)
    fold_over_facet_lengths = find_fold_over_facet_lengths(G)
    coincident_fold_lengths = find_conincident_fold_lengths(G)
    logger.info(
        "Found %d overlapping triplets, %d folds over facets, %d coincident folds..",
        len(triplet_overlap_areas),
        len(fold_over_facet_lengths),
        len(coincident_fold_lengths),
    )

    n_over_under_before = len(over_under_pairs)
    over_under_pairs = infer_additional_over_under_pairs(over_under_pairs, set(triplet_overlap_areas.keys()))

    logger.info(
        "Preprocessing increased number of known over-under relations from %d to %d..",
        n_over_under_before,
        len(over_under_pairs),
    )

    n_vars = 0

    def get_varname():
        nonlocal n_vars
        n_vars += 1
        return "x" + str(n_vars)

    over_dict = dict()

    @cache_all(over_dict)
    def over(face1, face2):
        opposite = over_dict.get((face2, face1), None)
        if opposite is not None:
            return 1 - opposite
        return pulp.LpVariable(get_varname(), 0, 1, cat=pulp.LpBinary)

    # go ahead with over-under-pairs
    for face1, face2 in over_under_pairs:
        over_dict[(face1, face2)] = 1

    # TODO: infer all possible other over/under pairs

    prob = pulp.LpProblem(problem_file, pulp.LpMinimize)
    prob += 0, ""  # empty objective function
    objective = 0

    def add_constraint(constraint):
        if isinstance(constraint, pulp.LpConstraint):
            nonlocal prob
            prob += constraint, ""
        else:
            if not allow_slack:
                assert constraint is True, f"{constraint}"

    for a, b, c in tqdm(triplet_overlap_areas, desc="adding triplet overlap constraints"):
        val = over(a, b) + over(b, c) + over(c, a)
        add_constraint(val <= 2)
        add_constraint(val >= 1)

    for (a, b), c in tqdm(fold_over_facet_lengths, desc="adding fold over facet constraints"):
        add_constraint(over(a, c) + over(c, b) == 1)

    for (a, b), (c, d) in tqdm(coincident_fold_lengths, desc="adding coincident fold constraints"):
        # use auxiliary variable to check if expression is even
        if double_fold_weight == 0:
            even_between_0_and_4 = 2 * pulp.LpVariable(get_varname(), 0, 2, pulp.LpInteger)
            add_constraint(over(c, a) + over(c, b) + over(d, a) + over(d, b) - even_between_0_and_4 == 0)
        else:
            zero_or_2 = 2 * pulp.LpVariable(get_varname(), 0, 1, pulp.LpBinary)
            zero_or_4 = 4 * pulp.LpVariable(get_varname(), 0, 1, pulp.LpBinary)
            add_constraint(over(c, a) + over(c, b) + over(d, a) + over(d, b) - zero_or_2 - zero_or_4 == 0)
            objective += double_fold_weight * zero_or_2

    if n_vars > 0:
        if double_fold_weight > 0 and objective != 0:
            prob.setObjective(objective)
        # write ILP to file and solve it
        logger.info("Writing problem to file %s ..", problem_file)
        prob.writeLP(problem_file)
        if solver is None:
            for solver_class in SOLVER_ORDER:
                candidate = solver_class(msg=False if quiet else True)
                if candidate.available():
                    solver = candidate
                    break
            if solver is None:
                solver = "auto"

        logger.info(
            "Solving ILP with solver %s..", solver.__class__.__name__ if isinstance(solver, pulp.LpSolver) else solver
        )
        # Some solvers (notably CPLEX's parallel optimizer) drop auxiliary log
        # files in CWD; run the solve from the problem file's directory so
        # they land there instead of next to the caller's notebook/script.
        problem_dir = os.path.dirname(problem_file) or "."
        prev_cwd = os.getcwd()
        os.chdir(problem_dir)
        try:
            prob.solve(solver=solver)
        finally:
            os.chdir(prev_cwd)
    else:
        logger.info("Skipping ILP since everything is already determined..")

    def comparison_func(a, b):
        result = over_dict.get((a, b), 1 - over_dict.get((b, a), 0.5))
        if isinstance(result, (pulp.LpVariable, pulp.LpAffineExpression)):
            result = result.value()
        if result is None:
            result = 0
        result = int(1 - 2 * result)
        if result == 0:
            logger.warning("order unclear!")
        return result

    logger.info("determining face order..")
    for f in G.faces:
        original_faces = list(f[ORIGINAL_FACES]).copy()
        original_faces.sort(key=cmp_to_key(comparison_func))
        f[SORTED_ORIGINAL_FACES] = original_faces

    logger.info("assigning creases..")
    crease_assignment = dict()
    original_faces = {f_orig for f in G.faces for f_orig in f[ORIGINAL_FACES]}
    original_edges = {e for f in original_faces for e in f.halfedge_iter()}
    for e in original_edges:
        if e.on_border() or e.rev.on_border():
            continue
        crease_assignment[e] = comparison_func(e.face, e.rev.face) * (1 if e.face["color_key"] else -1)

    return crease_assignment


"""
Temporary utility functions - Have to make up mind on how to organize / where to put them!
"""


def fold_wireframe(G: "EuclideanPositionHEG", initial_face: "Face | None" = None) -> "Face":
    """Fold a crease pattern in place by mirroring alternating faces about their shared edges.

    Returns the initial face used for two-coloring, which is guaranteed to be unflipped.
    """
    if initial_face is None:
        initial_face = G.central_face()
    G.twocolor_faces(initial_face=initial_face)
    for f in filter(lambda f: f["color_key"], G.faces):
        for e in f.halfedge_iter():
            e["in_angle"] *= -1
    G.recompute_positions()
    return initial_face


CREASE_ASSIGNMENT = "crease_assignment"


def get_over_under_pairs_from_creases(
    G: "EuclideanPositionHEG", two_coloring_key: str = "color_key"
) -> list[list["Face"]]:
    """Derive over/under face pairs from mountain/valley crease assignments on a two-colored graph."""
    over_under_pairs = []
    for e in G.halfedges:
        crease_type = e.attributes.get(CREASE_ASSIGNMENT, None)
        if crease_type in (MOUNTAIN, VALLEY) and not (e.on_border() or e.rev.on_border()):
            e_above = e if e.face[two_coloring_key] else e.rev
            if crease_type is MOUNTAIN:
                e_above = e_above.rev
            over_under_pairs.append([e_above.face, e_above.rev.face])
    logger.info("number of pairs: %d", len(over_under_pairs))
    return over_under_pairs


TOP = "top_side"
BOTTOM = "bottom_side"


def face_order_to_clean_graph(
    G: "EuclideanPositionHEG",
    side: str = TOP,
    top_color: tuple = (0.5, 0.5, 0.9),
    bottom_color: tuple = (0.8, 0.8, 0.8),
) -> "EuclideanPositionHEG":
    """Extract a clean renderable graph showing only the top or bottom layer of a folded model."""
    view = G.copy()

    color_key_mapping = {
        True: bottom_color,
        False: top_color,
    }

    if side == BOTTOM:
        layer = 0
    else:
        layer = -1
    for f in view.faces:
        key = f["sorted_original_faces"][layer].attributes["color_key"]
        if side == TOP:
            key = not key
        f["color_key"] = color_key_mapping[key]

    to_delete = [
        e
        for e in view.halfedges
        if not (e.on_border() or e.rev.on_border())
        and e.face["sorted_original_faces"][layer] is e.rev.face["sorted_original_faces"][layer]
    ]
    view.delete_subset(to_delete)
    view.recompute_lengths_and_angles()

    # join unnecessary vertices
    to_join = []
    for v in view.vertices:
        if v.order() != 2:
            continue
        h = v.any_outgoing.pre
        if h.on_border():
            h = h.rev.pre
        assert not h.on_border()
        if v.order() == 2 and np.allclose(h["in_angle"] - np.pi, 0):
            to_join.append(v)
    for v in to_join:
        view.join_vertex(v)

    for e in view.halfedges:
        e["color_key"] = (0, 0, 0)

    return view


def color_creases(G: "EuclideanPositionHEG", colors: dict | None = None, color_border: bool = False) -> None:
    """Assign edge colors based on crease assignments (mountain=red, valley=blue, flat=black)."""
    if colors is None:
        colors = {0: BORDER_COLOR, 1: MOUNTAIN_COLOR, -1: VALLEY_COLOR}
    for e in G.halfedges:
        if not color_border and (e.on_border() or e.rev.on_border()):
            e["color_key"] = colors[0]
        else:
            e["color_key"] = colors[e.attributes.get(CREASE_ASSIGNMENT, 0)]


def fold_complete(
    G: "EuclideanPositionHEG",
    initial_face: "Face | None" = None,
    overlap_eps: float = 1e-6,
    area_eps: float = 0,
    quiet: bool = False,
) -> "FoldResult":
    """Fold a crease pattern and compute the full folded state, including overlap and stacking order.

    Args:
        G: The crease pattern graph (mutated in place).
        initial_face: Face to keep fixed during folding; auto-picked if ``None``.
        overlap_eps: Distance tolerance used by :func:`overlap_graph`.
        area_eps: Reserved; must be 0.
        quiet: If True, suppress all progress bars and log output produced
            during the pipeline.

    Returns a :class:`FoldResult` with attributes/keys ``'CP'``,
    ``'folded_state'``, ``'folded_view_top'``, and ``'folded_view_bottom'``.
    """
    with _quiet_progress(quiet):
        initial_face = fold_wireframe(G, initial_face=initial_face)
        over_under_pairs = get_over_under_pairs_from_creases(G)
        G_over = overlap_graph(G, overlap_eps)
        assert area_eps == 0, "Not implemented!"
        crease_assignment = find_folded_face_order(G_over, over_under_pairs, quiet=quiet)
        # make CP
        for e in G.halfedges:
            e[CREASE_ASSIGNMENT] = crease_assignment.get(e, 0)
        color_creases(G)
        fold_wireframe(G, initial_face=initial_face)
        return FoldResult(
            CP=G,
            folded_state=G_over,
            folded_view_top=face_order_to_clean_graph(G_over, BOTTOM),
            folded_view_bottom=face_order_to_clean_graph(G_over, TOP),
        )


@contextmanager
def _quiet_progress(quiet: bool):
    """Context manager: while ``quiet`` is True, silence this module's tqdm bars and logger."""
    if not quiet:
        yield
        return

    module = sys.modules[__name__]
    original_tqdm = module.tqdm

    def silent_tqdm(iterable=None, *args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(iterable, *args, **kwargs)

    module.tqdm = silent_tqdm
    previous_level = logger.level
    previous_disabled = logger.disabled
    logger.setLevel(logging.CRITICAL + 1)
    logger.disabled = True
    try:
        yield
    finally:
        module.tqdm = original_tqdm
        logger.setLevel(previous_level)
        logger.disabled = previous_disabled


class FoldResult(dict):
    """Container for the output of :func:`fold_complete`.

    Behaves like a dict for backwards compatibility (``result['CP']``) and
    additionally exposes its entries as attributes (``result.CP``).  Any of
    the slots may be ``None`` / missing.
    """

    _FIELDS = ("CP", "folded_state", "folded_view_top", "folded_view_bottom", "CP_for_origami_simulator")

    def __init__(
        self,
        CP: "EuclideanPositionHEG | None" = None,
        folded_state: "EuclideanPositionHEG | None" = None,
        folded_view_top: "EuclideanPositionHEG | None" = None,
        folded_view_bottom: "EuclideanPositionHEG | None" = None,
        CP_for_origami_simulator: "EuclideanPositionHEG | None" = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("CP", CP),
            ("folded_state", folded_state),
            ("folded_view_top", folded_view_top),
            ("folded_view_bottom", folded_view_bottom),
            ("CP_for_origami_simulator", CP_for_origami_simulator),
        ):
            if value is not None:
                self[name] = value

    def __getattr__(self, name: str):
        if name in self._FIELDS:
            return self.get(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._FIELDS:
            if value is None:
                self.pop(name, None)
            else:
                self[name] = value
        else:
            super().__setattr__(name, value)

    def show(
        self,
        render_settings: dict | None = None,
        opacity: float = 0.15,
        ncols: int | None = 2,
        suptitle: str | None = None,
        cell_size: float = 4.0,
    ) -> None:
        """Display the available result graphs side-by-side via :func:`multi_show`.

        Uses the same per-panel render settings as :func:`save_results`:
        the CP is drawn without face fills, the top/bottom views with
        half-thickness edges, and the folded state as a back-lit composite
        with face opacity stacking.

        Args:
            render_settings: Base ``G.show`` kwargs; defaults to
                ``dict(face_inset=0, render_vertices=False, render_faces=True, height=512)``.
            opacity: Per-layer alpha used by the back-lit folded-state panel.
            ncols: Number of columns in the matplotlib grid.
            suptitle: Optional figure-level title.
            cell_size: Per-cell size in matplotlib inches.
        """
        if render_settings is None:
            render_settings = dict(face_inset=0, render_vertices=False, render_faces=True, height=512)
        render_settings = copy(render_settings)

        # Establish a consistent line width across the panels so the halved
        # value used for top/bottom views matches save_results.
        if render_settings.get("line_width", "auto") == "auto" and self.CP is not None:
            render_settings["line_width"] = CairoRenderer.auto_line_width(self.CP)

        graphs: list = []
        titles: list[str] = []
        per_subplot: list[dict] = []

        if self.CP is not None:
            graphs.append(self.CP)
            titles.append("CP")
            per_subplot.append(dict(render_faces=False))

        if self.folded_state is not None:
            # Back-lit composite: faces only, with stacking-derived opacity.
            for f in self.folded_state.faces:
                if "original_faces" in f:
                    f["color_key"] = [0, 0, 0, 1 - (1 - opacity) ** (len(f["original_faces"]))]
                else:
                    f["color_key"] = [0, 0, 0, opacity]
            graphs.append(self.folded_state)
            titles.append("folded (backlit)")
            per_subplot.append(dict(render_edges=False, render_faces=True, line_width=0))

        if self.folded_view_top is not None:
            graphs.append(self.folded_view_top)
            titles.append("folded (top)")
            per_subplot.append(dict(line_width=render_settings.get("line_width", "auto")))
            if isinstance(per_subplot[-1]["line_width"], (int, float)):
                per_subplot[-1]["line_width"] = per_subplot[-1]["line_width"] / 2

        if self.folded_view_bottom is not None:
            graphs.append(self.folded_view_bottom)
            titles.append("folded (bottom)")
            per_subplot.append(dict(line_width=render_settings.get("line_width", "auto")))
            if isinstance(per_subplot[-1]["line_width"], (int, float)):
                per_subplot[-1]["line_width"] = per_subplot[-1]["line_width"] / 2

        if self.CP_for_origami_simulator is not None:
            graphs.append(self.CP_for_origami_simulator)
            titles.append("CP (origami simulator)")
            per_subplot.append(dict(render_faces=False))

        if not graphs:
            return

        multi_show(
            graphs,
            titles=titles,
            ncols=ncols,
            suptitle=suptitle,
            cell_size=cell_size,
            per_subplot_kwargs=per_subplot,
            **render_settings,
        )

    def save(self, path: str, **save_results_kwargs):
        """Convenience wrapper around :func:`save_results`."""
        save_results(self, path=path, **save_results_kwargs)


def save_results(
    results,
    path: str = "results",
    render_settings: dict | None = None,
    min_foldable_length: float | None = None,
    bbox: tuple[float, float] | None = None,
    extra_info: str = None,
    opacity: float = 0.15,
):
    """Save rendered crease pattern, folded views, and a plotter-ready SVG to a directory.

    The plotter SVG is scaled to fit within bbox (in cm) or so that the shortest
    fold equals min_foldable_length. Results is the output dict from fold_complete, with keys 'CP', 'folded_state', 'folded_view_top', and 'folded_view_bottom'.
    Additionally, a key 'CP_for_origami_simulator' can be provided, which is a version of the CP optimized for folding in Origami Simulator.
    """
    valid_keys = ["CP", "folded_state", "folded_view_top", "folded_view_bottom", "CP_for_origami_simulator"]
    for key in results.keys():
        if key not in valid_keys:
            raise ValueError(f"Unexpected key {key} in results dict. Expected keys are {valid_keys}.")

    if render_settings is None:
        render_settings = dict(face_inset=0, render_vertices=False, render_faces=True, height=2048)
    render_settings = copy(render_settings)

    # optimize rotation and save for cutting
    SRG = results["CP"]
    if bbox is not None:
        optimize_rotation(SRG, angle_offset=0 if bbox[0] < bbox[1] else np.pi / 2)
        scale = max(angle_to_height(SRG, 0) / bbox[0], angle_to_height(SRG, np.pi / 2) / bbox[1])
        if min_foldable_length is not None:
            assert min_edge_length(SRG, include_border=False) / scale <= min_foldable_length, (
                f"Sheet to small, would result in a crease with length of only {min_edge_length(SRG) / scale:4.2}cm."
                f"Specify a larger bbox or set min_foldable_length to None to ignore this."
            )
    else:
        optimize_rotation(SRG, angle_offset=0)
        min_foldable_length = 0.5 if min_foldable_length is None else min_foldable_length
        scale = min_edge_length(SRG, include_border=False) / min_foldable_length

    folded = results["folded_state"]
    folded_angle = optimize_rotation(folded, angle_offset=0)

    if render_settings.get("line_width", "auto") == "auto":
        render_settings["line_width"] = CairoRenderer.auto_line_width(results["CP"])

    os.makedirs(path, exist_ok=True)
    cp_settings = render_settings.copy()
    cp_settings.update(dict(render_faces=False))
    results["CP"].save(os.path.join(path, "CP"), **cp_settings)

    folded_settings = render_settings.copy()
    folded_settings["line_width"] /= 2
    if "folded_view_top" in results:
        rotate_graph(results["folded_view_top"], folded_angle)
        results["folded_view_top"].save(os.path.join(path, "top"), **folded_settings)
    if "folded_view_bottom" in results:
        rotate_graph(results["folded_view_bottom"], folded_angle)
        results["folded_view_bottom"].save(os.path.join(path, "bottom"), **folded_settings)

    backlight_settings = copy(render_settings)
    backlight_settings["render_edges"] = False
    backlight_settings["render_faces"] = True
    backlight_settings["line_width"] = 0
    for f in results["folded_state"].faces:
        if "original_faces" in f:
            f["color_key"] = [0, 0, 0, 1 - (1 - opacity) ** (len(f["original_faces"]))]
        else:
            f["color_key"] = [0, 0, 0, opacity]
    results["folded_state"].save(os.path.join(path, "backlit"), **backlight_settings)

    if "CP_for_origami_simulator" in results:
        optimize_rotation(results["CP_for_origami_simulator"], angle_offset=0)
        cp_for_simulator_settings = cp_settings.copy()
        results["CP_for_origami_simulator"].save(
            os.path.join(path, "CP_for_origami_simulator"), **cp_for_simulator_settings
        )

    sheet_height = angle_to_height(SRG, np.pi / 2) / scale

    plotter = SvgwriteRenderer()
    plotter.render_graph(os.path.join(path, "cp_for_cutting.svg"), SRG, height=sheet_height)

    text = (
        f"{len([h for h in SRG.halfedges if not (h.on_border() or h.rev.on_border())])//2} creases, "
        f"{len([h for h in SRG.halfedges if h.on_border()])} edges on border, "
        f"{len(SRG.faces)} faces."
        f"\nThe shortest fold has a length of {min_edge_length(SRG, include_border=False) / scale:4.2f}cm "
        f"({min_edge_length(SRG, include_border=True) / scale:4.2f}cm including the border)."
        f"\nThe crease pattern fits in a box of {angle_to_height(SRG, np.pi/2)/scale:4.2f}cm x "
        f"{angle_to_height(SRG, 0)/scale:4.2f}cm."
        f"\nThe folded model fits in a box of {angle_to_height(folded, np.pi/2)/scale:4.2f}cm x "
        f"{angle_to_height(folded, 0)/scale:4.2f}cm."
    )
    if extra_info:
        text += "\n" + extra_info
    with open(os.path.join(path, "info.txt"), "w") as f:
        f.write(text)

    logger.info(text)


def remove_duplicates(G: "EuclideanPositionHEG", eps: float = 1e-6, exclude_edges=()) -> "EuclideanPositionHEG":
    """Merge duplicate vertices and edges within eps distance, returning a clean graph.

    The faces of the resulting graph are built from scratch based on the new edges, so this only works for planar graphs.
    """
    vs = list(G.vertices)
    pos = np.stack([v["pos"] for v in vs])

    groups = group_closeby(pos, eps)
    _, index, inverse = np.unique(groups, return_index=True, return_inverse=True)
    node_mapping = {i: j for i, j in enumerate(index[inverse])}
    v_index = {v: node_mapping[i] for i, v in enumerate(vs)}

    nxG = nx.Graph()
    nxG.add_nodes_from(v_index.values())
    nx_positions = {i: pos[i] for i in node_mapping.values()}
    nxG.add_edges_from(
        [
            (v_index[e.orig], v_index[e.dest])
            for e in G.halfedges_representing_edges()
            if e not in set(exclude_edges).union({e.rev for e in exclude_edges})
        ]
    )

    G2 = EHEG_from_nx(nxG, nx_positions)
    G2.recompute_lengths_and_angles()
    return G2
