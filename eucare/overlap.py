import logging
import numpy as np
import networkx as nx
import os
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm
from fastcluster import linkage_vector, linkage
from scipy.cluster.hierarchy import fcluster
import pulp
import tempfile
import os
import itertools
from collections import defaultdict
from functools import cmp_to_key
from copy import copy
from numba import jit

logger = logging.getLogger(__name__)

from .utils import random_directed_set
from .half import rotate_by
from .base import orientation
from .conversions import EHEG_from_nx
from .rendering import SvgwriteRenderer, CairoRenderer
from .layout import *


def intervals_overlapping(interval1, interval2):
    """
    Checks if two intervals are overlapping

    Parameters
    ----------
    interval1 : tuple
    interval2 : tuple

    Returns
    -------
    bool

    Examples
    --------
    >>> intervals_overlapping([0, 1], [-1, 0.5])
    True
    >>> intervals_overlapping([0, 1], [2, 3])
    False
    >>> intervals_overlapping([0, 1], [1, 2])
    True

    """
    return not (interval1[0] > interval2[1]) and not (interval1[1] < interval2[0])


def get_potential_intersections(segments, epsilon=1e-12):
    # list of start and end points, with index of corresponding segment and flag whether it is a start or an end point.
    segments = np.array(segments).copy()
    segments.sort(axis=1)  # sort each segment by y coordinate
    # move all segments start point by epsilon in x and y, to also register just non-intersections
    segments[:, 0, :] -= epsilon
    assert len(segments.shape) == 3, f'{segments.shape}'
    assert segments.shape[1:] == (2, 2), f'{segments.shape}'
    x_coords = segments[:, :, 0]
    x_coords.sort(axis=1)

    # create tuples (position, index, is_start)
    points = [(s[0][0] - epsilon, i, 1) for i, s in enumerate(segments)] + [(s[1][0], i, 0) for i, s in
                                                                            enumerate(segments)]
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
def _on_segment(p, q, r):
    """Checks if q lies on segment pr. Points are assumed to be collinear."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])) and (min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


@jit(nopython=True)
def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


@jit(nopython=True)
def line_segment_intersections(s1, s2, eps=1e-12):
    """
    see https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """

    if s1.dtype is not np.float64:
        s1 = s1.astype(np.float64)
    if s2.dtype is not np.float64:
        s2 = s2.astype(np.float64)

    p1, q1 = s1[0], s1[1]
    p2, q2 = s2[0], s2[1]
    # FIXME: eps here is arbitrary; should be scaled with edge lengths
    l1, l2 = np.linalg.norm(p1 - q1), np.linalg.norm(p2 - q2)
    # In particular: if one edge length is very small, the orientation will always be almost zero
    o1 = orientation(np.stack((p1, q1, p2)), eps * l1)  # divide eps by l1, since area/base = height = distance to segment
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
            raise ValueError('lines do not intersect')
        d = (_det(p1, q1), _det(p2, q2))
        x = _det(d, xdiff) / div
        y = _det(d, ydiff) / div
        return [np.array((x, y))]
        # return [[x, y]]

    # Special Cases (or no intersection)
    result = []

    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and _on_segment(p1, p2, q1):
        result.append(p2)

    # p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and _on_segment(p1, q2, q1):
        result.append(q2)

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and _on_segment(p2, p1, q2):
        result.append(p1)

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and _on_segment(p2, q1, q2):
        result.append(q1)

    return result


def fast_group_closeby(pts, eps):
    Z = linkage_vector(pts, method='single', metric='cityblock')
    unsorted = fcluster(Z, eps, criterion='distance')
    result = []
    i = 0
    id_map = {}
    for idx in unsorted:
        if idx not in id_map:
            id_map[idx] = i
            i += 1
        result.append(id_map[idx])
    return np.array(result, dtype=np.int32)


def faster_group_closeby_nx(arr, eps):
    """
    Cluster closeby points using (approximage) single linkage clustering implemented via connected components in networkx.
    :param arr: np.ndarray of shape (N, D)
    Array cointaining positions to be clustered.
    :param eps: float
    Clustering radius, points less than epsilon apart will get the same label.
    :return: np.ndarray of shape (N)
    Array which maps each input point to its cluster label.
    """
    if len(arr) == 0:
        return np.zeros(0, dtype=np.int32)
    scaled = arr / eps / 2
    G = nx.Graph()
    for offset in itertools.product([0, 0.5], repeat=arr.shape[-1]):
        _, index, inverse_ids, counts = np.unique(np.floor(scaled + offset), axis=0, return_counts=True, return_inverse=True, return_index=True)
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

To use this for cleaning of e.g. svg-imported CPs, provide addtional method that 'cleans' the CP: 
Join all straight degree 2 vertices that 

"""


def overlap_graph(G, eps=1e-10):
    from .utils import VerboseTimer
    # TODO: first step: group points in graph
    # list of halfedges representing edges, on the border using the inside halfedge
    edges = [h.rev if h.on_border() else h for h in G.halfedges_representing_edges()]

    logger.info('number of edges: %d', len(edges))
    line_segments = np.array([[e.orig['pos'], e.dest['pos']] for e in edges])

    timer = VerboseTimer()
    # ------ get all crossings ------
    # TODO implement sweeping line algorithm https://www.geeksforgeeks.org/given-a-set-of-line-segments-find-if-any-two-segments-intersect/

    # This will be a list of the positions of crossings between line segments. This includes the original vertices.
    crossings = []
    # This will be a list mapping the index of a crossing to the pair (i, j) of indices of the corresponding edges.
    crossings_to_edges = []

    # iterate over all pairs
    potential_intersections = get_potential_intersections(line_segments, epsilon=eps)
    timer.round('potential intersections')
    for i, j in potential_intersections:
        l1, l2 = line_segments[i], line_segments[j]
        intersections = line_segment_intersections(l1, l2, eps=eps)
        if not intersections:
            continue
        crossings.extend(intersections)
        for _ in range(len(intersections)):
            crossings_to_edges.append((i, j))
    crossings = np.array(crossings)
    timer.round('crossings')

    # ------ group closeby crossings ------

    logger.info('n crossings: %d. clustering them..', len(crossings))
    clustering = group_closeby(crossings, eps)
    timer.round('clustered crossings')
    logger.info('reduced n crossings: %d', np.max(clustering) + 1)

    # first_occurences = np.argmax(np.equal(clustering[None], np.arange(np.max(clustering) + 1)[:, None]), axis=1)
    first_occurences = np.array([np.argmax(clustering == i) for i in range(max(clustering)+1)])  # memory efficient
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
    timer.round('filtered crossings')

    # ------ get crossing orders, construct nx graph ------
    logger.info('getting crossing orders..')
    # nodes = np.arange(n_filtered_crossings)
    nx_edges = []
    edge_to_ordered_ids = []
    # for each edge (=line segment), order the crossings on it from its orig to dest.
    for i in tqdm(range(len(line_segments))):
        e = edges[i]
        crossing_ids = np.array(list(edges_to_crossings[i]))
        crossing_positions = filtered_crossings[crossing_ids]
        progression_along_edge = ((crossing_positions - e.orig['pos'][None]) * (e.dest['pos'] - e.orig['pos'])[None]).sum(
            -1)
        order = np.argsort(progression_along_edge)
        ordered_ids = crossing_ids[order]
        edge_to_ordered_ids.append(ordered_ids)
        # add line segments between crossings along the edge to the nx graph
        nx_edges.append(np.stack([ordered_ids[:-1], ordered_ids[1:]], axis=-1))
    nx_edges = np.concatenate(nx_edges)
    timer.round('crossing orders')

    logger.info('constructing nx graph..')
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(nx_edges)
    nx_positions = {i: pos for i, pos in enumerate(filtered_crossings)}
    timer.round('nx graph constructed.')
    logger.info('order of nx graph: %d', nx_graph.order())
    logger.info('converting to EHEG..')

    overlap_G, v_lookup = EHEG_from_nx(nx_graph, nx_positions, return_v_lookup=True)
    overlap_G.recompute_lengths_and_angles()
    timer.round('EHEG constructed.')

    # ------ assign original vertices, edges ------
    overlap_G.check_consistency()
    logger.info('assigning original vertices and edges..')
    for v in overlap_G.vertices:
        v['original_vertices'] = set()
    for e in overlap_G.halfedges:
        e['original_edges'] = set()
        # to keep track of which faces adjacent to original edges were next to each other
        # this dict will be a map from folds (=sets {h, h.rev} for h in the original graph) aligned with a new halfedge
        # to the faces corresponding to that fold on that side of the halfedge.
        # So the values can have two or one elements, respectively if the fold is a true (180°) fold, or lying flat.
        e['original_face_groups'] = dict()
    for f in overlap_G.faces:
        f['original_faces'] = set()

    # get face orientations
    face_orientations = {}
    for f in G.faces:
        face_orientations[f] = f.area() > 0

    for i in tqdm(range(len(edges)), desc='assigning original vertices, edges, faces'):
        ordered_ids_0 = edge_to_ordered_ids[i]
        if len(ordered_ids_0) == 0:
            raise RuntimeError(f'orig and dest of every edge should be crossings, but edge {edges[i]} has no crossings')
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
                if idx in v_lookup and 'original_vertices' in v_lookup[idx]:
                    v_lookup[idx]['original_vertices'].add(e.orig)
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
                new_edge['original_edges'].add(e)
                if not e.on_border():
                    if frozenset((e, e.rev)) not in new_edge['original_face_groups']:
                        new_edge['original_face_groups'][frozenset((e, e.rev))] = set()
                    new_edge['original_face_groups'][frozenset((e, e.rev))].add(e.face)

    # convert the face groups from a dict to a list of sets, as the information which edge the individual original faces
    # were coming from is no longer required.
    for e in overlap_G.halfedges:
        e['original_face_groups'] = [frozenset(group) for group in e['original_face_groups'].values()]

    # ------ assign original faces ------
    logger.info('assigning original faces..')
    # start on the border where there are no original faces crossing the edge (as can and does happen in the middle of
    # the folded model).
    initial_edge = next(overlap_G.border_edge_iter()).rev
    assert not initial_edge.on_border()
    frontier = [(initial_edge, {e_orig.face for e_orig in initial_edge['original_edges'] if not e_orig.on_border()})]
    yet_to_assign = set(overlap_G.faces)
    while yet_to_assign:
        current_halfedge, original_faces = frontier.pop()
        assert not current_halfedge.on_border()
        current_face = current_halfedge.face
        if current_face not in yet_to_assign:
            continue
        current_face['original_faces'] = original_faces
        yet_to_assign.remove(current_face)
        for e in current_face.halfedge_iter():
            if not e.rev.on_border() and e.rev.face in yet_to_assign:
                frontier.append((
                    e.rev,
                    (original_faces - {e_orig.face for e_orig in e['original_edges'] if not e_orig.on_border()}).union(
                        {e_orig.face for e_orig in e.rev['original_edges'] if not e_orig.on_border()})
                ))
    timer.round('complete')
    logger.info('done.')
    return overlap_G


MOUNTAIN = 1
VALLEY = -1
ORIGINAL_FACES = 'original_faces'
FACES_OF_FOLDS_ON_EDGE = 'original_face_groups'
LENGTH = 'length'
SORTED_ORIGINAL_FACES = 'sorted_original_faces'

def find_triplet_overlap_areas(G):
    result = defaultdict(float)
    bar = tqdm(G.faces, desc='finding triplet overlap areas')
    for f in bar:
        area = f.area()
        for triplet in itertools.combinations(f[ORIGINAL_FACES], 3):
            result[frozenset(triplet)] += area
        bar.set_description(f'finding triplet overlap areas ({len(result)} so far)')
    return result


def find_fold_over_facet_lengths(G):
    result = defaultdict(float)
    for e in tqdm(G.halfedges, desc='finding fold over facet lengths'):
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


def find_conincident_fold_lengths(G):
    result = defaultdict(float)
    for e in tqdm(G.halfedges, desc='finding conicident fold lengths'):
        if e.on_border():
            continue
        folds_on_edge = [group for group in e[FACES_OF_FOLDS_ON_EDGE] if len(group) == 2]
        length = e[LENGTH]
        for fold1, fold2 in itertools.combinations(folds_on_edge, 2):
            result[frozenset((frozenset(fold1), frozenset(fold2)))] += length
            #e['color_key'] = (1, 0, 0)
    return dict(result)


def cache_all(cache=None):
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


SOLVER_ORDER = [pulp.CPLEX, pulp.GLPK]


def infer_additional_over_under_pairs(over_under_pairs, facet_triplets):
    # over_dict[f] is set of all facets that f lies over.
    over_dict = defaultdict(set)
    for over, under in over_under_pairs:
        over_dict[over].add(under)

    def n_pairs():
        return sum(len(over_set) for over_set in over_dict.values())

    current = n_pairs()
    while True:
        for over, under_set in list(over_dict.items()):
            to_add = set([even_lower
                          for under in under_set
                          for even_lower in over_dict[under]
                          if frozenset((over, under, even_lower)) in facet_triplets])
            over_dict[over].update(to_add)
        if n_pairs() == current:
            break
        current = n_pairs()

    return [(over, under) for over, under_set in over_dict.items() for under in under_set]


def find_folded_face_order(G, over_under_pairs=(), solver=None, double_fold_weight=0, allow_slack=True, problem_file=None):
    if problem_file is None:  # by default, use temporary file
        with tempfile.TemporaryDirectory() as directory_name:
            filename = os.path.join(directory_name, 'Problem.lp')
            return find_folded_face_order(G, over_under_pairs, solver,
                                          double_fold_weight=double_fold_weight,
                                          allow_slack=allow_slack,
                                          problem_file=filename)
    logger.info('Processing overlap graph with %d facets..', G.order)
    triplet_overlap_areas = find_triplet_overlap_areas(G)
    fold_over_facet_lengths = find_fold_over_facet_lengths(G)
    coincident_fold_lengths = find_conincident_fold_lengths(G)
    logger.info('Found %d overlapping triplets, %d folds over facets, %d coincident folds..',
                len(triplet_overlap_areas), len(fold_over_facet_lengths), len(coincident_fold_lengths))

    n_over_under_before = len(over_under_pairs)
    over_under_pairs = infer_additional_over_under_pairs(over_under_pairs, set(triplet_overlap_areas.keys()));

    logger.info('Preprocessing increased number of known over-under relations from %d to %d..',
                n_over_under_before, len(over_under_pairs))

    n_vars = 0

    def get_varname():
        nonlocal n_vars
        n_vars += 1
        return 'x' + str(n_vars)

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
                assert constraint is True, f'{constraint}'

    for a, b, c in tqdm(triplet_overlap_areas, desc='adding triplet overlap constraints'):
        val = over(a, b) + over(b, c) + over(c, a)
        add_constraint(val <= 2)
        add_constraint(val >= 1)

    for (a, b), c in tqdm(fold_over_facet_lengths, desc='adding fold over facet constraints'):
        add_constraint(over(a, c) + over(c, b) == 1)

    for (a, b), (c, d) in tqdm(coincident_fold_lengths, desc='adding coincident fold constraints'):
        # use auxilliary variable to check if expression is even
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
        logger.info('Writing problem to file %s ..', problem_file)
        prob.writeLP(problem_file)
        if solver is None:
            for solver_class in SOLVER_ORDER:
                solver = solver_class()
                if solver.available():
                    break
            if solver is None:
                solver = 'auto'

        logger.info('Solving ILP with solver %s..', solver.__class__.__name__ if isinstance(solver, pulp.LpSolver) else solver)
        prob.solve(solver=solver)
    else:
        logger.info('Skipping ILP since everything is already determined..')

    def comparison_func(a, b):
        result = over_dict.get((a, b), 1 - over_dict.get((b, a), 0.5))
        if isinstance(result, (pulp.LpVariable, pulp.LpAffineExpression)):
            result = result.value()
        if result is None:
            result = 0
        result = int(1 - 2 * result)
        if result == 0:
            logger.warning('order unclear!')
        return result

    logger.info('determining face order..')
    for f in G.faces:
        original_faces = list(f[ORIGINAL_FACES]).copy()
        original_faces.sort(key=cmp_to_key(comparison_func))
        f[SORTED_ORIGINAL_FACES] = original_faces

    logger.info('assigning creases..')
    crease_assignment = dict()
    original_faces = {f_orig for f in G.faces for f_orig in f[ORIGINAL_FACES]}
    original_edges = {e for f in original_faces for e in f.halfedge_iter()}
    for e in original_edges:
        if e.on_border() or e.rev.on_border():
            continue
        crease_assignment[e] = comparison_func(e.face, e.rev.face) * (1 if e.face['color_key'] else -1)

    return crease_assignment


"""
Temporary utility functions - Have to make up mind on how to organize / where to put them!
"""


def fold_wireframe(G, initial_face=None):
    """mirrors all faces of G in-place"""
    if initial_face is None:
        for f in G.faces:
            pos = np.array([v['pos'] for v in f.vertex_iter()])
            if np.all(np.min(pos, axis=0) <= 0) and np.all(np.max(pos, axis=0) > 0):
                initial_face = f
                break
    if initial_face is None:
        initial_face = next(iter(G.faces))
    G.twocolor_faces(initial_face=initial_face)
    for f in filter(lambda f: f['color_key'], G.faces):
        for e in f.halfedge_iter():
            e['in_angle'] *= -1
    G.recompute_positions()


CREASE_ASSIGNMENT = 'crease_assignment'


def color_creases(G):
    colors = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        -1: (0, 0, 1)
    }
    for e in G.halfedges:
        e['color_key'] = colors[e.attributes.get('crease_assignment', 0)]


def get_over_under_pairs_from_creases(G, two_coloring_key='color_key'):
    # return list of pairs (f1, f2) with f1 over f2
    # G is assumed to be two-colored
    over_under_pairs = []
    for e in G.halfedges:
        crease_type = e.attributes.get(CREASE_ASSIGNMENT, None)
        if crease_type in (MOUNTAIN, VALLEY) and not (e.on_border() or e.rev.on_border()):
            e_above = e if e.face[two_coloring_key] else e.rev
            if crease_type is MOUNTAIN:
                e_above = e_above.rev
            over_under_pairs.append([e_above.face, e_above.rev.face])
    logger.info('number of pairs: %d', len(over_under_pairs))
    return over_under_pairs


TOP = 'top_side'
BOTTOM = 'bottom_side'


def face_order_to_clean_graph(G, side=TOP, top_color=(0.5, 0.5, 0.9), bottom_color=(0.8, 0.8, 0.8)):
    view = G.copy()

    color_key_mapping = {
        True: bottom_color,
        False: top_color,
    }

    if side == TOP:
        layer = 0
    else:
        layer = -1
    for f in view.faces:
        key = f['sorted_original_faces'][layer].attributes['color_key']
        if side == BOTTOM:
            key = not key
        f['color_key'] = color_key_mapping[key]

    to_delete = [e for e in view.halfedges
                 if not (e.on_border() or e.rev.on_border())
                 and e.face['sorted_original_faces'][layer] is e.rev.face['sorted_original_faces'][layer]]
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
        if v.order() == 2 and np.allclose(h['in_angle'] - np.pi, 0):
            to_join.append(v)
    for v in to_join:
        view.join_vertex(v)

    for e in view.halfedges:
        e['color_key'] = (0, 0, 0)

    return view


def color_creases(G, colors=None, color_border=False):
    if colors is None:
        colors = {
            0: (0, 0, 0),
            1: (1, 0, 0),
            -1: (0, 0, 1)
        }
    for e in G.halfedges:
        if not color_border and (e.on_border() or e.rev.on_border()):
            e['color_key'] = colors[0]
        else:
            e['color_key'] = colors[e.attributes.get(CREASE_ASSIGNMENT, 0)]


def fold_complete(G, initial_face=None, overlap_eps=1e-6, area_eps=0):
    fold_wireframe(G, initial_face=initial_face)
    over_under_pairs = get_over_under_pairs_from_creases(G)
    result = dict()
    G_over = overlap_graph(G, overlap_eps)
    assert area_eps == 0, 'Not implemented!'
    crease_assignment = find_folded_face_order(G_over, over_under_pairs)
    # make CP
    for e in G.halfedges:
        e[CREASE_ASSIGNMENT] = crease_assignment.get(e, 0)
    color_creases(G)
    fold_wireframe(G, initial_face=initial_face)
    result['CP'] = G
    result['folded_state'] = G_over
    result['folded_view_top'] = face_order_to_clean_graph(G_over, TOP)
    result['folded_view_bottom'] = face_order_to_clean_graph(G_over, BOTTOM)
    return result


def save_results(results, path='results', render_settings=None, min_foldable_length=None, bbox=None, extra_info=None,
                 opacity=0.15):
    """
    Saves .png and .svg versions of crease pattern as well as top and bottom views of folded state in a directory.
    Additionally, a .svg of the crease pattern optimized for scoring with a cutting plotter or laser cutter is saved.
    The dimensions of the latter are adjusted such that they fit in the specified bounding box.
    :param results: dict
    Output of fold_complete.
    :param path: str
    Path to save results in. Will be overwritten if already present.
    :param render_settings: dict, optional
    Render settings for everything but the CP for cutting.
    :param min_foldable_length: float, optional
    Minimum length a fold should have. If bbox is None, the CP for cutting will be scaled such that the shortest fold
    has this length.
    :param bbox: list or tuple, optional
    Bounding box dimensions in cm. The CP for cutting will be scaled to fit into this box.
    :param extra_info: str, optional
    String containing extra information to write to append to info.txt.
    :param opacity: float, optional
    opacity of paper for folded state
    :return:
    """
    if render_settings is None:
        render_settings = dict(face_inset=0, render_vertices=False, render_faces=True, height=2048)
    render_settings = copy(render_settings)

    # optimize rotation and save for cutting
    SRG = results['CP']
    if bbox is not None:
        optimize_rotation(SRG, angle_offset=0 if bbox[0] < bbox[1] else np.pi / 2)
        scale = max(angle_to_height(SRG, 0) / bbox[0], angle_to_height(SRG, np.pi / 2) / bbox[1])
        if min_foldable_length is not None:
            assert min_edge_length(SRG, include_border=False) / scale <= min_foldable_length, \
                f'Sheet to small, would result in a crease with length of only {min_edge_length(SRG) / scale:4.2}cm.' \
                f'Specify a larger bbox or set min_foldable_length to None to ignore this.'
    else:
        optimize_rotation(SRG, angle_offset=0)
        min_foldable_length = 0.5 if min_foldable_length is None else min_foldable_length
        scale = min_edge_length(SRG, include_border=False) / min_foldable_length

    folded = results['folded_state']
    folded_angle = optimize_rotation(folded, angle_offset=0)

    if render_settings.get('line_width', 'auto') == 'auto':
        render_settings['line_width'] = CairoRenderer.auto_line_width(results['CP'])

    os.makedirs(path, exist_ok=True)
    cp_settings = render_settings.copy()
    cp_settings.update(dict(filename=os.path.join(path, 'CP'), render_faces=False))
    results['CP'].show(**cp_settings)

    folded_settings = render_settings.copy()
    folded_settings['line_width'] /= 2
    if 'folded_view_top' in results:
        rotate_graph(results['folded_view_top'], folded_angle)
        folded_settings.update(dict(filename=os.path.join(path, 'top')))
        results['folded_view_top'].show(**folded_settings)
    if 'folded_view_bottom' in results:
        rotate_graph(results['folded_view_bottom'], folded_angle)
        folded_settings.update(dict(filename=os.path.join(path, 'bottom')))
        results['folded_view_bottom'].show(**folded_settings)

    backlight_settings = copy(render_settings)
    backlight_settings['render_edges'] = False
    backlight_settings['render_faces'] = True
    backlight_settings['line_width'] = 0
    backlight_settings['filename'] = os.path.join(path, 'backlit')
    for f in results['folded_state'].faces:
        if 'original_faces' in f:
            f['color_key'] = [0, 0, 0, 1 - (1-opacity) ** (len(f['original_faces']))]
        else:
            f['color_key'] = [0, 0, 0, opacity]
    results['folded_state'].show(**backlight_settings)

    sheet_height = angle_to_height(SRG, np.pi / 2) / scale

    plotter = SvgwriteRenderer()
    plotter.render_graph(os.path.join(path, 'cp_for_cutting.svg'), SRG, height=sheet_height)

    text = f'{len([h for h in SRG.halfedges if not (h.on_border() or h.rev.on_border())])//2} creases, ' \
           f'{len([h for h in SRG.halfedges if h.on_border()])} edges on border, ' \
           f'{len(SRG.faces)} faces.' \
           f'\nThe shortest fold has a length of {min_edge_length(SRG, include_border=False) / scale:4.2f}cm ' \
           f'({min_edge_length(SRG, include_border=True) / scale:4.2f}cm including the border).' \
           f'\nThe crease pattern fits in a box of {angle_to_height(SRG, np.pi/2)/scale:4.2f}cm x ' \
           f'{angle_to_height(SRG, 0)/scale:4.2f}cm.' \
           f'\nThe folded model fits in a box of {angle_to_height(folded, np.pi/2)/scale:4.2f}cm x ' \
           f'{angle_to_height(folded, 0)/scale:4.2f}cm.'
    if extra_info:
        text += '\n' + extra_info
    with open(os.path.join(path, 'info.txt'), 'w') as f:
        f.write(text)

    logger.info(text)


def remove_duplicates(G, eps=1e-6, exclude_edges=()):
    vs = list(G.vertices)
    pos = np.stack([v['pos'] for v in vs])

    groups = group_closeby(pos, eps)
    _, index, inverse = np.unique(groups, return_index=True, return_inverse=True)
    node_mapping = {i:j for i,j in enumerate(index[inverse])}
    v_index = {v: node_mapping[i] for i, v in enumerate(vs)}

    nxG = nx.Graph()
    nxG.add_nodes_from(v_index.values())
    nx_positions = {i: pos[i] for i in node_mapping.values()}
    nxG.add_edges_from([(v_index[e.orig], v_index[e.dest]) for e in G.halfedges_representing_edges()
                        if e not in set(exclude_edges).union({e.rev for e in exclude_edges})])

    G2 = EHEG_from_nx(nxG, nx_positions)
    G2.recompute_lengths_and_angles()
    return G2
