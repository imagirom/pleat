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

from .utils import random_directed_set
from .half import rotate_by
from .base import orientation
from .conversions import EHEG_from_nx


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
    segments.sort(axis=1)  # sort by y coordinate
    # move all segments start point by epsilon in x and y, to also register just non-intersections
    segments[:, 0, :] -= epsilon
    assert len(segments.shape) == 3, f'{segments.shape}'
    assert segments.shape[1:] == (2, 2), f'{segments.shape}'
    x_coords = segments[:, :, 0]
    x_coords.sort(axis=1)
    points = [(s[0][0] - epsilon, i, 1) for i, s in enumerate(segments)] + [(s[1][0], i, 0) for i, s in
                                                                            enumerate(segments)]
    points = np.array(points, dtype=tuple)
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


def _on_segment(p, q, r):
    """Checks if q lies on segment pr. Points are assumed to be collinear."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])) and (min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_segment_intersections(s1, s2, eps=1e-12):
    """
    see https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """

    #     if not intervals_overlapping([min(s1[:, 0]), max(s1[:, 0])], [min(s2[:, 0]), max(s2[:, 0])]):
    #         return []  # separated in x
    #     if not intervals_overlapping([min(s1[:, 1]), max(s1[:, 1])], [min(s2[:, 1]), max(s2[:, 1])]):
    #         return []  # separated in y

    p1, q1 = s1
    p2, q2 = s2
    # FIXME: eps here is arbitrary; should be scaled with edge lengths
    l1, l2 = np.linalg.norm(p1 - q1), np.linalg.norm(p2 - q2)
    # In particular: if one edge length is very small, the orientation will always be almost zero
    o1 = orientation([p1, q1, p2], eps * l1)  # divide eps by l1, since area/base = height = distance to segment
    o2 = orientation([p1, q1, q2], eps * l1)
    o3 = orientation([p2, q2, p1], eps * l2)
    o4 = orientation([p2, q2, q1], eps * l2)

    # General case
    if (o1 != o2 and o3 != o4):
        xdiff = (p1[0] - q1[0], p2[0] - q2[0])
        ydiff = (p1[1] - q1[1], p2[1] - q2[1])
        div = _det(xdiff, ydiff)
        if div == 0:
            # return [] # TODO: investigate when this happens..
            raise Exception('lines do not intersect')
        d = (_det(p1, q1), _det(p2, q2))
        x = _det(d, xdiff) / div
        y = _det(d, ydiff) / div
        return [[x, y]]

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


def poly_intersection_points(p1, p2):
    """
    Compute points of intersection bewteen two 2D polygons.

    Parameters
    ----------
    p1 : np.ndarray
    ccw points in first polygon.
    p2 : np.ndarray
    ccw points in second polygon.

    Returns
    -------
    np.ndarray or False
        List of intersection points. If none are found, returns False.

    """

    if not intervals_overlapping([np.min(p1[:, 0]), np.max(p1[:, 0])], [np.min(p1[:, 0]), np.max(p1[:, 0])]):
        return False  # separated in x
    if not intervals_overlapping([np.min(p1[:, 1]), np.max(p1[:, 1])], [np.min(p1[:, 1]), np.max(p1[:, 1])]):
        return False  # separated in y

    intersection_info = [(i, j, p)
                         for i, s1 in enumerate(zip(p1, rotate_by(p1, 1)))
                         for j, s2 in enumerate(zip(p2, rotate_by(p2, 1)))
                         for p in line_segment_intersections[s1, s2]]
    return intersection_info


def old_group_closeby(pts, eps):
    nxgraph = nx.Graph()
    nxgraph.add_nodes_from(np.arange(len(pts), dtype=int))
    nxgraph.add_edges_from(get_potential_intersections(np.stack([pts, pts + eps], axis=1)))
    components = nx.connected_components(nxgraph)
    new_labels = np.empty(len(pts), dtype=int)
    for label, c in enumerate(components):
        new_labels[np.array(list(c))] = label
    return new_labels


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
    edges = [e for e in G.halfedges if e.face is not None]
    new_edges = []
    for e in edges:
        if e.rev not in new_edges:
            new_edges.append(e)
    edges = new_edges

    print('number of edges:', len(edges))
    line_segments = np.array([[e.orig['pos'], e.dest['pos']] for e in edges])
    print(line_segments.shape)

    timer = VerboseTimer()
    # ------ get all crossings ------
    # TODO implement sweeping line algorithm https://www.geeksforgeeks.org/given-a-set-of-line-segments-find-if-any-two-segments-intersect/

    # crossing_dict = {i: {} for i in range(len(line_segments))}
    crossings = []
    crossings_to_edges = []
    # crossing_dict[i][j] is a list of points where edges[i] and edges[j] cross.
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

    print('n crossings:', len(crossings))
    clustering = group_closeby(crossings, eps)
    timer.round('clustered crossings')
    print('reduced n crossings', np.max(clustering) + 1)
    first_occurences = np.argmax(clustering[None] == np.arange(np.max(clustering) + 1)[:, None], axis=1)
    filtered_crossings = crossings[first_occurences]

    n_filtered_crossings = len(filtered_crossings)
    filtered_crossings_to_edges = [set() for i in range(n_filtered_crossings)]  # TODO: get rid of
    edges_to_crossings = [set() for i in range(len(edges))]
    for i, edge_ids in enumerate(crossings_to_edges):
        filtered_crossings_to_edges[clustering[i]].update(edge_ids)
        for e in edge_ids:
            edges_to_crossings[e].add(clustering[i])
    timer.round('filtered crossings')

    # ------ get crossing orders, construct nx graph ------
    print('getting crossing orders..')
    # nodes = np.arange(n_filtered_crossings)
    nx_edges = []
    edge_to_ordered_ids = []
    for i in tqdm(range(len(line_segments))):
        e = edges[i]
        crossing_ids = np.array(list(edges_to_crossings[i]))
        crossing_positions = filtered_crossings[crossing_ids]
        progression_along_edge = ((crossing_positions - e.orig['pos'][None]) * (e.dest['pos'] - e.orig['pos'])[None]).sum(
            -1)
        order = np.argsort(progression_along_edge)
        ordered_ids = crossing_ids[order]
        edge_to_ordered_ids.append(ordered_ids)
        nx_edges.append(np.stack([ordered_ids[:-1], ordered_ids[1:]], axis=-1))
    nx_edges = np.concatenate(nx_edges)
    timer.round('crossing orders')

    print('constructing nx graph..')
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(nx_edges)
    nx_positions = {i: pos for i, pos in enumerate(filtered_crossings)}
    timer.round('nx graph')
    print('order of nx graph:', nx_graph.order())
    print('converting to EHEG..')

    overlap_G, v_lookup = EHEG_from_nx(nx_graph, nx_positions, return_v_lookup=True)
    overlap_G.recompute_lengths_and_angles()
    timer.round('EHEG')

    # ------ assign original vertices, edges ------
    overlap_G.check_consistency()
    print('assigning original vertices and edges..')
    for v in overlap_G.vertices:
        v['original_vertices'] = set()
    for e in overlap_G.halfedges:
        e['original_edges'] = set()
        # to keep track of which faces adjacent to original edges were next to each other
        e['original_face_groups'] = dict()
    for f in overlap_G.faces:
        f['original_faces'] = set()

    # get face orientations
    face_orientations = {}
    for f in G.faces:
        face_orientations[f] = f.area() > 0

    for i in tqdm(range(len(edges))):
        ordered_ids_0 = edge_to_ordered_ids[i]
        if len(ordered_ids_0) == 0:
            assert False, f'this should not happen..'
        for e in (edges[i], edges[i].rev):
            ordered_ids = ordered_ids_0
            if e is edges[i].rev:
                ordered_ids = ordered_ids[::-1]
            for idx in ordered_ids:
                if idx in v_lookup and 'original_vertices' in v_lookup[idx]:  # FIXME: this is sketchy..
                    v_lookup[idx]['original_vertices'].add(e.orig)
                    break
            if not e.on_border() and not face_orientations[e.face]:
                ordered_ids = ordered_ids[::-1]
            for k, l in zip(ordered_ids[:-1], ordered_ids[1:]):
                if k not in v_lookup or l not in v_lookup:
                    continue  # this can happen if dangling edges were deleted
                for new_edge in v_lookup[k].outgoing_iter():
                    if v_lookup[l] is new_edge.dest:
                        new_edge['original_edges'].add(e)
                        #new_edge.rev['original_edges'].add(e)
                        if not e.on_border():
                            if frozenset((e, e.rev)) not in new_edge['original_face_groups']:
                                new_edge['original_face_groups'][frozenset((e, e.rev))] = set()
                            new_edge['original_face_groups'][frozenset((e, e.rev))].add(e.face)

    for e in overlap_G.halfedges:
        e['original_face_groups'] = [frozenset(group) for group in e['original_face_groups'].values()]
    # print([len(e['original_face_groups'][0]) for e in overlap_G.halfedges if e['original_face_groups']])
    # print(len([len(e['original_face_groups'][0]) for e in overlap_G.halfedges if e['original_face_groups'] and len(e['original_face_groups'][0]) == 2]))
    # print(len(set([tuple(e['original_face_groups']) for e in overlap_G.halfedges if e['original_face_groups'] and len(e['original_face_groups'][0]) == 2 ])))

    # ------ assign original faces ------
    print('assigning original faces..')
    initial_edge = next(overlap_G.border_edge_iter()).rev
    assert not initial_edge.on_border()
    frontier = [(initial_edge, {e_orig.face for e_orig in initial_edge['original_edges'] if not e_orig.on_border()})]
    yet_to_assign = set(overlap_G.faces)
    while yet_to_assign:
        current_halfedge, original_faces = frontier.pop()
        if current_halfedge.on_border():
            assert False
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
    print('done.')
    return overlap_G


MOUNTAIN = 1
VALLEY = -1
ORIGINAL_FACES = 'original_faces'
FACES_OF_FOLDS_ON_EDGE = 'original_face_groups'
LENGTH = 'length'
SORTED_ORIGINAL_FACES = 'sorted_original_faces'

def find_triplet_overlap_areas(G):
    result = defaultdict(float)
    for f in G.faces:
        area = f.area()
        for triplet in itertools.combinations(f[ORIGINAL_FACES], 3):
            result[frozenset(triplet)] += area
    return result


def find_fold_over_facet_lengths(G):
    result = defaultdict(float)
    for e in G.halfedges:
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
    for e in G.halfedges:
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


def find_folded_face_order(G, over_under_pairs=(), solver=None, double_fold_weight=1, allow_slack=True, problem_file=None):
    if problem_file is None:  # by default, use temporary file
        with tempfile.TemporaryDirectory() as directory_name:
            filename = os.path.join(directory_name, 'Problem.lp')
            return find_folded_face_order(G, over_under_pairs, solver, allow_slack, problem_file=filename)
    print(f'Processing overlap graph with {G.order} facets..')
    triplet_overlap_areas = find_triplet_overlap_areas(G)
    fold_over_facet_lengths = find_fold_over_facet_lengths(G)
    coincident_fold_lengths = find_conincident_fold_lengths(G)
    print(f'Found {len(triplet_overlap_areas)} overlapping triplets, ' \
          f'{len(fold_over_facet_lengths)} folds over facets, ' \
          f'{len(coincident_fold_lengths)} coincident folds..')

    n_over_under_before = len(over_under_pairs)
    over_under_pairs = infer_additional_over_under_pairs(over_under_pairs, set(triplet_overlap_areas.keys()));

    print(
        f'Preprocessing increased number of known over-under relations from {n_over_under_before} to {len(over_under_pairs)}..')

    n_vars = 0

    def get_varname():
        nonlocal n_vars
        n_vars += 1
        return n_vars

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

    for a, b, c in tqdm(triplet_overlap_areas):
        val = over(a, b) + over(b, c) + over(c, a)
        add_constraint(val <= 2)
        add_constraint(val >= 1)

    for (a, b), c in tqdm(fold_over_facet_lengths):
        add_constraint(over(a, c) + over(c, b) == 1)

    for (a, b), (c, d) in tqdm(coincident_fold_lengths):
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
        if objective != 0:
            prob.setObjective(objective)
        # write ILP to file and solve it
        print(f'Writing problem to file {problem_file} ..')
        prob.writeLP(problem_file)
        if solver is None:
            for solver_class in SOLVER_ORDER:
                solver = solver_class()
                if solver.available():
                    break
            if solver is None:
                solver = 'auto'

        print(f'Solving ILP with solver {solver.__class__.__name__ if isinstance(solver, pulp.LpSolver) else solver}..')
        prob.solve(solver=solver)
    else:
        print('Skipping ILP since everything is already determined..')

    def comparison_func(a, b):
        result = over_dict.get((a, b), 0.5)
        result = 1 - over_dict.get((b, a), 0.5) if result is 0.5 else result
        if not isinstance(result, (int, float, bool)):
            result = result.value()
        if result is None:
            result = 0
        result = int(1 - 2 * result)
        if result is 0:
            print('order unclear!')
        return result

    print(f'determining face order..')
    for f in G.faces:
        original_faces = list(f[ORIGINAL_FACES]).copy()
        original_faces.sort(key=cmp_to_key(comparison_func))
        f[SORTED_ORIGINAL_FACES] = original_faces

    print(f'assigning creases..')
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
THIS_WAY = 'this_way'


def assign_shrink_rotate_creases(SRG):
    """
    Assigns crease orientation to shrik rotate cp, given directions on the original graph.
    """
    twistfaces = [f for f in SRG.faces if 'twistrotate' in f.attributes]
    for f in twistfaces:
        e_twist = f.any_side
        while e_twist.rev.on_border():
            e_twist = e_twist.nex
        # get edge in orig graph that matches e_twist
        e = None
        for e_orig in f['pre_conway'].halfedge_iter():
            if e_orig.rev in e_twist.rev.nex.nex.rev.face['pre_conway'].halfedge_iter():
                e = e_orig
                break
        assert e is not None
        e_twist_initial = e_twist
        while True:
            if THIS_WAY in e.attributes and not e_twist.rev.on_border():
                e_twist.rev[CREASE_ASSIGNMENT] = MOUNTAIN
                e_twist.rev.nex.nex.nex[CREASE_ASSIGNMENT] = VALLEY
                if True: #e_twist.rev['in_angle'] < np.pi / 2:
                    e_twist.rev.nex[CREASE_ASSIGNMENT] = MOUNTAIN
                    e_twist.rev.nex.nex[CREASE_ASSIGNMENT] = VALLEY
                else:
                    e_twist.rev.nex[CREASE_ASSIGNMENT] = VALLEY
                    e_twist.rev.nex.nex[CREASE_ASSIGNMENT] = MOUNTAIN
            e = e.nex
            e_twist = e_twist.nex
            if e_twist is e_twist_initial:
                break
    for e in SRG.halfedges:  # make crease assignment of e and e.rev consistent
        if CREASE_ASSIGNMENT in e.attributes:
            if CREASE_ASSIGNMENT in e.rev.attributes:
                assert e[CREASE_ASSIGNMENT] == e.rev[CREASE_ASSIGNMENT]
            else:
                e.rev[CREASE_ASSIGNMENT] = e[CREASE_ASSIGNMENT]


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
    print('number of pairs', len(over_under_pairs))
    return over_under_pairs


TOP = 'top_side'
BOTTOM = 'bottom_side'


def face_order_to_clean_graph(G, side=TOP):
    from .classifiers import CountingClassifier, RepresentationClassifier, lambda_classifier
    from .conversions import EHEG_from_nx
    assert side in (TOP, BOTTOM)
    cc = CountingClassifier(RepresentationClassifier())
    G = G.copy()
    for f in G.faces:
        try:
            f['color_key'] = cc.classify(f['sorted_original_faces'][0 if side is TOP else -1])
        except IndexError:
            f['color_key'] = 1000

    to_delete = [e
                 for e in G.halfedges
                 if not (e.on_border() or e.rev.on_border()) and e.face['color_key'] is e.rev.face['color_key']]

    G.halfedges.difference_update(to_delete)
    G = EHEG_from_nx(G.to_networkx_undirected(), {v: v['pos'] for v in G.vertices})
#     to_join = []
#     for v in G.vertices:
#         if not v.on_border() and v.order() == 2:
#             to_join.append(v)
#     for v in to_join:
#         G.join_vertex(v)
    G.recompute_lengths_and_angles()
    cc = CountingClassifier(lambda_classifier(lambda f: f.area()//0.0001)())
    for f in G.faces:
        f['color_key'] = cc.classify(f)
    return G


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
    assert area_eps == 0, 'Not implemeted!'
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


def save_results(results, path='results', render_settings=None):
    if render_settings is None:
        render_settings = dict(
            figsize=(7, 7),
            render_edges=True,
            render_faces=False,
            render_vertices=False,
            face_inset=0,
            for_cutting = False,
            line_width=3,
        )
    os.makedirs(path, exist_ok=True)
    cp_settings = render_settings.copy()
    cp_settings.update(dict(filename=os.path.join(path, 'CP'), render_faces=False))
    results['CP'].show(**cp_settings)
    
    folded_settings = render_settings.copy()
    folded_settings.update(dict(filename=os.path.join(path, 'top')))
    results['folded_view_top'].show(**folded_settings)
    folded_settings.update(dict(filename=os.path.join(path, 'bottom')))
    results['folded_view_bottom'].show(**folded_settings)

from functools import lru_cache