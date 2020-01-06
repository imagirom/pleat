import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from functools import cmp_to_key
from tqdm.auto import tqdm
from fastcluster import linkage_vector, linkage
from scipy.cluster.hierarchy import fcluster

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
    o1 = orientation([p1, q1, p2], eps)
    o2 = orientation([p1, q1, q2], eps)
    o3 = orientation([p2, q2, p1], eps)
    o4 = orientation([p2, q2, q1], eps)

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


def group_closeby(pts, eps):
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


def overlap_graph(G, eps=1e-10):
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

    # ------ get all crossings ------
    # TODO implement sweeping line algorithm https://www.geeksforgeeks.org/given-a-set-of-line-segments-find-if-any-two-segments-intersect/

    crossing_dict = {i: {} for i in range(len(line_segments))}
    crossings = []
    crossings_to_edges = []
    # crossing_dict[i][j] is a list of points where edges[i] and edges[j] cross.
    # iterate over all pairs

    potential_intersections = get_potential_intersections(line_segments, epsilon=eps)
    # print('potential_intersections:', potential_intersections)
    for i, j in potential_intersections:
        l1, l2 = line_segments[i], line_segments[j]
        # if any([e.nex in [edges[j], edges[j].rev] for e in [edges[i], edges[i].rev]]):
        #    continue
        intersections = line_segment_intersections(l1, l2, eps=eps)
        if not intersections:
            continue
        # if len(intersections) > 1:
        #     print('oho', i, j, intersections)
        crossings.extend(intersections)
        for _ in range(len(intersections)):
            crossings_to_edges.append((i, j))
        crossing_indices = list(range(len(crossings) - len(intersections), len(crossings)))
        for k, l in [(i, j), (j, i)]:  # add intersections to both lines
            if l not in crossing_dict[k]:
                crossing_dict[k][l] = []
            crossing_dict[k][l].extend(crossing_indices)
    crossings = np.array(crossings)

    # ------ group closeby crossings ------

    print('n crossings:', len(crossings))

    clustering = fast_group_closeby(crossings, eps)
    #clustering = group_closeby(crossings, eps)
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
    print(nx_edges.shape)

    print('constructing nx graph..')
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(nx_edges)
    nx_positions = {i: pos for i, pos in enumerate(filtered_crossings)}

    print('order of nx graph:', nx_graph.order())
    print('converting to EHEG..')

    overlap_G, v_lookup = EHEG_from_nx(nx_graph, nx_positions, return_v_lookup=True)
    overlap_G.recompute_lengths_and_angles()

    # ------ assign original vertices, edges ------
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
            v_lookup[ordered_ids[0]]['original_vertices'].add(e.orig)
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
    print('done.')
    return overlap_G

MOUNTAIN = 1
VALLEY = -1


def find_face_order(overlap_G, over_under_pairs=None, solver=None, ignore_area_threshold=0):
    # U is the set of original faces whose local orderings are to be determined
    # V is the set of faces in the overlap graph
    # over_under_pairs is list of tuples of faces (f1, f2) in U with f1 over f2
    # v is a face in overlap_G, u one in G
    V = list(overlap_G.faces)
    # FIXME: using V_maximal can lead to problems in conjunction with filtering facets by size
    V_maximal = list(f for f in overlap_G.faces
                     if all([len(e['original_face_groups']) for e in f.halfedge_iter()]))
    #print(f'V: {len(V)}, V_maximal: {len(V_maximal)}')
    U = list(set.union(*(v['original_faces'] for v in V)))
    print('V', len(V))
    print('U', len(U))
    U2face = U
    U = list(range(len(U)))
    V2face = V
    V = list(range(len(V)))

    # phi maps face in overlap to group of original faces
    phi = {v: [u for u in U if U2face[u] in V2face[v]['original_faces']] for v in V}

    # filter out small faces in overlap graph:
    V_unfiltered = V
    V = [v for v in V_unfiltered
         if V2face[v].area() > ignore_area_threshold]


    # rho is set of lists of face groups (which are lists of faces), for all edges with more than two face groups
    face2U = {face: u for u, face in enumerate(U2face)}
    rho = set([frozenset(frozenset(face2U[face] for face in group)
                         for group in e['original_face_groups'] if group and len(group) == 2)
               for v in V
               for e in V2face[v].halfedge_iter() if e['original_face_groups']
               if len(e['original_face_groups']) >= 2])
    rho = {groups for groups in rho if len(groups) > 1}
    # tau is set of triples of faces in U, such that for every v one gets for every edge e all triples (a, b, c) with
    # c in phi[v] and not in any face group of e and
    # (a, b) a face group of e
    tau = set([(*[face2U[f] for f in group], face2U[c])
               for v in V
               for v_face in [V2face[v]]
               for e in v_face.halfedge_iter()
               for group in e['original_face_groups']
               for c in v_face['original_faces']
               #if v_face in V_maximal
               if len(group) == 2
               if c not in set.union(set(), *e['original_face_groups'])
    ])

    # Problem: Some faces in a group in 'original_face_groups' are not in 'original_faces'
    # How can that happen?

    # tau3 = set([(v, *[face2U[f] for f in group], face2U[c])
    #            for v in V2face
    #            for e in v.halfedge_iter()
    #            for group in e['original_face_groups']
    #            for c in v['original_faces']
    #            if len(group) == 2
    #            if c not in group
    #            ])
    # for v, a, b, c in tau3:
    #     assert U2face[a] in v['original_faces']
    #     assert U2face[b] in v['original_faces']
    #     assert U2face[c] in v['original_faces']

    # should be the same as tau but is not..
    tau2 = set([(*[face2U[f] for f in group], c)
                for v in V_unfiltered  # to use V_maximal here, use e['original_face_groups] instead of e.rev['..']!
                for e in V2face[v].halfedge_iter()
                for group in e.rev['original_face_groups'] if len(group) == 2  # should always be 2, except if border
                for c in phi[v] if U2face[c] not in set.union(set(), *e['original_face_groups'])
                ])

    print('rho', len(rho))
    print('tau', len(tau))
    print('tau2', len(tau2))
    #phi = {0: [0, 1, 2, 3, 4]}

    print('sum', sum([len(p) for p in phi.values()]))
    # X is the set of Variables, that is tuplesV in U that overlap in overlap_G
    X = list(set([(a, b)
                  for v in V_unfiltered  # there can be conditions
                  for U_v in [phi[v]]
                  for i, a in enumerate(U_v)
                  for b in U_v[i + 1:]
                  ]))
    print(len(X))

    # Y are triplets that can be compared. Is used for ordering constraints
    # Y = list(set([(a, b, c)
    #                for v in V_maximal
    #                for i, a in enumerate(phi[v])
    #                for j, b in enumerate(phi[v][i + 1:])
    #                for c in phi[v][i + j + 2:]]))

    import pulp
    prob = pulp.LpProblem("Overlap Problem", pulp.LpMinimize)
    choices = pulp.LpVariable.dicts("Choice", X, 0, 1, pulp.LpBinary)
    prob += 0, "Arbitrary Objective Function"

    for a, b in X:
        assert (b, a) not in X

    def over(a, b):
        # TODO: if a, b in over_under_pairs, just return a constant
        if (a, b) in choices:
            return choices[(a, b)]
        else:
            return 1 - choices[(b, a)]

    # add ordering constraints: no a > b > c > a is allowed
    added_constraints = set()
    from tqdm.auto import tqdm
    n_triplets = sum(n * (n-1) * (n-2) // 6
                     for n in (len(phi[v]) for v in V if V2face[v] in V_maximal))
    for a, b, c, in tqdm(((a, b, c)
                          for v in V #V_maximal #FIXME V vs V_maximal makes a differenc, which it definitely should not
                          for i, a in enumerate(phi[v])
                          for j, b in enumerate(phi[v][i + 1:])
                          for c in phi[v][i + j + 2:]
                          if V2face[v] in V_maximal
                          ), total=n_triplets, desc='adding ordering constraints'):

        if (a, b, c) in added_constraints:
            continue
        added_constraints.add((a, b, c))
        prob += pulp.lpSum([over(a, b), over(b, c), over(c, a)]) <= 2, ""
        prob += pulp.lpSum([over(a, c), over(c, b), over(b, a)]) <= 2, ""
    print(f'added {len(added_constraints)} ordering constraints.')

    # add fold-adjacency constraints: if a was next to b, it cannot be c -> a c b (where -> means over an edge)
    print(f'adding {len(tau2)} fold-adjacency constraints..')
    errors = 0
    for a, b, c in tqdm(tau2):
        try:
            prob += pulp.lpSum([over(a, c), over(c, b)]) == 1, ""
        except Exception as e:
            errors += 1
            print(e)
    if errors > 0:
        print(f'\ncould not add {errors} constraints!\n')

    # add directly specified over under constraints
    print(f'adding {len(over_under_pairs)} over-under constraints')
    if over_under_pairs is not None:
        for f1, f2 in over_under_pairs:
            try:
                prob += over(face2U[f1], face2U[f2]) == 1, ""
            except:
                print('not overlapping..')

    # add rho constraints corresponding to edges with multiple face groups
    included = set()
    for groups in tqdm(rho):
        groups = list(groups)
        for i, (a, b) in enumerate(groups):
            for c, d in groups[i+1:]:
                if a in (c, d):
                    print('whoops')
                    assert b in (c, d)
                    continue
                if frozenset((frozenset((a, b)), frozenset((c, d)))) in included:
                    continue
                included.add(frozenset((frozenset((a, b)), frozenset((c, d)))))
                group1_over_group2 = pulp.lpSum([over(c, a), over(c, b), over(d, a), over(d, b)])
                # add auxilliary variable
                aux_var = pulp.LpVariable(f"{a},{b},{c},{d}", 0, 2, pulp.LpInteger)
                prob += group1_over_group2 - 2 * aux_var == 0, ""

    prob.writeLP("Overlap.lp")

    # Solve the problem
    if solver is None:
        solver_order = [pulp.CPLEX, pulp.GLPK]
        for solver_class in solver_order:
            solver = solver_class()
            if solver.available():
                break
        if solver is None:
            solver = 'auto'

    print(f'solving ILP with solver {solver.__class__.__name__ if isinstance(solver, pulp.LpSolver) else solver}..')
    prob.solve(solver=solver)

    # The status of the solution is printed to the screen
    #print(("status:", pulp.LpStatus[prob.status]))
    assert prob.status is pulp.LpStatusOptimal, f'ILP status: {pulp.LpStatus[prob.status]}'
    print('ILP solved')
    #for t in choices:
    #    print(pulp.value(choices[t]))

    def comparison_func(a, b):
        if (a, b) in choices:
            order = pulp.value(choices[(a, b)])
        elif (b, a) in choices:
            order = pulp.value(choices[(b, a)])
            if order is not None:
                order = 1-order
        else:  # this case happens when some faces in V are ignored
            order = None
        if order is None:
            print('order unclear!')
            return 0
        else:
            return 2 * order - 1
    print(f'determining face order..')
    for v in V_unfiltered:
        face = V2face[v]
        phi[v].sort(key=cmp_to_key(comparison_func))
        face['sorted_original_faces'] = [U2face[u] for u in phi[v]]

    print(f'determining crease assignment..')
    crease_assignment = dict()
    for f1 in U2face:
        # TODO: this assumes the graph is two colored with color key 'color_key'..
        for e in f1.halfedge_iter():
            f2 = e.rev.face
            if f2 is None:
                continue
            if f1['color_key']:
                crease_assignment[e] = comparison_func(face2U[f1], face2U[f2])
            else:
                crease_assignment[e] = -1 * comparison_func(face2U[f1], face2U[f2])
            e['color_key'] = crease_assignment[e]

    print('done.\n')
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


def get_over_under_pairs_from_creases(G, two_coloring_key='color_key'):
    # return list of pairs (f1, f2) with f1 over f2
    # G is assumed to be two-colored
    over_under_pairs = []
    for e in G.halfedges:
        crease_type = e.attributes.get('crease_type', None)
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


def fold_complete(G, initial_face=None, overlap_eps=1e-6, area_eps=0):
    fold_wireframe(G, initial_face=initial_face)
    over_under_pairs = get_over_under_pairs_from_creases(G)
    result = dict()
    G_over = overlap_graph(G, overlap_eps)
    crease_assignment = find_face_order(G_over, over_under_pairs, ignore_area_threshold=area_eps)
    # make CP
    colors = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        -1: (0, 0, 1)
    }
    fold_wireframe(G, initial_face=initial_face)
    for e in G.halfedges:
        e['crease_assignment'] = crease_assignment.get(e, 0)
        e['color_key'] = colors[crease_assignment.get(e, 0)]
    result['CP'] = G
    result['folded_state'] = G_over
    result['folded_view_top'] = face_order_to_clean_graph(G_over, TOP)
    result['folded_view_bottom'] = face_order_to_clean_graph(G_over, BOTTOM)
    return result
