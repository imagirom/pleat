import numpy as np
import scipy as sc
from copy import copy

from .conway import dual_graph, twist_rotate_graph
from .half import HalfEdgeGraph
from .base import rotation_matrix
from .utils import invert_mapping
from .overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from .layout import optimize_rotation


def random_directed_set(edges):
    if isinstance(edges, HalfEdgeGraph):
        edges = edges.halfedges
    directed_edges = set()
    for e in edges:
        if e.rev not in directed_edges:
            directed_edges.add(e)
    return directed_edges


def reciprocal_figure(G, reciprocal_pos_key='reciprocal_pos', rcond=1e-7):
    # Step 1: Choose direction for every interior edge.

    #print(len(G.halfedges))
    directed_edges = random_directed_set([e for e in G.halfedges
                                          if not (e.on_border() or e.rev.on_border())])
    #print(f'n edges: {len(directed_edges)}')

    # Step 2: Construct array of all vectors of the directed edges, mapping from edge to index
    edge_vectors = np.stack([e.orig['pos'] - e.dest['pos'] for e in directed_edges])

    dual_vectors = edge_vectors @ rotation_matrix(np.pi/2)
    dual_directions = dual_vectors / np.linalg.norm(dual_vectors, axis=1, keepdims=True)
    edges_to_ids = {e: i for i, e in enumerate(directed_edges)}
    #print('dual directions shape:', dual_directions.shape)

    # Step 3: Formulate constraints as linear problem Ax = 0
    # Every constraint is a row in the matrix A.
    # Every interior vertex leads to a constraint.
    # Hence, compute one row for each interior vertex.

    interior_vertices = [v for v in G.vertices if not v.on_border()]
    #print('number of interor vertices=constraints:', len(interior_vertices))

    rows = []
    n_edges = len(directed_edges)
    for v in interior_vertices:
        row = np.zeros(n_edges, dtype=np.float32)
        for e in v.outgoing_iter():
            if e in directed_edges:
                row[edges_to_ids[e]] = -1
            else:
                row[edges_to_ids[e.rev]] = 1
        rows.append(row)
    B = np.stack(rows)
    A = (B[:, None, :] * dual_directions.T[: None]).reshape(-1, n_edges)
    #print('A.shape:', A.shape)
    U = sc.linalg.null_space(A, rcond=rcond)
    #print('U.shape:', U.shape)
    assert U.shape[1] > 0, f'G does not have a reciprocal figure!'
    # Step 4: Formulate and solve least squares problem to make reciprocal graph as
    # similar as possible to result of conway.dual_graph()(G)

    # need map face in primal -> vertex in dual!

    # need linear map coords in solution space -> dual edge lenghts
    # this is just U @ coords
    #coords = np.random.rand(U.shape[1])
    #print(((U @ coords)[:, None] * dual_directions).shape)

    # need linear map dual edge lengths -> dual edge offsets
    # this is just dual_directions

    # need linear map (dual edge offsets, position of interior_vertices[0]) -> dual vertex positons
    to_process = set(G.faces)
    anchor = to_process.pop()
    coefficients = {anchor: np.zeros(n_edges, dtype=np.float32)}
    border = {anchor}
    while border:
        new_border = set()
        for f in border:
            for e in f.halfedge_iter():
                f2 = e.rev.face
                if f2 not in coefficients:
                    if e in directed_edges:
                        coefficients[f2] = copy(coefficients[f])
                        coefficients[f2][edges_to_ids[e]] = -1
                    elif e.rev in directed_edges:
                        coefficients[f2] = copy(coefficients[f])
                        coefficients[f2][edges_to_ids[e.rev]] = 1
                    else:
                        continue
                    new_border.add(f2)
        border = new_border
    assert set(coefficients.keys()) == set(G.faces)

    faces = G.faces
    n_faces = len(faces)
    #print('n_faces:', n_faces)
    D2P = np.stack([coefficients[f] for f in faces])
    #print('D2P.shape:', D2P.shape)

    #M = (D2P @ U)#[:, None] * dual_directions

    M = np.moveaxis(np.dot(D2P, np.moveaxis(U[:, :, None] * dual_directions[:, None], 0, 1)), 1, 2)
    #print('M.shape', M.shape)

    # Add two columns to M, corresponding to the offset of the dual graph
    xy_columns = np.zeros((n_faces, 2, 2), dtype=np.float32)
    xy_columns[:, 0, 0] = 1
    xy_columns[:, 1, 1] = 1
    M = np.concatenate([xy_columns, M], axis=-1)

    # Get 'ground truth' face centers: for now just com of the faces
    face_centers = np.stack([f.midpoint() for f in faces])
    #print('face_centers.shape', face_centers.shape)

    # flatten xy
    M = M.reshape(n_faces * 2, -1)
    face_centers = face_centers.reshape(n_faces * 2)

    print(f'optimizing rotation centers using {M.shape[-1]} degrees of freedom..')
    # solve the least squares problem
    sol = sc.optimize.lsq_linear(M, face_centers, lsq_solver='exact')
    assert sol['success'], f"{sol['message']}"
    sol = sol['x']

    #sol[2:] *= 1 + np.random.randn(len(sol) - 2) * 0.3
    dual_vertices = M @ sol
    dual_vertices = dual_vertices.reshape(-1, 2)

    # save reciprocal positions in attribute of faces of G
    if reciprocal_pos_key is not None:
        for i, f in enumerate(faces):
            f[reciprocal_pos_key] = dual_vertices[i]

    # Step 5: make reciprocal figure into face graph
    D, (v_map, e_map, f_map) = G.copy(return_mappings=True)
    face2reciprocalpos = {f_map[f]: dual_vertices[i] for i, f in enumerate(faces)}

    D = dual_graph()(D)
    for v in D.vertices:
        v['pos'] = face2reciprocalpos[v['pre_conway']]

    inv_v_map = invert_mapping(f_map)
    for v in D.vertices:
        v['pre'] = inv_v_map[v['pre_conway']]
    inv_e_map = invert_mapping(e_map)
    for e in D.halfedges:
        if 'pre_conway' in e.attributes:
            e['pre'] = inv_e_map[e['pre_conway']]
    inv_f_map = invert_mapping(v_map)
    for f in D.faces:
        f['pre'] = inv_f_map[f['pre_conway']]

    # idea: always have 'pre' and 'nex' keys for each operation mapping graphs to graphs, when applicable

    return D


def shrink_rotate_graph(G, alpha=np.pi/5, factor=0.5, **reciprocal_figure_kwargs):
    f = next(iter(G.faces))
    if 'reciprocal_pos' not in f or len(reciprocal_figure_kwargs):
        print('Calculating reciprocal figure..')
        _ = reciprocal_figure(G, **reciprocal_figure_kwargs)
        print('Done with reciprocal figure.')
    # Step 6: get shrink-rotate graph and apply mapping
    SRG, (_, _, f_map) = G.copy(return_mappings=True)
    # faces = []
    # dual_vertices = []
    # for v in D.vertices:
    #     dual_vertices.append(v['pos'])
    #     faces.append(v['pre'])

    inverse_f_map = invert_mapping(f_map)  # to get from 'pre_conway' of SRG to G

    SRG = twist_rotate_graph()(SRG)

    twistfaces = list(filter(lambda f: 'twistrotate' in f.attributes, SRG.faces))
    for f in twistfaces:
        ps, vs = np.array([[v['pos'], v] for v in f.vertex_iter()], dtype=object).T
        ps = np.stack(ps)

        midpoint = np.mean(ps, axis=0, keepdims=True)
        ps = midpoint + (ps - midpoint) * 2

        for v, p in zip(vs, ps):
            v['base_pos'] = p

    for f in twistfaces:
        ps, vs = np.array([[v['base_pos'], v] for v in f.vertex_iter()], dtype=object).T
        ps = np.stack(ps)
        assert 'pre_conway' in f.attributes
        f['pre_conway'] = inverse_f_map[f['pre_conway']]
        rotation_center = f['pre_conway']['reciprocal_pos']
        f['rotation_center'] = rotation_center

        ps = rotation_center + (ps - rotation_center) @ rotation_matrix(alpha) * factor

        for v, p in zip(vs, ps):
            v['pos'] = p

    SRG.recompute_lengths_and_angles()
    return SRG


def kawasaki_sum(v):
    angles = np.abs(np.array([e['in_angle'] for e in v.incoming_iter()]))
    assert len(angles) % 2 == 0
    return np.sum(((angles + 2*np.pi) % (2*np.pi)) * (-1) ** np.arange(len(angles)))


def max_kawasaki_sum(vertices):
    if isinstance(vertices, HalfEdgeGraph):
        vertices = [v for v in vertices.vertices if not v.on_border()]
    return np.max([kawasaki_sum(v) for v in vertices])


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
        if e.on_border() or e.rev.on_border():
            if CREASE_ASSIGNMENT in e.attributes:
                del e[CREASE_ASSIGNMENT]
        elif CREASE_ASSIGNMENT in e.attributes:
            if CREASE_ASSIGNMENT in e.rev.attributes:
                assert e[CREASE_ASSIGNMENT] == e.rev[CREASE_ASSIGNMENT]
            else:
                e.rev[CREASE_ASSIGNMENT] = e[CREASE_ASSIGNMENT]


def assign_this_way_by_face_z_order(G, key='z_order'):
    for e in G.halfedges:
        if e.on_border() or e.rev.on_border():
            continue
        if THIS_WAY in e:
            del e[THIS_WAY] # only in case it was assigned before

        f1, f2 = e.face, e.rev.face
        if f1[key] > f2[key]:
            e[THIS_WAY] = True
        elif f1[key] == f2[key]:
            z_orig = np.mean([f[key] for f in e.orig.true_face_iter()])
            z_dest = np.mean([f[key] for f in e.dest.true_face_iter()])
            if z_orig > z_dest:
                e[THIS_WAY] = True


def assign_this_way_by_vertex_z_order(G, key='z_order'):
    for e in G.halfedges:
        if e.on_border() or e.rev.on_border():
            continue
        if THIS_WAY in e:
            del e[THIS_WAY]  # only in case it was assigned before

        v1, v2 = e.orig, e.dest
        if v1[key] > v2[key]:
            e[THIS_WAY] = True
        elif v1[key] == v2[key]:
            z_orig = np.mean([v[key] for v in e.face.vertex_iter()])
            z_dest = np.mean([v[key] for v in e.rev.face.vertex_iter()])
            if z_orig > z_dest:
                e[THIS_WAY] = True


def make_SRG(G, simplify_boundary=True, **srg_kwargs):
    SRG = shrink_rotate_graph(G, **srg_kwargs)
    SRG.recompute_lengths_and_angles()
    assign_shrink_rotate_creases(SRG)

    colors = {
        0: (0, 0, 0),
        1: (1, 0, 0),
        -1: (0, 0, 1)
    }
    for e in SRG.halfedges:
        e['color_key'] = colors[e.attributes.get('crease_assignment', 0)]

    if simplify_boundary:
        # join unneccessary boundary vertices
        to_join = []
        for v in G.vertices:
            if v.on_border() and v.order() == 2:
                to_join.append(v)
        for v in to_join:
            G.join_vertex(v)
        G.recompute_lengths_and_angles()

    mks = max_kawasaki_sum(SRG)
    if mks > 1e-12:
        print('High mks: ', mks)

    print(f'CP has {len(SRG.halfedges) // 2} edges')

    return SRG