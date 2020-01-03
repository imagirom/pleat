import numpy as np
import eucare as ec
from eucare.base import orientation

render_settings = dict(
    width=1500, height=1500,
    figsize=(7, 7),
    #scale=100,
    render_edges=True,
    render_faces=True,
    render_vertices=False,
    line_width=0.003,
    face_inset=0.0,
    for_cutting=False
)


def transform_heg(G, transformation):
    assert isinstance(G, ec.half.EuclideanPositionHEG), f'{type(G)}'
    if G.order == 0:
        return
    p = G.get_position_view()[0]
    p[:] = transformation(p)


def n_gon_graph(n):
    return ec.half.EuclideanPositionHEG(
        other=ec.prototiles.RegularEuclideanTile(n).make_graph(add_positions=True)[0])


def cyclic_graph(points):
    return ec.half.EuclideanPositionHEG(
        other=ec.prototiles.EuclideanProtoTile(points).make_graph(add_positions=True)[0])


def penta_grid_graph(n):
    # G = n_gon_graph(5)
    #
    # transform_heg(G, lambda p: p * [8, 0.2] + [0, 0])
    #
    # G.add_graph(n_gon_graph(4))
    #
    # transform_heg(G, lambda p: p * [0.5, 3] + [0, 0])
    #
    # G.add_graph(cyclic_graph(points=(np.array([[3, -2], [1, 3], [-0.8, 1.7]]))))
    #
    # G.check_consistency()
    #
    # transform_heg(G, lambda p: p * 0.4 + [0.5, 0.7])
    # G.add_graph(n_gon_graph(6))

    G = ec.half.EuclideanPositionHEG()
    for i in range(n):
        for j in range(n):
            v = [i / 2, j / 2]
            transform_heg(G, lambda p: p + v)
            G.add_graph(n_gon_graph(5))
            transform_heg(G, lambda p: p - v)


def concentric_rings(n, rings, factor):
    r = rings
    G = ec.prototiles.RegularEuclideanTile(n).make_graph(add_positions=True)[0]
    G = ec.half.EuclideanPositionHEG(other=G)
    # Idea: first chamfer without border-delete, then loft.
    G = ec.conway.chamfer_graph(t=1/factor)(
        G,
        delete_on_border=False,
        delete_inner_border=False,
        faces=[f for f in G.faces if f.order() == n]
    )
    for i in range(r):
        G = ec.conway.loft_graph(t=1/factor)(
            G,
            delete_on_border=False,
            faces=[f for f in G.faces if f.order() == n]
        )
    ps, vs = G.get_position_view()
    k = ps.copy()
    k = np.array([complex(*ki) for ki in k])
    k -= np.mean(k)
    k = k / np.max(np.abs(k)) * 3
    k = np.stack([k.real, k.imag], axis=-1)
    ps[:] = k
    G.recompute_lengths_and_angles()
    return G

def sr_graph(n=5, angle=np.pi/4, factor=0.4):
    #G = ec.example_graphs.from_tiles(ec.example_tilesets.t_4_6_12(), n, vertex_based=True) # FIXME: result is one face
    #G = ec.conway.gyro_graph()(G)

    #beta, gamma = 0.4156, 0.65  # this leads to for some reason infeasible problem
    beta, gamma = 0.35, 0.65
    beta *= np.pi
    angle = np.arccos((gamma + np.sin(beta)) / np.sqrt(gamma ** 2 + 2 * gamma * np.sin(beta) + 1))
    factor = gamma / np.sqrt(gamma ** 2 + 2 * gamma * np.sin(beta) + 1)

    #angle, factor = 0.11 * np.pi, 0.54
    G = concentric_rings(12, 2, 3.1)
    #G = ec.example_graphs.from_tiles(ec.example_tilesets.platonic(4), 1, vertex_based=True)
    #G = ec.example_graphs.from_tiles(ec.example_tilesets.t_3_3_4_3_4(), n, vertex_based=True)
    #G = ec.conway.kis_graph()(G)

    G.check_consistency()

    G.normalize_positions()
    #direct_around_origin(G)
  #   for f in G.faces:
  #       if f.order() == 12:
  #           for e in f.halfedge_iter():
  #               e[THIS_WAY] = True
  #       if f.order() == 4:
  #           for e in f.halfedge_iter():
  #               e.rev[THIS_WAY] = False

    G = ec.reciprocal_figures.shrink_rotate_graph(G, angle, factor)
    initial_face = None
    for f in G.faces:
        pos = np.array([v['pos'] for v in f.vertex_iter()])
        if np.all(np.min(pos, axis=0) <= 0) and np.all(np.max(pos, axis=0) > 0):
            initial_face = f
    G.twocolor_faces(initial_face=initial_face)
    G.show(**render_settings)
    assign_MV_to_SRG(G)
    for f in filter(lambda f: f['color_key'], G.faces):
        for e in f.halfedge_iter():
            e['in_angle'] *= -1
    G.recompute_positions()
    #G.show(**render_settings)
    return G


THIS_WAY = 'this_way'


def direct_around_origin(G):
    processed = set()
    for e in G.halfedges:
        if e in processed:
            continue
        processed.add(e)
        processed.add(e.rev)
        ori = orientation([[0, 0], e.orig['pos'], e.dest['pos']], eps=1e-10)
        if ori == 1 or (ori == 0 and np.linalg.norm(e.orig['pos']) > np.linalg.norm(e.dest['pos'])):
            e[THIS_WAY] = True
        else:
            e.rev[THIS_WAY] = True


def assign_MV_to_SRG(SRG):
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
                e_twist.rev['crease_type'] = 'mountain'
                e_twist.rev.nex.nex.nex['crease_type'] = 'valley'
                if True: #e_twist.rev['in_angle'] < np.pi / 2:
                    e_twist.rev.nex['crease_type'] = 'mountain'
                    e_twist.rev.nex.nex['crease_type'] = 'valley'
                else:
                    e_twist.rev.nex['crease_type'] = 'valley'
                    e_twist.rev.nex.nex['crease_type'] = 'mountain'
            e = e.nex
            e_twist = e_twist.nex
            if e_twist is e_twist_initial:
                break
    for e in SRG.halfedges:
        if 'crease_type' in e.attributes:
            if 'crease_type' in e.rev.attributes:
                assert e['crease_type'] == e.rev['crease_type']
            else:
                e.rev['crease_type'] = e['crease_type']


def get_over_under_pairs(G, two_coloring_key='color_key'):
    # return list of pairs (f1, f2) with f1 over f2
    # G is assumed to be two-colored
    over_under_pairs = []
    for e in G.halfedges:
        crease_type = e.attributes.get('crease_type', None)
        if crease_type in ('mountain', 'valley') and not (e.on_border() or e.rev.on_border()):
            e_above = e if e.face[two_coloring_key] else e.rev
            if crease_type is 'mountain':
                e_above = e_above.rev
            over_under_pairs.append([e_above.face, e_above.rev.face])
    print('number of pairs', len(over_under_pairs))
    return over_under_pairs


#G = sr_graph(factor=0.1, angle=0)
G = sr_graph()
#G.add_graph(G.copy())
print('inner edges', len(G.halfedges) / 2 - len(G.border_edges()))

#G.show(**render_settings)

print('assigned creases', len([e for e in G.halfedges
              if e.attributes.get('crease_type', None) in ('mountain', 'valley') and not (e.on_border() or e.rev.on_border())]))
over_under_pairs = get_over_under_pairs(G)

G = ec.overlap.overlap_graph(G, eps=1e-12) #1e-4
G.show(**render_settings)
import pulp
ec.overlap.find_face_order(G, over_under_pairs)

#for e in G.halfedges:
#    print(e['original_face_groups'])


TOP = 'top_side'
BOTTOM = 'bottom_side'


def show_folded(G, side=TOP):
    assert side in (TOP, BOTTOM)
    cc = ec.classifiers.CountingClassifier(ec.classifiers.RepresentationClassifier())
    G = G.copy()
    for f in G.faces:
        #f['color_key'] = len(f['original_faces'])
        f['color_key'] = cc.classify(f['sorted_original_faces'][0 if side is TOP else -1])
        #print(f['color_key'])
    #G.show(**render_settings)

    to_delete = [e
                 for e in G.halfedges
                 if not (e.on_border() or e.rev.on_border()) and e.face['color_key'] is e.rev.face['color_key']]

    G.halfedges.difference_update(to_delete)
    G = ec.conversions.EHEG_from_nx(G.to_networkx_undirected(), {v: v['pos'] for v in G.vertices})
    to_join = []
    for v in G.vertices:
        if not v.on_border() and v.order() == 2:
            to_join.append(v)
    for v in to_join:
        G.join_vertex(v)
    G.recompute_lengths_and_angles()
    cc = ec.classifiers.CountingClassifier(ec.classifiers.lambda_classifier(lambda f: f.area()//0.0001)())
    for f in G.faces:
        # over = 0
        # under = 0
        # for e in f.halfedge_iter():
        #     print(e.attributes)
        #f['color_key'] = over / (over + under)
        f['color_key'] = cc.classify(f)
    G.show(**render_settings)

show_folded(G, TOP)
show_folded(G, BOTTOM)
