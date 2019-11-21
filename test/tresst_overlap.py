import numpy as np
import eucare as ec

render_settings = dict(
    width=1000, height=1000,
    figsize=(7, 7),
    #scale=100,
    render_edges=True,
    render_faces=True,
    render_vertices=False,
    line_width=0.01,
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


def sr_graph(n=5, angle=np.pi/4, factor=0.4):
    G = ec.example_graphs.from_tiles(ec.example_tilesets.t_4_6_12(), n, vertex_based=True)
    G = ec.reciprocal_figures.shrink_rotate_graph(G, angle, factor)
    G.twocolor_faces()
    #G.show(**render_settings)
    for f in filter(lambda f: f['color_key'], G.faces):
        for e in f.halfedge_iter():
            e['in_angle'] *= -1
    G.recompute_positions()
    return G

# FIXME: edges on border do not get proper edge_groups

#G = sr_graph(factor=0.1, angle=0)
G = sr_graph()
#G.add_graph(G.copy())
print('inner edges', len(G.halfedges) / 2 - len(G.border_edges()))

#G.show(**render_settings)

G = ec.overlap.overlap_graph(G)
ec.overlap.find_face_order(G)
#for e in G.halfedges:
#    print(e['original_face_groups'])
cc = ec.classifiers.CountingClassifier(ec.classifiers.RepresentationClassifier())
for f in G.faces:
    #f['color_key'] = len(f['original_faces'])
    f['color_key'] = cc.classify(f['sorted_original_faces'][0])
    #print(f['color_key'])


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
    f['color_key'] = cc.classify(f)
G.show(**render_settings)

