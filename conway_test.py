from eucare.half import EuclideanPositionHEG, IdObject, HalfEdgeGraph
from eucare.conway import *
from eucare.example_graphs import *
from eucare.example_tilesets import *
from eucare.classifiers import congruency_classifier
from copy import deepcopy
from matplotlib import pyplot as plt
IdObject.reset_ids()

tiles = pgg_2x()

render_settings = dict(scale=100, line_width=0.04, face_inset=0.0,
                       render_edges=True, render_vertices=False, render_faces=False)

G = from_tiles(t_4_6_12(), 5)

#G.show(**render_settings, block=False)
for op in [ambo_graph(), gyro_graph(), starify_graph()]: #, join_graph(), gyro_graph()]:
    G = op(G, delete_on_border=True)
    G.recompute_lengths_and_angles()

    for f in G.faces:
        f['color_key'] = congruency_classifier.classify(f)
    G.show(**render_settings, block=True)


plt.show()
#ambo_graph()(join_graph()(join_graph()(G))).show(**render_settings)

# for op_number, op in enumerate([join_graph(), kis_graph(), ambo_graph(), truncate_graph(), dual_graph(), gyro_graph()]):
#     G = EuclideanPositionHEG(eps=1e-6, other=tiles[0].make_graph(add_positions=True)[0])
#     for i in range(5):
#         if i == 0:
#             faces = frozenset(G.faces)
#         for h in G.border_edges():
#             if h.on_border() and h in G.halfedges:
#                 G.execute_edge_instruction(h)
#     if op_number == 0:
#         G.show(**render_settings)
#
#     g = G
#     print(g.order),
#     g = op(g, [f for f in g.faces if f.order() == 12])
#     print(g.order)
#     G.show(**render_settings)
#     #g.check_consistency()