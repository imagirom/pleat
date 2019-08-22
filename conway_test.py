from eucare.example_tilesets import *
from eucare.half import EuclideanPositionHEG, IdObject, HalfEdgeGraph
from eucare.conway import *
from copy import deepcopy
from matplotlib import pyplot as plt
IdObject.reset_ids()

tiles = pgg_2x()

render_settings = dict(scale='auto', line_width=0.06, face_inset=0.0,
                       render_edges=True, render_vertices=False, render_faces=True)

G = EuclideanPositionHEG(eps=1e-6, other=tiles[-1].make_graph(add_positions=True)[0])
for i in range(15):
    if i == 0:
        faces = frozenset(G.faces)
    for h in G.border_edges():
        if h.on_border() and h in G.halfedges:
            G.execute_edge_instruction(h)

ambo_graph()(dual_graph()(G)).show(**render_settings)
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