import networkx as nx
import collections
import itertools
import matplotlib.pyplot as plt

# TODO: Removal of nodes / edges / faces
# TODO: merging of nodes / edges / faces
# TODO: incorporate positions (first in a general manner?)
# TODO: automatic name simplification
# TODO: incorporate 'Tiles' in some sense, such that attaching new ones is straightforward..
# TODO: performance! dual takes 2s for 50 x 50 grid! Use pycharm profiling?


class NEFGraph:
    # Node, Edge, Face Graph
    # a node in this graph (not a Node as in NEF) is called n

    def __init__(self):
        self.nef = nx.DiGraph()
        self.faces = []
        self.edges = []
        self.nodes = []

    def __getitem__(self, n):
        return self.nef.__getitem__(n)

    def _fw(self, n):
        return list(self.nef.successors(n))

    def _bw(self, n):
        return list(self.nef.predecessors(n))

    def _fw_bw(self, n):
        return [pre
                for suc in self._fw(n)
                for pre in self._bw(suc)
                if pre != n]

    def _bw_fw(self, n):
        return [suc
                for pre in self._bw(n)
                for suc in self._fw(pre)
                if suc != n]

    def n2n(self, n):
        return self._bw_fw(n)

    def n2e(self, n):
        return self._bw(n)

    def n2f(self, n):
        return self._fw(n)

    def e2n(self, n):
        return self._fw(n)

    def e2e(self, n):
        return self._fw_bw(n)

    def e2f(self, n):
        return self._bw(n)

    def f2n(self, n):
        return self._bw(n)

    def f2e(self, n):
        return self._fw(n)

    def f2f(self, n):
        return self._fw_bw(n)

    def _cycle_around(self, n): #TODO: debug
        return nx.find_cycle(self.nef.subgraph(itertools.chain(self._fw(n) + self._bw(n))).to_undirected())

    def n2f_cycle(self, n):
        with_edges = self._cycle_around(n)
        if with_edges[0][0] not in self.faces:
            with_edges = with_edges[1:]
        return [edge[0] for edge in with_edges[::2]]

    @property
    def inner_edges(self):
        return [e for e in self.edges if len(self.e2f(e)) == 2]

    @property
    def border_edges(self):
        return [e for e in self.edges if len(self.e2f(e)) == 1]

    @property
    def dangling_edges(self):
        return [e for e in self.edges if len(self.e2f(e)) == 0]

    @property
    def inner_faces(self):
        return [f for f in self.faces if len(self.f2f(f)) == len(self.f2n(f))]

    @property
    def border_faces(self):
        return [f for f in self.faces if len(self.f2f(f)) < len(self.f2n(f))]

    @property
    def inner_nodes(self):
        inner_edges = self.inner_edges
        return [n for n in self.nodes if all([e in inner_edges for e in self.n2e(n)])]

    def construct_n2n_graph(self):
        result = nx.Graph()
        result.add_edges_from([self.e2n(e) for e in self.edges])
        return result

    def construct_f2f_graph(self):
        result = nx.Graph()
        result.add_edges_from([self.e2f(e) for e in self.inner_edges])
        return result

    def construct_perimeter_graph(self, include_dangling=True):
        result = nx.Graph()
        result.add_edges_from([self.e2n(e) for e in self.border_edges])
        if include_dangling:
            result.add_edges_from([self.e2n(e) for e in self.dangling_edges])
        return result

    def add_node(self, node):
        if node in self.nodes:
            return
        self.nodes.append(node)
        self.nef.add_node(node)

    def add_nodes_from(self, nodes):
        self.nef.add_nodes_from(nodes)
        self.nodes.extend([n for n in nodes if n not in self.nodes])

    def add_edge(self, nodes, e=None):
        e = frozenset(nodes) if e is None else e
        if e in self.edges:
            return
        self.edges.append(e)
        # add the nodes
        self.add_nodes_from(nodes)
        # add the two corresponding edges to the nef
        self.nef.add_edges_from([[e, nodes[0]], [e, nodes[1]]])

    def add_edges_from(self, node_tuples, es=None):
        es = [None,] * len(node_tuples) if es is None else es
        es = [frozenset(nodes) if e is None else e
              for e, nodes in zip(es, node_tuples)]
        #TODO: parallelize
        for nodes, e in zip(node_tuples, es):
            self.add_edge(nodes, e)

    def add_face(self, nodes, f=None, add_edges=True):
        f = frozenset(nodes) if f is None else f
        if f in self.faces:
            return
        # add the edges if requested (makes sense to not request if the nodes are not in correct order)
        if add_edges:
            # TODO: stuff goes wrong if edges are already present with different names
            nodes_r = nodes[1:] + [nodes[0]]
            self.add_edges_from([(n0, n1) for n0, n1 in zip(nodes, nodes_r)])
        else:
            # add only the nodes
            self.add_nodes_from(nodes)
        self.faces.append(f)
        # add connections from nodes to face to nef
        self.nef.add_edges_from([[n, f] for n in nodes])
        # add connections from face to edges to nef
        adjacent_edges = [e
                          for e in set([n for node in nodes for n in self._bw(node)])
                          if all([n in nodes for n in list(self._fw(e))])]
        self.nef.add_edges_from([f, e] for e in adjacent_edges)
        #TODO: self.face_cycles[f] = self._find_oriented_cycle(f)

    def add_faces_from(self, faces, fs=None, add_edges=True):
        # TODO: handel fs
        if fs is not None:
            raise NotImplementedError
        for face in faces:
            self.add_face(face, add_edges=add_edges)


    #@property
    #def has_positions(self):
    #    return all('position' in self.nodes[n] for n in self.nodes)

    #def edge_midpoint(self, e):
    #    return (self.node[e[0]]['position'] + self.node[e[1]]['position']) / 2

    #def face_midpoint(self, f):
    #    return np.mean(np.stack([self.node[e[0]]['position'] for e in self.face_cycles[f]]), axis=0)

    # def _find_oriented_cycle(self, f, nodes=None, inherit_from_adjacent=True):
    #     def reversed_cycle(cycle):
    #         return list(reversed([(e[1], e[0]) for e in cycle]))
    #
    #     nodes = self.faces_to_nodes[f] if nodes is None else nodes
    #     # print(nodes)
    #     # nx.draw_networkx(self.subgraph(nodes), nx.spring_layout(self.subgraph(nodes)), with_labels=False)
    #     # plt.show()
    #     cycle = nx.find_cycle(self.subgraph(nodes))
    #     # find out if cycle goes the right way
    #     for e in self.faces_to_edges[f]:
    #         for f2 in self.faces_to_edges[e]:
    #             if f2 != f and f2 in self.face_cycles:
    #                 e = tuple(e)
    #                 other_cycle = self.face_cycles[f2]
    #                 # print(f'adjacent cycle found:\n {cycle} \n {other_cycle}')
    #                 # print(e, e in cycle, e in other_cycle)
    #                 if (e in cycle) == (e in other_cycle):
    #                     # print('returning reversed')
    #                     return reversed_cycle(cycle)
    #                 else:
    #                     # print('returning normal')
    #                     return cycle
    #     # print('no adjacent face has cycle')
    #     # look at node positions to determine orientation
    #     vertices = np.stack([self.node[e[0]]['position'] for e in cycle])
    #     com = np.mean(vertices, axis=0)
    #     vertices -= com
    #     e1 = normalize(np.cross(com, [0, 0, 1]))
    #     e2 = normalize(np.cross(com, e1))
    #     vertices = vertices @ np.stack([e1, e2], axis=-1)
    #     return cycle if poly_orientation(vertices) else reversed_cycle(cycle)
    # def remove_face(self, f):
    #     self.faces.remove(f)
    #     del self.face_cycles[f]
    #     self.faces_to_nodes.remove_node(f)
    #     self.faces_to_edges.remove_node(f)
    #
    # def remove_edge(self, n0, n1):
    #     super(FaceGraph, self).remove_edge(n0, n1)
    #
    #     faces = self.faces_to_edges[frozenset([n0, n1])]
    #     # merge adjacent faces to one big face
    #     nodes = []
    #     if len(faces) == 2:
    #         for f in faces:
    #             nodes.extend(self.faces_to_nodes[f])
    #         #    cycle = self.face_cycles[f]
    #         #   i = cycle.index(frozenset([n0, n1]))
    #         #    nodes.append(cycle[i+1:] + cycle[:i])
    #         # nodes = [e[0] for e in nodes]
    #
    #     self.faces_to_edges.remove_node(frozenset([n0, n1]))
    #
    #     for f in faces:
    #         self.remove_face(f)
    #
    #     if nodes:
    #         self.add_face(nodes, frozenset(faces), add_edges=False)
    #         f = frozenset(faces)
    #
    # def construct_faces_to_faces(self):
    #     result = nx.Graph()
    #     for e in self.edges:
    #         assert len(self.faces_to_edges[frozenset(e)]) == 2, f'edge {e} not adjacent to two faces, but to {
    #         self.faces_to_edges[frozenset(e)]}'
    #     result.add_edges_from([(*self.faces_to_edges[frozenset(e)], {'comes_from': e}) for e in self.edges])
    #     return result
    #
    # def is_valid(self):
    #     result = True
    #     for e in self.edges:
    #         e = frozenset(e)
    #         if e not in self.faces_to_edges:
    #             print(f'edge {e} has no neighboring faces!')
    #         if len(self.faces_to_edges[e]) != 2:
    #             print(f'edge {e} adjacent to {list(self.faces_to_edges[e])}')
    #             result = False
    #     for n in self.nodes:
    #         assert len(list(self[n])) > 2
    #         assert len(list(self.faces_to_nodes[n])) > 2
    #     return result
    #
    # def position(self, n):
    #     return self.node[n]['position']
    #
    # def project_to_sphere(self):
    #     for n in self.nodes:
    #         pos = self.node[n]['position']
    #         self.node[n]['position'] = pos / np.linalg.norm(pos)
    #     return self
    #
    # def to_spring_layout(self):
    #     self.add_nodes_from([(node, {'position': pos}) for node, pos in nx.spring_layout(self, dim=3).items()])
    #     return self
    #

    def dual(self):
        result = NEFGraph()
        result.add_faces_from([self.n2f_cycle(n) for n in self.inner_nodes])
        return result

        #if self.has_positions:
        #    result.add_nodes_from([(f, {'position': self.face_midpoint(f)}) for f in self.faces])
        #else:
        #result.add_nodes_from(self.faces)
        #for e in self.edges:
        #    assert len(self.faces_to_edges[frozenset(e)]) == 2, f'edge {e} not adjacent to two faces, but to {
        #    self.faces_to_edges[frozenset(e)]}'
        #result.add_edges_from([(self.faces_to_edges[frozenset(e)]) for e in self.edges], add_edges=False)
        #result.add_faces_from([(list(self.faces_to_nodes[n]), n) for n in self.nodes], add_edges=False)

    #
    # def ambo(self):
    #     result = FaceGraph()
    #     if self.has_positions:
    #         result.add_nodes_from([(frozenset(e), {'position': self.edge_midpoint(e)}) for e in self.edges])
    #     else:
    #         result.add_nodes_from([frozenset(e) for e in self.edges])
    #
    #     for f in self.faces:
    #         c = self.face_cycles[f]
    #         c = c + [c[0], ]
    #         result.add_edges_from([(frozenset(e0), frozenset(e1)) for e0, e1 in zip(c[:-1], c[1:])])
    #
    #     result.add_faces_from([(self.faces_to_edges[f], f) for f in self.faces], add_edges=False)
    #     result.add_faces_from([[frozenset([n, m]) for m in self[n]] for n in self.nodes], add_edges=False)
    #     return result
    #
    # def kis(self):
    #     result = FaceGraph()
    #     result.add_nodes_from([(n, self.node[n]) for n in self.nodes])
    #     result.add_nodes_from([(f, {'position': self.face_midpoint(f)}) for f in self.faces])
    #     # result.add_edges_from([(f, n) for f in self.faces for n in self.faces_to_nodes[f]])
    #     # result.add_edges_from(self.edges)
    #     for f in self.faces:
    #         for e in self.face_cycles[f]:
    #             result.add_face([f, e[0], e[1]], (f, frozenset(e)))
    #     return result
    #
    # def gyro(self):
    #     result = FaceGraph()
    #     # original nodes
    #     result.add_nodes_from([(n, self.node[n]) for n in self.nodes])
    #     # face centers
    #     result.add_nodes_from([(f, {'position': self.face_midpoint(f)}) for f in self.faces])
    #
    #     # nodes to one side of edge
    #     result.add_nodes_from(
    #         [((frozenset(e), e[0]), {'position': 2 / 3 * self.position(e[0]) + 1 / 3 * self.position(e[1])})
    #          for e in self.edges])
    #     # nodes to other side of edge
    #     result.add_nodes_from(
    #         [((frozenset(e), e[1]), {'position': 1 / 3 * self.position(e[0]) + 2 / 3 * self.position(e[1])})
    #          for e in self.edges])
    #     # faces
    #     for f in self.faces:
    #         cycle = self.face_cycles[f]
    #         cycle_r = cycle[1:] + [cycle[0]]
    #         for e0, e1 in zip(cycle, cycle_r):
    #             # e0[1] == e1[0] adjacent edges
    #             assert e0[1] == e1[0]
    #             result.add_face([f, (frozenset(e0), e0[1]), e0[1], (frozenset(e1), e0[1]), (frozenset(e1), e1[1])],
    #                             (f, e0[1]), add_edges=True)
    #     return result
    #
    # def loft(self, faces=None):
    #     faces = self.faces if faces is None else faces
    #     result = FaceGraph()
    #     # original nodes
    #     result.add_nodes_from([(n, self.node[n]) for n in self.nodes])
    #     result.add_edges_from([e for e in self.edges])
    #     for f in self.faces:
    #         if f in faces:
    #             result.add_nodes_from([((f, n), {'position': 1 / 2 * self.position(n) + 1 / 2 * self.face_midpoint(f)})
    #                                    for n in self.faces_to_nodes[f]])
    #             result.add_face([(f, e[0]) for e in self.face_cycles[f]])
    #             result.add_faces_from([e[0], (f, e[0]), (f, e[1]), e[1]]
    #                                   for e in self.face_cycles[f])
    #         else:
    #             result.add_face(f, add_edges=False)
    #     return result
    #
    # def chamfer(self):
    #     result = self.loft()
    #     for e in self.edges:
    #         result.remove_edge(*e)
    #     return result
    #
    # def _direct_join(self):
    #     result = FaceGraph()
    #     result.add_nodes_from([(n, self.node[n]) for n in self.nodes])
    #     result.add_nodes_from([(f, {'position': self.face_midpoint(f)}) for f in self.faces])
    #     result.add_edges_from([(f, n) for f in self.faces for n in self.faces_to_nodes[f]])
    #     for e in self.edges:
    #         f0, f1 = list(self.faces_to_edges[frozenset(e)])
    #         result.add_face([e[0], f0, e[1], f1], frozenset(e))
    #     return result
    #
    # def join(self):
    #     return self.ambo().dual()
    #
    # def snub(self):
    #     return self.gyro().dual()
    #
    # def meta(self):
    #     return self.ambo().dual().kis()
    #
    # def bevel(self):
    #     return self.ambo().join().kis().dual()
    #
    # def truncate(self):
    #     return self.dual().kis().dual()
    #
    # def expand(self):
    #     return self.ambo().ambo()
    #
    # def ortho(self):
    #     return self.ambo().ambo().dual()
    #
    # def zip(self):
    #     return self.kis().dual()
    #
    # def needle(self):
    #     return self.dual().kis()
    #
    # def show(self, **kwargs):
    #     nx.draw_networkx(self, nx.spring_layout(self), **kwargs)
    #     plt.show()
    #     nx.draw_networkx(self.faces_to_edges, nx.spring_layout(self.faces_to_edges), **kwargs)
    #     plt.title('faces to edges')
    #     plt.show()
    #     nx.draw_networkx(self.faces_to_nodes, nx.spring_layout(self.faces_to_nodes), **kwargs)
    #     plt.title('faces to nodes')
    #     plt.show()
