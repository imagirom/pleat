"""Alternative NEF graph representation using a NetworkX DiGraph.

Nodes, edges, and faces are all represented as nodes in a single directed
graph.  Less used than the half-edge structure; kept for legacy compatibility.
"""
import networkx as nx
import collections
import itertools
import matplotlib.pyplot as plt


class NEFGraph:
    # Node, Edge, Face Graph
    # a node in this graph (not a Node as in NEF) is called n

    def __init__(self):
        self.nef = nx.DiGraph()
        self.faces = set()
        self.edges = set()
        self.nodes = set()

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
        self.nodes.add(node)
        self.nef.add_node(node)

    def add_nodes_from(self, nodes):
        self.nef.add_nodes_from(nodes)
        [self.nodes.add(n) for n in nodes if n not in self.nodes]

    def add_edge(self, nodes, e=None):
        e = frozenset(nodes) if e is None else e
        if e in self.edges:
            return
        self.edges.add(e)
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
        self.faces.add(f)
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


    def compute_face_midpoints(self):
        raise NotImplementedError

    def dual(self):
        result = NEFGraph()
        result.add_faces_from([self.n2f_cycle(n) for n in self.inner_nodes])
        return result
