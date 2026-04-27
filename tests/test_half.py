"""Tests for the core half-edge data structure."""
import numpy as np
import pytest
from eucare.half import (
    HalfEdge, Vertex, Face, HalfEdgeGraph, CyclicHalfedgeGraph,
    InAngleHEG, EuclideanPositionHEG, RegularNGon,
)


class TestCyclicHalfedgeGraph:
    @pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
    def test_construction(self, n):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(n)])
        assert len(G.vertices) == n
        assert len(G.halfedges) == 2 * n
        assert len(G.faces) == 1

    def test_face_order(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(5)])
        f = next(iter(G.faces))
        assert f.order() == 5

    def test_consistency(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(6)])
        G.check_consistency()

    def test_border_exists(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(4)])
        border_edges = list(G.border_edges())
        assert len(border_edges) == 4
        for e in border_edges:
            assert e.on_border()


class TestRegularNGon:
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_construction(self, n):
        G = RegularNGon(n)
        assert len(G.vertices) == n
        assert len(G.faces) == 1
        G.check_consistency()


class TestHalfEdgeGraphOperations:
    def _make_triangle(self):
        return CyclicHalfedgeGraph([Vertex() for _ in range(3)])

    def test_copy(self):
        G = self._make_triangle()
        G2 = G.copy()[0] if isinstance(G.copy(), tuple) else G.copy()
        assert len(G2.vertices) == 3
        assert len(G2.halfedges) == 6
        G2.check_consistency()

    def test_copy_with_mappings(self):
        G = self._make_triangle()
        G2, (v_map, h_map, f_map) = G.copy(return_mappings=True)
        assert len(v_map) == 3
        assert len(h_map) == 6
        assert len([k for k in f_map if k is not None]) == 1

    def test_delete_face(self):
        G = self._make_triangle()
        f = next(iter(G.faces))
        G.delete_face(f)
        assert len(G.faces) == 0
        # all halfedges should still exist (as border)
        for h in G.halfedges:
            assert h.on_border()

    def test_glue_two_triangles(self):
        G1 = CyclicHalfedgeGraph([Vertex() for _ in range(3)])
        G2 = CyclicHalfedgeGraph([Vertex() for _ in range(3)])
        e1 = next(G1.border_edge_iter())
        e2 = next(G2.border_edge_iter())
        G1.glue_graph_e2e(G2, e1, e2)
        assert len(G1.faces) == 2
        assert len(G1.vertices) == 4  # 3 + 3 - 2 shared
        G1.check_consistency()

    def test_subdivide_edge(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(3)])
        h = next(iter(G.halfedges))
        G.subdivide_edge(h)
        assert len(G.vertices) == 4
        G.check_consistency()


class TestVertexIterators:
    def test_outgoing_iter_cyclic(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(4)])
        v = next(iter(G.vertices))
        outgoing = list(v.outgoing_iter())
        assert len(outgoing) == v.order()
        # all should originate from v
        for h in outgoing:
            assert h.orig is v

    def test_face_iter(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(4)])
        v = next(iter(G.vertices))
        faces = list(v.face_iter())
        assert len(faces) == v.order()


class TestFaceIterators:
    def test_halfedge_iter(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(5)])
        f = next(iter(G.faces))
        halfedges = list(f.halfedge_iter())
        assert len(halfedges) == 5

    def test_vertex_iter(self):
        G = CyclicHalfedgeGraph([Vertex() for _ in range(5)])
        f = next(iter(G.faces))
        vertices = list(f.vertex_iter())
        assert len(vertices) == 5
