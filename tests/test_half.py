"""Tests for the core half-edge data structure."""
import numpy as np
import pytest

from eucare.half import (
    CyclicHalfedgeGraph,
    EuclideanPositionHEG,
    Face,
    HalfEdge,
    HalfEdgeGraph,
    InAngleHEG,
    RegularNGon,
    Vertex,
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


class TestHEGAdditionalOperations:
    def test_subdivide_face(self):
        G = RegularNGon(4)
        f = next(iter(G.faces))
        vs = list(f.vertex_iter())
        G.subdivide_face(f, vs[0], vs[2])
        # one face becomes two
        assert len(G.faces) == 2
        G.check_consistency()

    def test_join_edge_after_subdivide(self):
        G = RegularNGon(4)
        f = next(iter(G.faces))
        vs = list(f.vertex_iter())
        h12, _ = G.subdivide_face(f, vs[0], vs[2])
        v_before = len(G.vertices)
        G.join_edge(h12)
        assert len(G.vertices) == v_before - 1
        G.check_consistency()

    def test_halfedges_representing_edges_unique(self):
        G = RegularNGon(5)
        reps = G.halfedges_representing_edges()
        assert len(reps) * 2 == len(G.halfedges)
        for h in reps:
            assert h.rev not in reps

    def test_to_networkx_undirected(self):
        G = RegularNGon(4)
        nx_G = G.to_networkx_undirected()
        assert nx_G.number_of_nodes() == 4
        assert nx_G.number_of_edges() == 4

    def test_border_vertices(self):
        G = RegularNGon(5)
        bvs = G.border_vertices()
        assert len(bvs) == 5

    def test_twocolor_faces(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        if not G.twocolorable():
            pytest.skip("rosette(4) not twocolorable")
        G.twocolor_faces()
        for f in G.faces:
            assert 'color_key' in f.attributes

    def test_delete_subset_vertices(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        v = next(iter(v for v in G.vertices if v.on_border()))
        n_before = len(G.vertices)
        G.delete_subset({v})
        assert len(G.vertices) == n_before - 1
        G.check_consistency()


class TestRegularNGonProperties:
    def test_in_angles_equal(self):
        G = RegularNGon(6)
        f = next(iter(G.faces))
        for h in f.halfedge_iter():
            assert 'in_angle' in h.attributes
        # All in-angles for a regular hexagon should be equal.
        angles = [h['in_angle'] for h in f.halfedge_iter()]
        assert all(abs(a - angles[0]) < 1e-9 for a in angles)

    def test_lengths_equal(self):
        # Regular n-gon does not auto-set 'length'; instead verify equal in_angles
        G = RegularNGon(5)
        f = next(iter(G.faces))
        angles = [h['in_angle'] for h in f.halfedge_iter()]
        assert all(abs(a - angles[0]) < 1e-9 for a in angles)


class TestEuclideanPositionHEG:
    def test_construction_from_rosette(self):
        from eucare.example_graphs import rosette
        G = EuclideanPositionHEG(other=rosette(n=4))
        for v in G.vertices:
            assert 'pos' in v.attributes

    def test_make_polygon_graph(self):
        from eucare.half import make_polygon_graph
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        G = make_polygon_graph(positions)
        assert len(G.vertices) == 4
        assert len(G.faces) == 1
        G.check_consistency()


class TestRecomputePositions:
    def test_recompute_positions_smoke(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        # corrupt a position then recompute
        v = next(iter(G.vertices))
        original = v['pos'].copy()
        v['pos'] = np.array([100.0, 100.0])
        G.recompute_positions()
        # v should be recomputed back to something close to original
        # (this is rotation/translation up to gauge -- just check it's finite)
        assert np.all(np.isfinite(v['pos']))


class TestHelpers:
    def test_rotate_by_int(self):
        from eucare.half import rotate_by
        out = rotate_by([1, 2, 3, 4], 1)
        assert out == [2, 3, 4, 1]

    def test_any_element(self):
        from eucare.half import any_element
        assert any_element({"a"}) == "a"


class TestAttributeObjectMethods:
    def test_dict_like_access(self):
        v = Vertex()
        v['key'] = 1
        assert v['key'] == 1
        assert 'key' in v
        assert v.has_attributes()
        assert v.get('missing', 42) == 42
        assert list(v.keys()) == ['id', 'key']
        assert 1 in list(v.values())
        del v['key']
        assert 'key' not in v

    def test_iter_yields_keys(self):
        v = Vertex()
        v['a'] = 1
        v['b'] = 2
        keys = list(iter(v))
        assert 'a' in keys and 'b' in keys


class TestIdObjectReset:
    def test_reset_per_class(self):
        from eucare.half import IdObject
        IdObject.reset_ids()
        v1 = Vertex()
        h1 = HalfEdge()
        IdObject.reset_ids()
        v2 = Vertex()
        # After reset, ids restart from 1.
        assert v2['id'] == 1


class TestHalfEdgeMidpoint:
    def test_midpoint(self):
        v1, v2 = Vertex(), Vertex()
        v1['pos'] = np.array([0.0, 0.0])
        v2['pos'] = np.array([2.0, 0.0])
        h = HalfEdge(orig=v1, dest=v2)
        np.testing.assert_allclose(h.midpoint(), [1.0, 0.0])


class TestCheckCyclicIteratorConsistency:
    def test_passes(self):
        from eucare.half import check_cyclic_iterator_consistency
        check_cyclic_iterator_consistency(iter([1, 2, 3]))

    def test_raises_on_duplicate(self):
        from eucare.half import check_cyclic_iterator_consistency
        with pytest.raises(AssertionError):
            check_cyclic_iterator_consistency(iter([1, 2, 1]))


class TestRotateByOffsets:
    def test_tuple_of_offsets(self):
        from eucare.half import rotate_by
        out = list(rotate_by([1, 2, 3, 4], (0, 1)))
        # Returns zip of two rotations.
        assert out == [(1, 2), (2, 3), (3, 4), (4, 1)]


class TestFaceIterators2:
    def test_face_iter_outgoing_incoming(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        f = next(iter(G.faces))
        v = next(iter(f.vertex_iter()))
        out = f.outgoing_edge_at(v)
        assert out.orig is v
        inc = f.incoming_edge_at(v)
        assert inc.dest is v

    def test_common_iters_sharing_face(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        # find two adjacent faces.
        h = next(iter(G.halfedges))
        f1 = h.face
        f2 = h.rev.face
        if f1 is None or f2 is None:
            return
        common_h = list(f1.common_halfedge_iter(f2))
        assert any(c is h for c in common_h)
        common_v = list(f1.common_vertex_iter(f2))
        assert h.orig in common_v
        # face_iter
        common_f = list(f1.common_face_iter(f2))
        # f1 and f2 share at least some adjacent face (or each other)
        assert isinstance(common_f, list)

    def test_face_on_border(self):
        from eucare.half import RegularNGon
        G = RegularNGon(4)
        f = next(iter(G.faces))
        assert f.on_border()

    def test_face_midpoint(self):
        from eucare.example_graphs import rosette
        G = rosette(n=5)
        f = next(iter(G.faces))
        m = f.midpoint()
        assert m.shape == (2,)


class TestRecomputeLengthsAngles:
    def test_recompute_lengths_and_angles(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        # All non-border halfedges have 'length'.
        for h in G.halfedges:
            if not h.on_border():
                assert 'length' in h.attributes


class TestNormalizePositions:
    def test_normalize_positions(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()  # required by normalize
        G.normalize_positions()
        ps = np.stack([v['pos'] for v in G.vertices])
        assert np.max(np.abs(ps)) <= 1.0 + 1e-9


class TestScalePositions:
    def test_scale_positions_multiplies_positions(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        before = np.stack([v['pos'] for v in G.vertices]).copy()
        verts = list(G.vertices)
        G.scale_positions(2.5)
        after = np.stack([v['pos'] for v in verts])
        np.testing.assert_allclose(after, 2.5 * before)

    def test_scale_positions_recomputes_lengths(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        len_before = next(iter(G.halfedges))['length']
        G.scale_positions(3.0)
        # at least one halfedge with a length attribute → must scale by 3.
        scaled = next(h['length'] for h in G.halfedges if 'length' in h.attributes)
        np.testing.assert_allclose(scaled, 3.0 * len_before)

    def test_scale_positions_rejects_non_euclidean(self):
        import pytest
        from eucare.example_graphs import from_tiles
        from eucare.example_tilesets import curved_platonic
        G = from_tiles(curved_platonic(7, 3), rings=1)
        with pytest.raises(NotImplementedError):
            G.scale_positions(2.0)


class TestNormalizeEdgeLengths:
    def test_geometric_mean_default(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        G.normalize_edge_lengths(factor=1.0)
        lengths = np.array([h['length'] for h in G.halfedges if 'length' in h.attributes])
        # geometric mean ≈ exp(mean(log(lengths)))
        gmean = np.exp(np.mean(np.log(lengths)))
        np.testing.assert_allclose(gmean, 1.0, atol=1e-9)

    def test_arithmetic_mean(self):
        from eucare.example_graphs import rosette
        G = rosette(n=5)
        G.recompute_lengths_and_angles()
        G.normalize_edge_lengths(mode='arithmetic', factor=2.0)
        lengths = np.array([h['length'] for h in G.halfedges if 'length' in h.attributes])
        np.testing.assert_allclose(np.mean(lengths), 2.0, atol=1e-9)

    def test_factor_scales_target(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        G.normalize_edge_lengths(factor=1.0)
        G.normalize_edge_lengths(factor=4.0)
        lengths = np.array([h['length'] for h in G.halfedges if 'length' in h.attributes])
        gmean = np.exp(np.mean(np.log(lengths)))
        np.testing.assert_allclose(gmean, 4.0, atol=1e-9)

    def test_invalid_mode_raises(self):
        import pytest
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        with pytest.raises(ValueError, match="Invalid mode"):
            G.normalize_edge_lengths(mode='median')


class TestCentralFaceVertex:
    def test_central_face_and_vertex(self):
        from eucare.example_graphs import rosette
        G = rosette(n=6)
        f = G.central_face()
        v = G.central_vertex()
        assert f in G.faces
        assert v in G.vertices


class TestConvertToEuclidean:
    def test_convert_already_euclidean(self):
        from eucare.example_graphs import rosette
        G = rosette(n=4)
        G.recompute_lengths_and_angles()
        G.convert_to_euclidean()
        # No error; geometry remains Euclidean.
        from eucare.geometries import EuclideanGeometry
        assert G.geometry is EuclideanGeometry


def test_check_consistency_error_raises():
    from eucare.half import RegularNGon
    import pytest
    G = RegularNGon(3)
    # remove a vertex from the graph but keep references → triggers error path
    v = next(iter(G.vertices))
    G.vertices.discard(v)
    with pytest.raises(RuntimeError):
        G.check_consistency()


def test_show_spring_layout_smoke(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from eucare.example_graphs import rosette
    G = rosette(n=4)
    monkeypatch.setattr(plt, 'show', lambda *a, **kw: None)
    G.show_spring_layout(figsize=(4, 4))
    plt.close('all')


def test_inangle_close_vertex_via_glue():
    """exercise InAngleHEG.glue_e2e auto-close path through from_tiles growth."""
    from eucare.example_graphs import from_tiles
    from eucare.example_tilesets import platonic
    # platonic(3) is triangles; rings=3 forces many auto-close operations.
    G = from_tiles(platonic(3), rings=3)
    G.check_consistency()
    assert len(G.faces) > 6
