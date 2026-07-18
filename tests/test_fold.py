"""Tests for pleat.io.fold: FOLD round-trip, FOLD validity, and OS launcher HTML."""

from __future__ import annotations

from pleat.example_graphs import rosette
from pleat.half import EuclideanPositionHEG
from pleat.io.fold import (
    fold_to_graph,
    graph_to_fold,
    load_fold,
    open_in_origami_simulator,  # noqa: F401  (import-smoke; used in Task 3 tests)
    origami_simulator_button,
    origami_simulator_html,
    save_fold,
)
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY

VALID_ASSIGNMENTS = {"M", "V", "B", "F", "U"}


def _creased_rosette():
    """A hexagonal rosette with every interior edge creased M/V (alternating)."""
    G = EuclideanPositionHEG(other=rosette(n=6))
    interior = [h for h in G.halfedges if not h.on_border() and not h.rev.on_border()]
    for i, h in enumerate(interior):
        a = MOUNTAIN if i % 2 == 0 else VALLEY
        h[CREASE_ASSIGNMENT] = a
        h.rev[CREASE_ASSIGNMENT] = a
    return G


def _undirected_counts(G):
    """(#border, #interior) undirected edges."""
    seen, border, interior = set(), 0, 0
    for h in G.halfedges:
        if h in seen:
            continue
        seen.add(h)
        seen.add(h.rev)
        if h.on_border() or h.rev.on_border():
            border += 1
        else:
            interior += 1
    return border, interior


def _crease_multiset(g):
    seen, out = set(), []
    for h in g.halfedges:
        if h in seen:
            continue
        seen.add(h)
        seen.add(h.rev)
        out.append(h.attributes.get(CREASE_ASSIGNMENT, 0))
    return sorted(out)


def test_graph_to_fold_is_valid_fold():
    G = _creased_rosette()
    fold = graph_to_fold(G)
    n_border, n_interior = _undirected_counts(G)

    assert fold["file_spec"] == 1.2
    assert fold["file_creator"] == "pleat"
    assert fold["frame_classes"] == ["creasePattern"]

    n_v = len(fold["vertices_coords"])
    n_e = len(fold["edges_vertices"])
    assert n_e == n_border + n_interior
    assert len(fold["edges_assignment"]) == n_e
    assert len(fold["edges_foldAngle"]) == n_e
    assert len(fold["faces_vertices"]) == len(G.faces)

    for a in fold["edges_assignment"]:
        assert a in VALID_ASSIGNMENTS
    for ang in fold["edges_foldAngle"]:
        assert ang is None or -180.0 <= ang <= 180.0
    for u, v in fold["edges_vertices"]:
        assert 0 <= u < n_v and 0 <= v < n_v

    # every interior edge was creased, so no "U"; border edges are all "B"
    assert fold["edges_assignment"].count("B") == n_border
    assert fold["edges_assignment"].count("M") + fold["edges_assignment"].count("V") == n_interior
    assert "U" not in fold["edges_assignment"]
    assert set(fold["edges_assignment"]) == {"M", "V", "B"}


def test_fold_roundtrip_preserves_topology_and_creases():
    G = _creased_rosette()
    G2 = fold_to_graph(graph_to_fold(G))
    G2.check_consistency()
    assert (len(G.vertices), len(G.halfedges), len(G.faces)) == (
        len(G2.vertices),
        len(G2.halfedges),
        len(G2.faces),
    )
    assert _crease_multiset(G) == _crease_multiset(G2)


def test_save_load_fold_roundtrip(tmp_path):
    G = _creased_rosette()
    save_fold(str(tmp_path / "rose"), G)
    assert (tmp_path / "rose.fold").exists()
    G2 = load_fold(str(tmp_path / "rose.fold"))
    G2.check_consistency()
    assert len(G.faces) == len(G2.faces)


def test_fold_to_graph_rejects_faceless_frame():
    try:
        fold_to_graph({"vertices_coords": [[0, 0], [1, 0]], "edges_vertices": [[0, 1]]})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a FOLD frame without faces_vertices")


def test_origami_simulator_html_embeds_fold_and_importfold():
    html = origami_simulator_html(_creased_rosette())
    assert "importFold" in html
    assert "origamisimulator.org" in html
    assert "<iframe" in html
    assert '"edges_assignment"' in html  # the FOLD JSON is embedded
    # the embedded FOLD JSON must not contain "</" (would break out of <script>)
    from pleat.io.fold import _fold_json

    assert "</" not in _fold_json(_creased_rosette())


def test_origami_simulator_button_repr_html():
    html = origami_simulator_button(_creased_rosette())._repr_html_()
    assert "importFold" in html
    assert "<button" in html
    assert "origamisimulator.org" in html


def test_origami_simulator_iframe_html():
    from pleat.io.fold import _origami_simulator_iframe_html

    html = _origami_simulator_iframe_html(_creased_rosette(), height=555)
    assert "<iframe" in html and "srcdoc=" in html
    assert "height:555px" in html
    # the embedded page (escaped into srcdoc) still carries the handshake + suppressed default
    assert "importFold" in html
    assert "origamisimulator.org/?model=" in html


def test_origami_simulator_html_uses_fullscreen_not_popup():
    # the enlarge button must use the Fullscreen API (works in sandboxed webviews),
    # not window.open (blocked there)
    html = origami_simulator_html(_creased_rosette())
    assert "requestFullscreen" in html
    assert "window.open" not in html
