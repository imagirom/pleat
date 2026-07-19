"""Tests for pleat.io: round-trip save/load of half-edge graphs in the .heg format."""

from __future__ import annotations

import numpy as np

from pleat.example_graphs import rosette
from pleat.half import EuclideanPositionHEG, RegularNGon
from pleat.io import dict_to_graph, graph_to_dict, load_graph, save_graph


def _topology_signature(G):
    return (len(G.vertices), len(G.halfedges), len(G.faces))


def test_graph_to_dict_roundtrip_preserves_topology():
    G = RegularNGon(5)
    d = graph_to_dict(G)
    G2 = dict_to_graph(d)
    assert _topology_signature(G) == _topology_signature(G2)
    G2.check_consistency()


def test_save_load_roundtrip(tmp_path):
    G = EuclideanPositionHEG(other=rosette(n=6))
    filename = str(tmp_path / "rosette.heg")
    save_graph(filename, G)
    G2 = load_graph(filename)
    assert _topology_signature(G) == _topology_signature(G2)
    G2.check_consistency()


def test_save_load_preserves_positions(tmp_path):
    G = EuclideanPositionHEG(other=rosette(n=4))
    filename = str(tmp_path / "rose4.heg")
    save_graph(filename, G)
    G2 = load_graph(filename)
    # Positions are preserved up to floating-point round-trip.
    pos_in = sorted(tuple(np.round(v["pos"], 8)) for v in G.vertices)
    pos_out = sorted(tuple(np.round(v["pos"], 8)) for v in G2.vertices)
    assert pos_in == pos_out


def test_save_refuses_overwrite_by_default(tmp_path):
    G = RegularNGon(3)
    filename = str(tmp_path / "tri.heg")
    save_graph(filename, G)
    try:
        save_graph(filename, G)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError on second save without overwrite=True")
    save_graph(filename, G, overwrite=True)


def test_save_appends_heg_extension(tmp_path):
    G = RegularNGon(3)
    base = str(tmp_path / "noext")
    save_graph(base, G)
    assert (tmp_path / "noext.heg").exists()


def test_save_load_string_and_array_attributes(tmp_path):
    # colour_key may be a hex string (e.g. "#cc2222") or an RGBA array; both must
    # survive .heg round-trip. Regression for float()-ing a string scalar.
    G = EuclideanPositionHEG(other=rosette(n=4))
    hs = list(G.halfedges)
    hs[0]["color_key"] = "#cc2222"
    hs[1]["color_key"] = np.array([0.0, 0.0, 0.0, 0.15])
    filename = str(tmp_path / "colored.heg")
    save_graph(filename, G)
    G2 = load_graph(filename)
    G2.check_consistency()
    keys = [h.attributes.get("color_key") for h in G2.halfedges if "color_key" in h.attributes]
    assert any(isinstance(k, str) and k == "#cc2222" for k in keys)
    assert any(isinstance(k, np.ndarray) and np.allclose(k, [0, 0, 0, 0.15]) for k in keys)


def test_roundtrip_curved(tmp_path):
    """Curved tilings (complex hyperbolic and 3D spherical positions) survive a save/load cycle."""
    from pleat.example_graphs import from_tiles
    from pleat.example_tilesets import curved_platonic

    for p, q in [(7, 3), (3, 5)]:  # hyperbolic, spherical
        G = from_tiles(curved_platonic(p, q), rings=2)
        filename = str(tmp_path / f"curved_{p}_{q}.heg")
        save_graph(filename, G)
        G2 = load_graph(filename)
        assert _topology_signature(G) == _topology_signature(G2)
        G2.check_consistency()

        def pos_key(v):
            return tuple(np.round(np.atleast_1d(v["pos"]).view(np.float64), 8))

        assert sorted(map(pos_key, G.vertices)) == sorted(map(pos_key, G2.vertices))
        # complex hyperbolic positions must come back complex, not truncated to their real part
        if p == 7:
            assert all(np.iscomplexobj(v["pos"]) for v in G2.vertices)
