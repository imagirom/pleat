"""Smoke tests for the Cairo and svgwrite renderers."""

from __future__ import annotations

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: plt.show() must be a no-op

import numpy as np
import pytest

from eucare.example_graphs import rosette
from eucare.half import RegularNGon
from eucare.rendering import (
    CairoRenderer,
    Rendering,
    SvgwriteRenderer,
    inset_corner,
    inset_poly,
    is_color,
    random_color,
)


def test_inset_corner_degenerate_returns_b():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    # Degenerate: a == c -> v == w -> early return.
    out = inset_corner(a, b, a, 0.1)
    assert np.allclose(out, b)


def test_inset_corner_right_angle_inset():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])
    out = inset_corner(a, b, c, 0.1)
    # The inset point is offset from b by a non-zero amount.
    assert not np.allclose(out, b)
    assert np.linalg.norm(out - b) == pytest.approx(0.1 * np.sqrt(2), rel=1e-6)


def test_inset_poly_returns_same_length():
    pts = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])]
    out = inset_poly(pts, 0.05)
    assert len(out) == len(pts)


def test_random_color_shape():
    c = random_color(seed=42)
    assert c.shape == (3,)
    assert (0 <= c).all() and (c <= 1).all()


def test_is_color_true_false():
    assert is_color((0.1, 0.2, 0.3))
    assert is_color([0.1, 0.2, 0.3, 0.4])
    assert not is_color("red")
    assert not is_color((0.1, 0.2))


def test_cairo_renderer_render_graph_returns_rendering(tmp_path):
    G = rosette(n=6)
    r = CairoRenderer(width=128, height=128)
    rendering = r.render_graph(G)
    assert rendering.svg_bytes and rendering.png_bytes
    assert rendering.width == 128 and rendering.height == 128
    rendering.save(str(tmp_path / "out"))
    assert (tmp_path / "out.svg").stat().st_size > 0
    assert (tmp_path / "out.png").stat().st_size > 0


def test_cairo_renderer_writes_nothing_on_construction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    CairoRenderer(width=64, height=64)
    assert list(tmp_path.iterdir()) == []


def test_cairo_renderer_color_attributes():
    G = rosette(n=5)
    for f in G.faces:
        f["color_key"] = (0.2, 0.4, 0.6)
    for h in G.halfedges:
        h["color_key"] = (0.5, 0.5, 0.5, 0.5)
    r = CairoRenderer(width=64, height=64)
    rendering = r.render_graph(G)
    assert rendering.png_bytes


def test_cairo_renderer_dashed_delete():
    G = rosette(n=4)
    for h in G.halfedges:
        h["delete"] = True
    rendering = CairoRenderer(width=64, height=64).render_graph(G)
    assert rendering.svg_bytes


def test_cairo_renderer_vertex_color_join_delete():
    G = rosette(n=4)
    vs = list(G.vertices)
    vs[0]["join"] = True
    vs[1]["delete"] = True
    vs[2]["color_key"] = (0.1, 0.2, 0.3)
    rendering = CairoRenderer(width=64, height=64).render_graph(G)
    assert rendering.svg_bytes


def test_cairo_renderer_explicit_scale():
    G = rosette(n=6)
    rendering = CairoRenderer(width=64, height=64, scale=20.0).render_graph(G)
    assert rendering.png_bytes


def test_svgwrite_renderer_render_graph(tmp_path):
    out = tmp_path / "cp.svg"
    G = rosette(n=6)
    r = SvgwriteRenderer()
    r.render_graph(str(out), G, render_interior_and_borders=True, extra_render_keys=())
    assert out.exists()
    # Borders/interior auxiliary files were also generated.
    assert (tmp_path / "cp_borders.svg").exists()
    assert (tmp_path / "cp_interior.svg").exists()


def test_svgwrite_renderer_extra_render_key(tmp_path):
    out = tmp_path / "cp.svg"
    G = rosette(n=5)
    # Mark each edge and its reverse so the extra-render set is closed under .rev.
    for h in list(G.halfedges):
        h["drawing_edge"] = True
        h.rev["drawing_edge"] = True
    r = SvgwriteRenderer()
    r.render_graph(str(out), G, render_interior_and_borders=False)
    assert out.exists()
    # An additional file for the 'drawing_edge' subset is also produced.
    assert (tmp_path / "cpdrawing_edge.svg").exists()


def test_svgwrite_renderer_for_cutting_false_raises(tmp_path):
    out = tmp_path / "cp.svg"
    G = rosette(n=4)
    r = SvgwriteRenderer()
    with pytest.raises(NotImplementedError):
        r.render_graph(str(out), G, for_cutting=False, render_interior_and_borders=False)


def test_svgwrite_renderer_render_faces_raises(tmp_path):
    out = tmp_path / "cp.svg"
    G = rosette(n=4)
    r = SvgwriteRenderer()
    with pytest.raises(NotImplementedError):
        r.render_graph(str(out), G, render_faces=True, render_interior_and_borders=False)


def _dummy_rendering():
    return Rendering(svg_bytes=b"<svg xmlns='...'></svg>", png_bytes=b"\x89PNG\r\n", width=64, height=64)


def test_rendering_repr_svg_returns_text():
    r = _dummy_rendering()
    assert isinstance(r._repr_svg_(), str)
    assert "<svg" in r._repr_svg_()


def test_rendering_repr_png_returns_bytes():
    r = _dummy_rendering()
    assert r._repr_png_() == b"\x89PNG\r\n"


def test_rendering_save_no_extension_writes_both(tmp_path):
    r = _dummy_rendering()
    r.save(str(tmp_path / "out"))
    assert (tmp_path / "out.svg").exists()
    assert (tmp_path / "out.png").exists()


def test_rendering_save_svg_only(tmp_path):
    r = _dummy_rendering()
    r.save(str(tmp_path / "out.svg"))
    assert (tmp_path / "out.svg").exists()
    assert not (tmp_path / "out.png").exists()


def test_rendering_save_png_only(tmp_path):
    r = _dummy_rendering()
    r.save(str(tmp_path / "out.png"))
    assert (tmp_path / "out.png").exists()
    assert not (tmp_path / "out.svg").exists()


def test_rendering_save_unknown_extension_raises(tmp_path):
    r = _dummy_rendering()
    with pytest.raises(ValueError):
        r.save(str(tmp_path / "out.pdf"))


def test_rendering_show_headless_is_noop(monkeypatch):
    import matplotlib.pyplot as plt

    called = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(True))
    r = _dummy_rendering()
    assert r.show() is None
    assert called == []  # headless (Agg) backend -> early return, no display attempted


def test_geometric_render_returns_rendering():
    G = rosette(n=6)
    rendering = G.render(width=64, height=64)
    assert rendering.svg_bytes and rendering.png_bytes


def test_geometric_save_writes_both(tmp_path):
    G = rosette(n=6)
    G.save(str(tmp_path / "g"), width=64, height=64)
    assert (tmp_path / "g.svg").exists()
    assert (tmp_path / "g.png").exists()


def test_geometric_show_headless_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    G = rosette(n=6)
    assert G.show(width=64, height=64) is None
    assert list(tmp_path.iterdir()) == []


def test_multi_show_runs_in_memory(tmp_path, monkeypatch):
    from eucare.rendering import multi_show

    monkeypatch.chdir(tmp_path)
    G1 = rosette(n=4)
    G2 = rosette(n=6)
    assert multi_show([G1, G2], titles=["a", "b"], height=64) is None
    # No stray files and no disk round-trip.
    assert list(tmp_path.iterdir()) == []
