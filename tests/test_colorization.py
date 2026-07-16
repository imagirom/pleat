"""Tests for ``pleat.colorization``."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pleat.classifiers import congruency_classifier
from pleat.colorization import (
    EDGE_PRESETS,
    FACE_PRESETS,
    VERTEX_PRESETS,
    colorize,
    congruency_colorize,
    is_color,
    resolve_colors,
)
from pleat.example_graphs import from_tiles, rosette
from pleat.example_tilesets import t_4_6_12
from pleat.rendering import CairoRenderer

# --- existing helper tests ------------------------------------------------


def test_colorize_assigns_color_keys():
    G = rosette(n=6)
    colorize(G, congruency_classifier())
    for f in G.faces:
        assert "color_key" in f.attributes


def test_congruency_colorize_uses_alternate_key():
    G = rosette(n=4)
    congruency_colorize(G, key="my_color")
    for f in G.faces:
        assert "my_color" in f.attributes


# --- preset dispatch ------------------------------------------------------


def _t_4_6_12_graph():
    G = from_tiles(t_4_6_12(), rings=1)
    G.recompute_lengths_and_angles()
    return G


def test_face_preset_congruency_dispatches():
    G = _t_4_6_12_graph()
    colors = resolve_colors(G.faces, "congruency", "tab10", FACE_PRESETS)
    assert len(colors) == len(G.faces)


def test_face_preset_order_dispatches():
    G = _t_4_6_12_graph()
    colors = resolve_colors(G.faces, "order", "tab10", FACE_PRESETS)
    # t_4_6_12 has squares (4), hexagons (6), dodecagons (12) → 3 distinct face orders.
    distinct = {tuple(c) for c in colors.values()}
    assert len(distinct) == 3


def test_edge_preset_length_dispatches():
    G = rosette(n=6)
    G.recompute_lengths_and_angles()
    colors = resolve_colors(G.halfedges, "length", "tab10", EDGE_PRESETS)
    assert len(colors) == len(list(G.halfedges))


def test_edge_preset_orientation_dispatches():
    G = rosette(n=4)
    G.recompute_lengths_and_angles()
    colors = resolve_colors(G.halfedges, "orientation", "tab10", EDGE_PRESETS)
    # A square rosette has 2 distinct orientations (mod π): horizontal and vertical sides.
    distinct = {tuple(c) for c in colors.values()}
    assert len(distinct) == 2


def test_vertex_preset_order_dispatches():
    G = rosette(n=6)
    colors = resolve_colors(G.vertices, "order", "tab10", VERTEX_PRESETS)
    assert len(colors) == len(G.vertices)


def test_unknown_preset_raises():
    G = rosette(n=4)
    with pytest.raises(ValueError, match="unknown preset"):
        resolve_colors(G.faces, "no_such_preset", "tab10", FACE_PRESETS)


# --- callable form --------------------------------------------------------


def test_callable_color_by():
    G = _t_4_6_12_graph()
    colors = resolve_colors(G.faces, lambda f: f.order(), "tab10", FACE_PRESETS)
    distinct = {tuple(c) for c in colors.values()}
    assert len(distinct) == 3


# --- literal-colour priority ---------------------------------------------


def test_literal_color_key_wins_over_color_by():
    G = rosette(n=5)
    G.recompute_lengths_and_angles()
    fixed = (1.0, 0.0, 0.0, 1.0)
    for f in G.faces:
        f["color_key"] = fixed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the all-have-key warning is asserted elsewhere
        colors = resolve_colors(G.faces, "congruency", "tab10", FACE_PRESETS)
    for f in G.faces:
        assert np.allclose(colors[f], fixed)


def test_hashable_color_key_uses_palette_without_color_by():
    G = rosette(n=6)
    for i, f in enumerate(G.faces):
        f["color_key"] = "cls_a" if i % 2 == 0 else "cls_b"
    colors = resolve_colors(G.faces, None, "tab10", FACE_PRESETS)
    distinct = {tuple(c) for c in colors.values()}
    assert len(distinct) == 2  # two classes → two palette slots, no random collisions


# --- warning behaviour ---------------------------------------------------


def test_warns_when_color_by_has_nothing_to_do():
    G = rosette(n=5)
    G.recompute_lengths_and_angles()
    for f in G.faces:
        f["color_key"] = (0.2, 0.4, 0.6)
    with pytest.warns(UserWarning, match="classifier had no effect"):
        resolve_colors(G.faces, "congruency", "tab10", FACE_PRESETS)


# --- overflow: >10 classes ------------------------------------------------


def test_continuous_cmap_spans_full_range():
    # viridis is a 256-entry ListedColormap; n=5 classes should sample evenly across
    # the gradient (dark purple → yellow), not pack at the start.
    G = rosette(n=6)
    for i, f in enumerate(G.faces):
        f["color_key"] = f"cls_{i}"
    colors = resolve_colors(G.faces, None, "viridis", FACE_PRESETS)
    arr = np.array(list(colors.values()))
    # Sorted by frequency (all equal here, so iteration order); first ≈ viridis(0),
    # last ≈ viridis(1). Endpoints should be visibly different.
    assert np.linalg.norm(arr[0, :3] - arr[-1, :3]) > 0.5
    # First should be on the dark end, last on the bright end (viridis luminance grows).
    assert arr[0, :3].sum() < arr[-1, :3].sum()


def test_continuous_cmap_passed_via_kwarg():
    # End-to-end: viridis works through the renderer kwarg path.
    G = from_tiles(t_4_6_12(), rings=1)
    G.recompute_lengths_and_angles()
    rendering = CairoRenderer(
        width=128,
        height=128,
        face_color_by="congruency",
        face_cmap="viridis",
    ).render_graph(G)
    assert len(rendering.png_bytes) > 1000


def test_cyclic_cmap_avoids_endpoint_collision():
    # hsv(0) == hsv(1) (both red); with n=4 the sampler must use the half-open
    # interval so the last colour is distinct from the first.
    G = rosette(n=6)
    for i, f in enumerate(list(G.faces)[:4]):
        f["color_key"] = f"cls_{i}"
    colors = resolve_colors(list(G.faces)[:4], None, "hsv", FACE_PRESETS)
    arr = np.array(list(colors.values()))
    assert np.linalg.norm(arr[0, :3] - arr[-1, :3]) > 0.3


def test_overflow_above_tab10_size_distinct_colors():
    # 15 faces each with a unique class index → 15 distinct palette colours via hsv fallback.
    G = from_tiles(t_4_6_12(), rings=2)
    G.recompute_lengths_and_angles()
    faces = list(G.faces)[:15]
    for i, f in enumerate(faces):
        f["color_key"] = f"cls_{i}"
    colors = resolve_colors(faces, None, "tab10", FACE_PRESETS)
    distinct = {tuple(c) for c in colors.values()}
    assert len(distinct) == 15


# --- end-to-end renderer smoke test --------------------------------------


def test_renderer_face_color_by_congruency_smoke():
    G = from_tiles(t_4_6_12(), rings=1)
    G.recompute_lengths_and_angles()
    rendering = CairoRenderer(width=128, height=128, face_color_by="congruency").render_graph(G)
    # Non-trivial PNG indicates faces were filled with palette colours.
    assert len(rendering.png_bytes) > 1000


def test_is_color_basic():
    assert is_color((0.1, 0.2, 0.3))
    assert is_color([0.1, 0.2, 0.3, 0.4])
    assert is_color("#ff0044")
    assert is_color(np.array([0.1, 0.2, 0.3]))
    assert not is_color("red")
    assert not is_color((0.1, 0.2))
    assert not is_color("hello")
