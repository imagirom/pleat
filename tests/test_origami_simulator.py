"""Tests for pleat.origami_simulator: the launcher HTML and the public surface."""

from __future__ import annotations

import pleat
from pleat.example_graphs import rosette
from pleat.half import EuclideanPositionHEG
from pleat.origami_simulator import (
    _button_html,
    _iframe_html,
    _page_html,
    origami_simulator,
    origami_simulator_button,
)
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY


def _creased_rosette():
    G = EuclideanPositionHEG(other=rosette(n=6))
    for i, h in enumerate(h for h in G.halfedges if not h.on_border() and not h.rev.on_border()):
        a = MOUNTAIN if i % 2 == 0 else VALLEY
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = a
    return G


def test_page_embeds_fold_and_suppresses_default():
    html = _page_html(_creased_rosette())
    assert "importFold" in html
    assert '"edges_assignment"' in html  # the FOLD JSON is embedded
    assert "origamisimulator.org/?model=" in html  # empty ?model= => no waterbomb race


def test_page_enlarges_via_fullscreen_not_popup():
    html = _page_html(_creased_rosette())
    assert "requestFullscreen" in html
    assert "window.open" not in html


def test_iframe_carries_page_in_srcdoc():
    html = _iframe_html(_creased_rosette(), height=555)
    assert "<iframe" in html and "srcdoc=" in html
    assert "height:555px" in html
    assert "importFold" in html


def test_button_injects_iframe_on_click_without_popup():
    html = _button_html(_creased_rosette())
    assert "<button" in html
    assert "addEventListener" in html and "innerHTML" in html  # click -> inject iframe
    assert "window.open" not in html  # no popup (works in a sandboxed webview)
    assert "importFold" in html  # the iframe payload carries the pattern


def test_public_surface():
    import pleat.io

    # the OS feature lives in its own module, exposing exactly two entry points
    assert pleat.origami_simulator.__all__ == ["origami_simulator", "origami_simulator_button"]
    assert callable(origami_simulator) and callable(origami_simulator_button)
    # available as a graph method
    assert hasattr(EuclideanPositionHEG, "origami_simulator")
    # the OS names are gone from pleat.io (which is now FOLD/heg/circlepack I/O only)
    for gone in (
        "origami_simulator",
        "origami_simulator_button",
        "origami_simulator_html",
        "open_in_origami_simulator",
    ):
        assert gone not in pleat.io.__all__
