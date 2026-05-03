"""Tests for ``eucare.colorization``."""

from __future__ import annotations

from eucare.classifiers import congruency_classifier
from eucare.colorization import colorize, congruency_colorize
from eucare.example_graphs import rosette


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
