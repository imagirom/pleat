"""Assign per-face colour keys via :class:`~eucare.classifiers.Classifier` instances.

A *colour key* is any hashable used by renderers to look up a colour;
:func:`colorize` writes one to ``face['color_key']`` for every face.
"""

from __future__ import annotations

from typing import Any

from .classifiers import Classifier, congruency_classifier
from .half import HalfEdgeGraph


def colorize(graph: HalfEdgeGraph, classifier: Classifier, key: str = "color_key") -> None:
    """Assign ``face[key] = classifier.classify(face)`` for every face in *graph*."""
    for f in graph.faces:
        f[key] = classifier.classify(f)


def congruency_colorize(graph: HalfEdgeGraph, **kwargs: Any) -> None:
    """Colour faces by polygon congruence (same edge lengths and interior angles)."""
    colorize(graph, congruency_classifier(), **kwargs)
