"""Assign color keys to graph faces based on classifiers."""

from .classifiers import congruency_classifier


def colorize(graph, classifier, key='color_key'):
    """Assign a color key to each face using the given classifier."""
    for f in graph.faces:
        f[key] = classifier.classify(f)


def congruency_colorize(graph, **kwargs):
    """Colorize faces by polygon congruence."""
    colorize(graph, congruency_classifier(), **kwargs)
