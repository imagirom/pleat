from .classifiers import congruency_classifier


def colorize(graph, classifier, key='color_key'):
    for f in graph.faces:
        f[key] = classifier.classify(f)


def congruency_colorize(graph, **kwargs):
    colorize(graph, congruency_classifier(), **kwargs)
