from .half import HalfEdgeGraph


def invert_mapping(mapping):
    return {value: key for key, value in mapping.items()}


def random_directed_set(edges):
    if isinstance(edges, HalfEdgeGraph):
        edges = edges.halfedges
    directed_edges = set()
    for e in edges:
        if e.rev not in directed_edges:
            directed_edges.add(e)
    return directed_edges
