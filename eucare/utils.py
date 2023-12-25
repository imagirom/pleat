from .half import HalfEdgeGraph, IdObject, AttributeObject
from time import time
from collections import defaultdict


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


class VerboseTimer(IdObject):
    def __init__(self):
        super(VerboseTimer, self).__init__()
        print(f'Starting Timer {self["id"]}')
        self.last = time()
        self.rounds = []

    def round(self, msg=''):
        current = time()
        interval = current - self.last
        self.rounds.append(interval)
        print(f'Timer {self["id"]}, Round {len(self.rounds)} ({msg}): {interval}')
        self.last = time()


def print_attribute_info(objs):
    """Print info about the attributes of the AttributeObject(s) objs. Also works on HalfEdgeGraphs."""

    if isinstance(objs, HalfEdgeGraph):
        print('Vertices:')
        print_attribute_info(objs.vertices)
        print('Halfedges:')
        print_attribute_info(objs.halfedges)
        print('Faces:')
        print_attribute_info(objs.faces)
        return

    counter = defaultdict(int)
    attribute_dict = defaultdict(set)
    for obj in objs:
        assert isinstance(obj, AttributeObject)
        for key, val in obj.attributes.items():
            try:
                attribute_dict[key].add(val)
            except TypeError:
                pass
            counter[key] += 1
    print(f'{len(objs)} Objects')
    for key, count in sorted(counter.items()):
        print(f"Key '{key}': {count} objects" + f" ({len(attribute_dict[key])} distinct hashable values)")