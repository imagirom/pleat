from .half import HalfEdgeGraph, IdObject
from time import time


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