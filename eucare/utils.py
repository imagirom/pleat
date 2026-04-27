"""Miscellaneous utility functions and timing helpers."""
import logging
from .half import HalfEdgeGraph, IdObject, AttributeObject
from time import time
from collections import defaultdict

logger = logging.getLogger(__name__)


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
        logger.debug('Starting Timer %s', self["id"])
        self.last = time()
        self.rounds = []

    def round(self, msg=''):
        current = time()
        interval = current - self.last
        self.rounds.append(interval)
        logger.debug('Timer %s, Round %d (%s): %s', self["id"], len(self.rounds), msg, interval)
        self.last = time()


def print_attribute_info(objs):
    """Print info about the attributes of the AttributeObject(s) objs. Also works on HalfEdgeGraphs."""

    if isinstance(objs, HalfEdgeGraph):
        logger.info('Vertices:')
        print_attribute_info(objs.vertices)
        logger.info('Halfedges:')
        print_attribute_info(objs.halfedges)
        logger.info('Faces:')
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
    logger.info('%d Objects', len(objs))
    for key, count in sorted(counter.items()):
        logger.info("Key '%s': %d objects (%d distinct hashable values)", key, count, len(attribute_dict[key]))