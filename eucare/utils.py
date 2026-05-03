"""Miscellaneous utility functions and timing helpers."""

from __future__ import annotations

import logging
from collections import defaultdict
from time import time
from typing import Iterable, TypeVar

from .half import AttributeObject, HalfEdge, HalfEdgeGraph, IdObject

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


def invert_mapping(mapping: dict[K, V]) -> dict[V, K]:
    """Return ``{value: key for key, value in mapping.items()}``.

    The input is assumed to be injective; otherwise some keys are dropped.
    """
    return {value: key for key, value in mapping.items()}


def random_directed_set(edges: HalfEdgeGraph | Iterable[HalfEdge]) -> set[HalfEdge]:
    """Pick exactly one half-edge from each undirected pair.

    Args:
        edges: An iterable of :class:`~eucare.half.HalfEdge` objects, or a
            :class:`~eucare.half.HalfEdgeGraph` (in which case all its
            half-edges are used).

    Returns:
        A set containing one half-edge per ``(h, h.rev)`` pair.  Iteration
        order of the input determines which side of each pair is kept; the
        result is therefore deterministic only for ordered inputs.
    """
    if isinstance(edges, HalfEdgeGraph):
        edges = edges.halfedges
    directed_edges: set[HalfEdge] = set()
    for e in edges:
        if e.rev not in directed_edges:
            directed_edges.add(e)
    return directed_edges


class VerboseTimer(IdObject):
    """Logger-backed stopwatch that prints intervals between :meth:`round` calls."""

    def __init__(self) -> None:
        super(VerboseTimer, self).__init__()
        logger.debug("Starting Timer %s", self["id"])
        self.last = time()
        self.rounds: list[float] = []

    def round(self, msg: str = "") -> None:
        """Record an interval since the last :meth:`round` (or :meth:`__init__`) call."""
        current = time()
        interval = current - self.last
        self.rounds.append(interval)
        logger.debug("Timer %s, Round %d (%s): %s", self["id"], len(self.rounds), msg, interval)
        self.last = time()


def print_attribute_info(objs: HalfEdgeGraph | Iterable[AttributeObject]) -> None:
    """Log per-attribute counts for a collection of :class:`AttributeObject` s.

    For each attribute name found on any object in ``objs``, logs the number of
    objects carrying it and the number of distinct hashable values seen. When
    given a :class:`HalfEdgeGraph`, recurses into its vertices, half-edges and
    faces.

    Args:
        objs: A :class:`HalfEdgeGraph` or any iterable of attribute-bearing
            objects (vertices, half-edges, faces).
    """
    if isinstance(objs, HalfEdgeGraph):
        logger.info("Vertices:")
        print_attribute_info(objs.vertices)
        logger.info("Halfedges:")
        print_attribute_info(objs.halfedges)
        logger.info("Faces:")
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
    logger.info("%d Objects", len(objs))
    for key, count in sorted(counter.items()):
        logger.info("Key '%s': %d objects (%d distinct hashable values)", key, count, len(attribute_dict[key]))
