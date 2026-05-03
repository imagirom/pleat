"""Breadth-first search trees over half-edge graph elements.

Given a starting element (vertex or face) and a neighbour iterator, produce
the edge list of a BFS spanning tree.  Used to propagate orientation,
colouring, and stacking-order constraints across a graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, TypeVar

if TYPE_CHECKING:
    from .half import Face, Vertex

T = TypeVar("T")


def bfs_tree(start: T | set[T], neighbor_iter: Callable[[T], Iterable[T]]) -> list[tuple[T, T]]:
    """Compute a BFS spanning-tree edge list starting from a node or set.

    Args:
        start: A single starting node, or a set of starting nodes (forest root).
        neighbor_iter: Function mapping a node to an iterable of its neighbours.

    Returns:
        ``(parent, child)`` pairs in BFS discovery order.
    """
    boundary = start if isinstance(start, set) else {start}
    parsed = boundary.copy()
    edges: list[tuple[T, T]] = []
    while boundary:
        new_boundary: set[T] = set()
        for orig in boundary:
            for dest in neighbor_iter(orig):
                if dest not in parsed:
                    parsed.add(dest)
                    new_boundary.add(dest)
                    edges.append((orig, dest))
        boundary = new_boundary
    return edges


def face_bfs_tree(start: "Face | set[Face]") -> list[tuple["Face", "Face"]]:
    """BFS tree over faces (interior faces only; ``None`` boundary faces are skipped)."""
    return bfs_tree(start, lambda f: (f2 for f2 in f.face_iter() if f2 is not None))


def vertex_bfs_tree(start: "Vertex | set[Vertex]") -> list[tuple["Vertex", "Vertex"]]:
    """BFS tree over vertices, traversing along incident edges."""
    return bfs_tree(start, lambda v: v.vertex_iter())
