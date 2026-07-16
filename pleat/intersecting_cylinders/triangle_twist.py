"""Convert curved-vertex hubs into flat-foldable triangle twists.

Useful as a post-processing step on the output of
:func:`~pleat.intersecting_cylinders.pipeline.make_intersecting_cylinders` when
some curved regions should instead be replaced with classical flat-foldable twists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import base, half

if TYPE_CHECKING:
    from ..half import EuclideanPositionHEG, Vertex

_RED = (1.0, 0.0, 0.0)
_GREEN = (0.0, 1.0, 0.0)
_BLUE = (0.0, 0.0, 1.0)
_YELLOW = (1.0, 1.0, 0.0)


def _set_color(color, *objs) -> None:
    """Assign ``color_key = color`` to every element in ``objs`` (recursive over iterables)."""
    for obj in objs:
        if isinstance(obj, (half.Face, half.HalfEdge, half.Vertex)):
            obj["color_key"] = color
            if isinstance(obj, half.HalfEdge):
                obj.rev["color_key"] = color
        else:
            _set_color(color, *obj)


def convert_to_triangle_twist(G: "EuclideanPositionHEG", v: "Vertex") -> None:
    """Replace a 6-creased vertex with a flat-foldable triangle twist.

    ``v`` must be an interior vertex where 3 curved (red) and 3 straight creases
    meet — the typical hub produced by
    :func:`~pleat.intersecting_cylinders.pipeline.make_intersecting_cylinders`.
    The graph ``G`` is mutated in place: the curved creases are deleted, three
    new edges are added forming the boundary of a flat-foldable twist, and the
    central vertex is removed.
    """
    G.delete_subset([h for h in v.outgoing_iter() if h["color_key"] == _RED])

    tasks = []
    for h in list(v.outgoing_iter()):
        v2 = h.rev.nex.nex.dest
        _set_color(_GREEN, v2)
        direction = v["pos"] - h.rev.nex.dest["pos"]
        f = h.rev.face
        pos = base.line_intersection(
            [v2["pos"], v2["pos"] + direction],
            [h.orig["pos"], h.dest["pos"]],
        )
        tasks.append((h, f, v2, pos))

    for h, f, v2, pos in tasks:
        _, v3 = G.subdivide_edge(h, pos=pos, color_key=_YELLOW)
        h2, _ = G.subdivide_face(f, v2, v3)
        _set_color(_RED, h2)

    for h in v.outgoing_iter():
        h2, _ = G.subdivide_face(h.face, h.dest, h.pre.orig)
        _set_color(_BLUE, h2)

    G.delete_subset([v])


def convert_all_to_triangle_twists(G: "EuclideanPositionHEG") -> None:
    """Replace every interior degree-6 vertex with a flat-foldable triangle twist.

    All other vertices in intersecting-cylinder crease patterns are of smaller
    degree, so this targets exactly the curved hubs.
    """
    vs = [v for v in G.vertices if v.order() == 6 and not v.on_border()]
    for v in vs:
        convert_to_triangle_twist(G, v)
