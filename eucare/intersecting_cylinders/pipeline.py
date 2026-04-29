"""Build the intersecting-cylinders crease pattern from a tiling.

See :class:`eucare.intersecting_cylinders.profiles.Profile` for the cross-section
parameter, and :func:`make_intersecting_cylinders` for the main entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import base, conway, half
from .profiles import Profile

if TYPE_CHECKING:
    from ..half import EuclideanPositionHEG

# RGB tuples used throughout the pipeline as ``color_key`` values.
_BLACK = (0.0, 0.0, 0.0)
_BLUE = (0.0, 0.0, 1.0)
_RED = (1.0, 0.0, 0.0)


def _reverse_mapping(d: dict) -> dict:
    """Return ``{v: k for k, v in d.items()}``."""
    return {value: key for key, value in d.items()}


def _from_pre(obj) -> bool:
    """True iff ``obj`` carries a ``pre_conway`` attribute (set by Conway operators)."""
    return "pre_conway" in obj.attributes


def make_intersecting_cylinders(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float = 1.0,
) -> "EuclideanPositionHEG":
    """Construct an intersecting-cylinders crease pattern from a tiling.

    Every face of ``G`` must have a (pseudo-)incenter, and the incenters of
    adjacent faces should be tangential to each other for the resulting crease
    pattern to be foldable.

    The pattern places curved triangles on the edges of ``G``: each triangle's
    flat side sits on an edge, and its apex is the incenter of one of the two
    adjacent faces (shifted in the third dimension when folded).

    Args:
        G: Input tiling. Mutated in place to record midpoints, then a working
            copy is taken before applying Conway operators.
        profile: Cross-section curve. See
            :func:`~eucare.intersecting_cylinders.profiles.circular_profile` and
            :func:`~eucare.intersecting_cylinders.profiles.spherical_profile`.
        r: Triangle scaling. ``r=1`` produces curved triangles meeting at face
            incenters; ``0 < r < 1`` keeps a downscaled copy of the original
            face and produces curved quadrilaterals instead.

    Returns:
        A new ``EuclideanPositionHEG`` whose half-edges carry ``color_key``
        attributes (red curved creases, blue/black straight creases) and
        ``curve_pos`` arrays describing the curved-fold polylines.
    """
    if not 0.0 < r <= 1.0:
        raise ValueError("r must lie in (0, 1]")

    t_arr = profile.t
    l_arr = profile.l
    shrink_factor = profile.shrink_factor

    for v in G.vertices:
        if "pre_conway" in v:
            del v["pre_conway"]

    for f in list(G.faces):
        f["midpoint"] = f.pseudo_incenter()

    if r != 1:
        expand_t = (1 - r) * shrink_factor / (1 - (1 - r) * (1 - shrink_factor))
        G = conway.expand_graph(expand_t)(G, delete_on_border=False)
        fs = [f for f in G.faces if "pre_conway" in f]
        for f in fs:
            f["midpoint"] = f.pseudo_incenter()
    else:
        fs = list(G.faces)

    G, (v_map_f, h_map_f, f_map_f) = G.copy(return_mappings=True)

    v_map = _reverse_mapping(v_map_f)
    h_map = _reverse_mapping(h_map_f)
    # f_map = _reverse_mapping(f_map_f)  # reserved for future use
    del v_map  # currently unused, kept for symmetry / future extensions

    vertex_pairs_to_halfedges = {(h.orig, h.dest): h_map[h] for h in G.halfedges}

    G = conway.lace_graph(0.5, join=True)(
        G,
        faces=[f_map_f[f] for f in fs],
        delete_inner_border=True,
        delete_on_border=False,
        copy_graph=False,
    )
    G.delete_subset([f for f in G.faces if f.any_side not in G.halfedges])
    G.check_consistency()

    vs = [v for v in G.vertices if _from_pre(v) and isinstance(v["pre_conway"], half.Vertex)]

    for v in vs:
        for h in v.outgoing_iter():
            if h.dest in vs:
                h["color_key"] = h.rev["color_key"] = _BLACK
                continue
            elif h.nex.dest in vs:
                v2 = h.nex.dest
            else:
                h["color_key"] = h.rev["color_key"] = _BLUE
                continue
            try:
                h_orig = vertex_pairs_to_halfedges[(v2["pre_conway"], v["pre_conway"])]
            except KeyError:
                continue

            p = v["pos"]
            p2 = v2["pos"]

            c = h_orig.face.midpoint()
            hc = base.project_to_line(np.stack([p, p2]), c)

            alt = c - hc
            side = hc - p

            curve = (side[:, None] * l_arr + alt[:, None] * t_arr).T + p
            extra_point = base.line_intersection(
                np.stack([p2, c]),
                np.stack([curve[-1], curve[-1] + c - p]),
            )
            curve = np.concatenate([curve, extra_point[None]])

            h.dest["pos"] = (h.dest["pos"] + c) / 2
            h["curve_pos"] = curve
            h.rev["curve_pos"] = curve[::-1]
            h.dest["pos"] = curve[-1]
            h["color_key"] = h.rev["color_key"] = _RED

    for h in G.halfedges:
        if "color_key" not in h:
            h["color_key"] = _BLUE

    return G
