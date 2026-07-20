"""Build a shrink-rotate crease pattern from a tiling.

This is the high-level pipeline:

1. Compute the reciprocal figure of ``G`` (per-face rotation centers).
2. Apply the :func:`pleat.conway.shrink_rotate_graph` Conway operator to
   subdivide each face.
3. Rotate and scale each twist face around its rotation center.
4. (Optional) assign mountain/valley creases based on the
   :data:`pleat.shrink_rotate.crease_orientation.THIS_WAY` mark.

The single public entry point is :func:`shrink_rotate_pattern`.
"""

from __future__ import annotations

import logging

import numpy as np

from ..base import rotation_matrix
from ..conway import shrink_rotate_graph
from ..flat_foldable import max_kawasaki_sum
from ..half import EuclideanPositionHEG, GeometricHEG, HalfEdgeGraph
from ..overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY, color_creases
from ..utils import invert_mapping
from .crease_orientation import THIS_WAY
from .reciprocal_figures import reciprocal_figure

logger = logging.getLogger(__name__)


def shrink_rotate_pattern(
    G: GeometricHEG,
    alpha: float = np.pi / 5,
    factor: float = 0.5,
    *,
    assign_creases: bool = True,
    simplify_boundary: bool = True,
    **reciprocal_figure_kwargs,
) -> EuclideanPositionHEG:
    """Build a shrink-rotate crease pattern from tiling *G*.

    Each face of *G* is subdivided by the shrink-rotate Conway operator;
    the central polygon of every face is rotated by *alpha* and scaled by
    *factor* around the corresponding reciprocal-figure center.

    Parameters
    ----------
    G:
        Input tiling. If it does not yet have ``'reciprocal_pos'``
        attributes, the reciprocal figure is computed (and cached).
    alpha, factor:
        Shrink-rotate parameters in radians and as a ratio. The default
        ``alpha=π/5, factor=0.5`` is a generic flat-foldable choice.
    assign_creases:
        When True (the default), additionally call
        :func:`assign_shrink_rotate_creases`, populate ``'color_key'``
        attributes for rendering, and emit a Kawasaki-sum sanity warning.
        Set to False to obtain only the bare topology + positions.
    simplify_boundary:
        When True (and *assign_creases* is True), remove order-2 vertices
        introduced on the boundary by the operator.
    **reciprocal_figure_kwargs:
        Forwarded to :func:`reciprocal_figure` if a recomputation is
        triggered.
    """
    f0 = next(iter(G.faces))
    if "reciprocal_pos" not in f0 or len(reciprocal_figure_kwargs):
        logger.info("Calculating reciprocal figure..")
        _ = reciprocal_figure(G, **reciprocal_figure_kwargs)
        logger.info("Done with reciprocal figure.")

    SRG, (_, _, f_map) = G.copy(return_mappings=True)
    inverse_f_map = invert_mapping(f_map)  # SRG-pre_conway → G

    SRG = shrink_rotate_graph()(SRG)

    twistfaces = list(filter(lambda f: "shrink_rotate" in f.attributes, SRG.faces))
    for f in twistfaces:
        ps, vs = np.array([[v["pos"], v] for v in f.vertex_iter()], dtype=object).T
        ps = np.stack(ps)

        midpoint = np.mean(ps, axis=0, keepdims=True)
        ps = midpoint + (ps - midpoint) * 2

        for v, p in zip(vs, ps):
            v["base_pos"] = p

    for f in twistfaces:
        ps, vs = np.array([[v["base_pos"], v] for v in f.vertex_iter()], dtype=object).T
        ps = np.stack(ps)
        assert "pre_conway" in f.attributes
        f["pre_conway"] = inverse_f_map[f["pre_conway"]]
        rotation_center = f["pre_conway"]["reciprocal_pos"]
        f["rotation_center"] = rotation_center

        ps = rotation_center + (ps - rotation_center) @ rotation_matrix(alpha) * factor

        for v, p in zip(vs, ps):
            v["pos"] = p

    SRG.recompute_lengths_and_angles()

    if assign_creases:
        assign_shrink_rotate_creases(SRG)
        color_creases(SRG)

    if simplify_boundary:
        SRG.join_order_2_boundary_vertices()

    mks = max_kawasaki_sum(SRG)
    if mks > 1e-12:
        logger.warning("High max Kawasaki sum: %s", mks)

    logger.info("CP has %d edges", len(SRG.halfedges) // 2)

    return SRG


def assign_shrink_rotate_creases(SRG: HalfEdgeGraph) -> None:
    """Set MOUNTAIN/VALLEY creases on a shrink-rotate crease pattern.

    The assignment is driven by the
    :data:`~pleat.shrink_rotate.crease_orientation.THIS_WAY` halfedge
    marks on the *original* graph (i.e. on the ``'pre_conway'`` faces of
    each twist face): on each interior edge of the original tiling, the
    side carrying THIS_WAY indicates which neighbouring face lies *above*
    in the folded model.

    For each twist face this routine walks around its boundary in
    parallel with the corresponding boundary of the original face. Where
    THIS_WAY marks the original edge, the four halfedges of the
    corresponding quad in the SRG are assigned MOUNTAIN/VALLEY in a fixed
    pattern that produces a valid flat fold.

    Border halfedges are cleared of any spurious assignment, and each
    interior edge is made consistent across its two halfedges.

    The function expects ``THIS_WAY`` to be set externally — see the
    helpers in :mod:`pleat.shrink_rotate.crease_orientation` for ways to
    derive it from the geometry of the original tiling.
    """
    twistfaces = [f for f in SRG.faces if "shrink_rotate" in f.attributes]
    for f in twistfaces:
        e_twist = f.any_side
        while e_twist.rev.on_border():
            e_twist = e_twist.nex
        # find the original-graph edge corresponding to e_twist.
        e = None
        for e_orig in f["pre_conway"].halfedge_iter():
            if e_orig.rev in e_twist.rev.nex.nex.rev.face["pre_conway"].halfedge_iter():
                e = e_orig
                break
        assert e is not None
        e_twist_initial = e_twist
        while True:
            if THIS_WAY in e.attributes and not e_twist.rev.on_border():
                e_twist.rev[CREASE_ASSIGNMENT] = MOUNTAIN
                e_twist.rev.nex.nex.nex[CREASE_ASSIGNMENT] = VALLEY
                e_twist.rev.nex[CREASE_ASSIGNMENT] = MOUNTAIN
                e_twist.rev.nex.nex[CREASE_ASSIGNMENT] = VALLEY
            e = e.nex
            e_twist = e_twist.nex
            if e_twist is e_twist_initial:
                break

    # Make the assignment of e and e.rev consistent.
    for e in SRG.halfedges:
        if e.on_border() or e.rev.on_border():
            if CREASE_ASSIGNMENT in e.attributes:
                del e[CREASE_ASSIGNMENT]
        elif CREASE_ASSIGNMENT in e.attributes:
            if CREASE_ASSIGNMENT in e.rev.attributes:
                assert e[CREASE_ASSIGNMENT] == e.rev[CREASE_ASSIGNMENT]
            else:
                e.rev[CREASE_ASSIGNMENT] = e[CREASE_ASSIGNMENT]
