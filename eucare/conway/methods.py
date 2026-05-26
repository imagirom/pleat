"""Shorthand methods on :class:`GeometricHEG` for chained Conway operators.

Importing this module attaches one method per factory in :mod:`.factories`
to :class:`GeometricHEG`, enabling fluent chains like::

    G.ambo().dual().truncate(0.4).dual()

Each method instantiates the operator with the factory parameters and applies
it to the graph, forwarding the remaining keyword arguments to
:meth:`TopologicalConwayOperator.__call__`. Like the operator call itself,
each method mutates the graph in place (unless ``copy_graph=True``) and
returns the (possibly copied) graph so the result is chainable either way.

``faces`` may be either a set of :class:`Face` or a callable
``face -> bool`` used as a face filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..half import GeometricHEG
from .factories import (
    alternating_flagstone_graph,
    ambo_graph,
    chamfer_graph,
    dual_graph,
    expand_graph,
    flagstone_pvitelli_graph,
    goldberg2_graph,
    gyro_graph,
    join_graph,
    kis_graph,
    lace_graph,
    loft_graph,
    meta_graph,
    ortho_graph,
    shrink_rotate_graph,
    starify_graph,
    truncate_graph,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..half import Face


# --- non-parametric operators ---


def dual(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway dual operator. See :func:`dual_graph`."""
    return dual_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def kis(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway kis (raising) operator. See :func:`kis_graph`."""
    return kis_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def join(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway join operator. See :func:`join_graph`."""
    return join_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def meta(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway meta operator. See :func:`meta_graph`."""
    return meta_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def ortho(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway ortho operator. See :func:`ortho_graph`."""
    return ortho_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def ambo(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway ambo (rectification) operator. See :func:`ambo_graph`."""
    return ambo_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def goldberg2(
    self: GeometricHEG,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Goldberg-2 subdivision operator. See :func:`goldberg2_graph`."""
    return goldberg2_graph()(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


# --- parametric operators ---


def truncate(
    self: GeometricHEG,
    t: float = 1 / 2,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway truncate operator with cut depth ``t``. See :func:`truncate_graph`."""
    return truncate_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def gyro(
    self: GeometricHEG,
    g: tuple[float, float] = (1 / 4, -1 / 4),
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway gyro operator with snub point ``g``. See :func:`gyro_graph`."""
    return gyro_graph(g)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def starify(
    self: GeometricHEG,
    t: float = 1 / 3,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the starify operator with depth ``t``. See :func:`starify_graph`."""
    return starify_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def alternating_flagstone(
    self: GeometricHEG,
    t: float = 1 / 3,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the alternating flagstone operator with parameter ``t``. See :func:`alternating_flagstone_graph`."""
    return alternating_flagstone_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def shrink_rotate(
    self: GeometricHEG,
    t: float = 1 / 2,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the shrink-rotate operator with parameter ``t``. See :func:`shrink_rotate_graph`."""
    return shrink_rotate_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def loft(
    self: GeometricHEG,
    t: float = 1 / 2,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the loft operator with edge offset ``t``. See :func:`loft_graph`."""
    return loft_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def lace(
    self: GeometricHEG,
    t: float = 1 / 2,
    join: bool = False,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the lace operator with offset ``t``. See :func:`lace_graph`."""
    return lace_graph(t, join=join)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def expand(
    self: GeometricHEG,
    t: float = 1 / 2,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway expand operator with offset ``t``. See :func:`expand_graph`."""
    return expand_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def flagstone_pvitelli(
    self: GeometricHEG,
    t: float = 1 / 4,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Pvitelli flagstone operator with parameter ``t``. See :func:`flagstone_pvitelli_graph`."""
    return flagstone_pvitelli_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


def chamfer(
    self: GeometricHEG,
    t: float = 1 / 2,
    faces: "set[Face] | Callable[[Face], bool] | None" = None,
    delete_on_border: bool = True,
    delete_inner_border: bool = False,
    copy_graph: bool = False,
) -> GeometricHEG:
    """Apply the Conway chamfer operator with offset ``t``. See :func:`chamfer_graph`."""
    return chamfer_graph(t)(
        self,
        faces=faces,
        delete_on_border=delete_on_border,
        delete_inner_border=delete_inner_border,
        copy_graph=copy_graph,
    )


_METHODS = (
    dual,
    kis,
    join,
    meta,
    ortho,
    ambo,
    goldberg2,
    truncate,
    gyro,
    starify,
    alternating_flagstone,
    shrink_rotate,
    loft,
    lace,
    expand,
    flagstone_pvitelli,
    chamfer,
)
for _fn in _METHODS:
    setattr(GeometricHEG, _fn.__name__, _fn)
