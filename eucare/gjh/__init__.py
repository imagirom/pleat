"""GomJau-Hogg notation: compile tiling codes into tilesets and graphs.

The GomJau-Hogg notation describes uniform Euclidean tilings as a sequence of
construction stages separated by ``/``:

* Stage 1 places polygons starting from a seed (e.g. ``"6-3-3"`` = hex,
  triangle, triangle).
* Stage 2+ apply rotations or mirrors (e.g. ``"m30"``, ``"r(h2)"``,
  ``"r(c3)"``) to expand the placement into a full tiling.

See `Gómez-Jáuregui & Hogg, *Symmetry* 13(12), 2021
<https://www.mdpi.com/2073-8994/13/12/2376>`__.

Public API
----------

* :func:`gjh` — code → ``list[RegularEuclideanTile]`` (cached if known, compiled otherwise).
* :func:`gjh_spec` — code → :data:`~eucare.tileset_spec.TilesetSpec`.
* :func:`gjh_graph` — code → finite expanded :class:`~eucare.half.EuclideanPositionHEG`.
* :data:`GJH_CODES` — ordered list of all codes in the cached library.
* :func:`cached_spec` — strict cache-only lookup; raises :class:`KeyError` if the code is not in :data:`GJH_CODES`.
* :func:`compile_gjh_spec` — bypass cache: always run the parser + distiller.

Example::

    import eucare as ec
    from eucare.gjh import gjh

    tiles = gjh("6-3-3")
    G = ec.example_graphs.from_tiles(tiles, rings=5)
    G.show()
"""

from __future__ import annotations

from ..half import EuclideanPositionHEG
from ..prototiles import RegularEuclideanTile
from ..tileset_spec import TilesetSpec, tileset_from_spec
from .distill import spec_from_graph
from .library import CACHED_SPECS, GJH_CODES, cached_spec
from .parser import apply_transform, compile_gjh_graph, polygon_placement

__all__ = [
    "gjh",
    "gjh_spec",
    "gjh_graph",
    "compile_gjh_spec",
    "GJH_CODES",
    "cached_spec",
    # lower-level
    "polygon_placement",
    "apply_transform",
    "compile_gjh_graph",
    "spec_from_graph",
]


def gjh_spec(code: str, bbox_size: float = 20.0) -> TilesetSpec:
    """Return the :data:`~eucare.tileset_spec.TilesetSpec` for a GJH code.

    If ``code`` is in :data:`GJH_CODES` the cached spec is returned (fast). Otherwise
    the parser and distiller are run with the given ``bbox_size``.
    """
    code = code.replace(" ", "")
    if code in CACHED_SPECS:
        return CACHED_SPECS[code]
    return compile_gjh_spec(code, bbox_size=bbox_size)


def gjh(code: str, bbox_size: float = 20.0) -> list[RegularEuclideanTile]:
    """Return a list of :class:`RegularEuclideanTile` with edge instructions wired up.

    Equivalent to :func:`eucare.tileset_spec.tileset_from_spec` applied to
    :func:`gjh_spec`. The result is ready to pass to
    :func:`eucare.example_graphs.from_tiles`.
    """
    return tileset_from_spec(gjh_spec(code, bbox_size=bbox_size))


def gjh_graph(code: str, bbox_size: float = 20.0) -> EuclideanPositionHEG:
    """Return the raw expanded :class:`EuclideanPositionHEG` for a GJH code.

    Unlike :func:`gjh`, this never consults the cache — it always runs the
    full parser pipeline. Useful for inspection, debugging, or visualising
    intermediate state during code authoring.
    """
    return compile_gjh_graph(code, bbox_size=bbox_size)


def compile_gjh_spec(code: str, bbox_size: float = 20.0) -> TilesetSpec:
    """Run the parser + distiller pipeline, bypassing the cached library."""
    G = compile_gjh_graph(code, bbox_size=bbox_size)
    return spec_from_graph(G)
