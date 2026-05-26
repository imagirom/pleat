"""GomJau-Hogg notation: compile tiling codes into tilesets and graphs.

The GomJau-Hogg notation describes uniform Euclidean tilings as a sequence of
construction stages separated by ``/``:

* Stage 1 places polygons starting from a seed (e.g. ``"6-3-3"`` = hex,
  triangle, triangle).
* Stage 2+ apply rotations or mirrors (e.g. ``"m30"``, ``"r(h2)"``,
  ``"r(c3)"``) to expand the placement into a full tiling.

See `Gómez-Jáuregui & Hogg, *Symmetry* 13(2), 2021
<https://www.mdpi.com/2073-8994/13/12/2376>`__.

Public API
----------

* :func:`gjh` — code → ``list[RegularEuclideanTile]`` (cached if known).
* :func:`gjh_spec` — code → :data:`TilesetSpec`.
* :func:`gjh_graph` — code → finite expanded :class:`EuclideanPositionHEG`.
* :data:`GJH_CODES` — ordered list of all codes in the cached library.
* :func:`compile_gjh_spec` — bypass cache and run parser+distiller.
"""

from __future__ import annotations
