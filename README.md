# Pleat

[![CI](https://github.com/imagirom/pleat/actions/workflows/ci.yml/badge.svg)](https://github.com/imagirom/pleat/actions/workflows/ci.yml)

**Construct geometric tilings, and turn them into origami tessellations.**

<p align="center">
  <img src="docs/notebooks/images/shrink-rotate/Seven%20Flowers.jpg" alt="Seven Flowers — folded shrink-rotate tessellation" height="240" />
  <img src="docs/notebooks/images/intersecting-cylinders/Double%20Dodecagon.png" alt="Double Dodecagon — folded intersecting-cylinders tessellation" height="240" />
  <img src="docs/notebooks/images/shrink-rotate/7.4.3%20Circles.jpg" alt="7.4.3 Circles — folded shrink-rotate tessellation" height="240" />
</p>

Pleat is a Python library for constructing, manipulating, and visualizing geometric tilings across Euclidean, hyperbolic, and spherical geometries.
It can generate crease patterns for origami tessellations and corrugations using several algorithms which can be exported for printing or plotting, and can preview folded states.

## Features

- **Half-edge data structure** — efficient DCEL representation for planar graphs with topology, angles, and positions
- **Tiling construction** — all 11 Archimedean tilings, Platonic solids, hyperbolic tilings, and custom prototiles
- **Conway operators** — dual, ambo, truncate, kis, join, gyro, starify, and more (including alternating flagstones, loft, lace, expand, chamfer)
- **Three geometries** — Euclidean plane, Poincaré disk model (hyperbolic), stereographic sphere
- **Origami pipeline** — reciprocal figures → shrink-rotate → crease assignment → folding with ILP face ordering
- **Multiple renderers** — Cairo (PNG), SVG, Matplotlib, 3D mesh export (STL)

## Installation

```bash
# With uv (recommended)
uv venv --python 3.10
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

## Quick start

The whole pipeline — build a tiling, turn it into a crease pattern, preview the folded state:

```python
import numpy as np
import pleat as ec
from pleat.shrink_rotate import assign_this_way_by_bfs, shrink_rotate_pattern

# build a tiling: two rings of hexagons around a central one
G = ec.example_graphs.from_tiles(ec.example_tilesets.platonic(n=6), rings=2)

# decide which faces fold on top, then construct the crease pattern
assign_this_way_by_bfs(G, G.central_face())
CP = shrink_rotate_pattern(G, simplify_boundary=True, alpha=np.pi / 5, factor=0.5)

# fold it: preview the folded state with solved layer ordering
ec.overlap.fold_complete(CP, quiet=True).show()
```

## Documentation

The heart of the documentation is a series of Jupyter notebooks in
[`docs/notebooks/`](docs/notebooks/) (rendered directly on GitHub), starting with the
[pipeline overview](docs/index.ipynb). They cover constructing Euclidean and curved tilings,
Conway operators, styling, and the origami algorithms (shrink-rotate, intersecting cylinders,
alternating flagstones).

To browse the full documentation site locally:

```bash
uv pip install -e ".[docs]"
mkdocs serve                      # dev server at http://127.0.0.1:8000
```

## Development

```bash
uv run pytest -m "not slow"                    # tests (drop the -m flag for the full suite)
uv run --extra dev black --check pleat tests   # formatting check
pre-commit install                             # install git hooks
```

GitHub Actions CI runs tests, lint, and `mkdocs build --strict` on every push and pull request (`.github/workflows/ci.yml`). Contributor guidance and architecture notes for AI coding agents live in [AGENTS.md](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE).
