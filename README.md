# Eucare

**Geometric tilings, Conway operators, and origami crease patterns.**

Eucare is a Python library for constructing, manipulating, and visualizing geometric tilings across Euclidean, hyperbolic, and spherical geometries. It provides a suite of Conway topological operators and can generate origami crease patterns with computed folded states.

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

```python
from eucare.example_tilesets import platonic
from eucare.example_graphs import from_tiles
from eucare.conway import ambo_graph

# Build a hexagonal tiling and apply the ambo operator
G = from_tiles(platonic(6), rings=4)
G2 = ambo_graph()(G, delete_on_border=True)
G2.recompute_lengths_and_angles()
G2.show()
```

### Spherical and hyperbolic tilings

```python
from eucare.example_tilesets import curved_platonic

# Icosahedron (spherical)
G = from_tiles(curved_platonic(3, 5), rings=10)

# {7,3} tiling (hyperbolic, Poincaré disk)
G = from_tiles(curved_platonic(7, 3), rings=3)
```

### Origami crease patterns

```python
from eucare.reciprocal_figures import make_SRG, assign_this_way_by_face_z_order
from eucare.search_trees import face_bfs_tree
from eucare.overlap import fold_complete
import numpy as np

G = from_tiles(platonic(4), rings=3)

# Assign crease directions via face z-order
central = min(G.faces, key=lambda f: np.linalg.norm(f.midpoint()))
central['z_order'] = 0
for orig, dest in face_bfs_tree(central):
    dest['z_order'] = orig['z_order'] + 1
assign_this_way_by_face_z_order(G)

# Build shrink-rotate crease pattern and fold
SRG = make_SRG(G)
results = fold_complete(SRG, overlap_eps=1e-8)
```

## Architecture

The library is built around the **half-edge data structure** (DCEL):

```
HalfEdgeGraph              Topology only
  └─ InAngleHEG            + interior angles and edge lengths
      └─ GeometricHEG      + pluggable geometry backend
          └─ EuclideanPositionHEG  + 2D vertex positions
```

Key modules:

| Module | Purpose |
|--------|---------|
| `eucare.half` | Core half-edge data structure |
| `eucare.conway` | Conway topological operators |
| `eucare.example_tilesets` | Predefined Archimedean and curved tilings |
| `eucare.reciprocal_figures` | Reciprocal figures and shrink-rotate |
| `eucare.overlap` | Folding, overlap graphs, ILP face ordering |
| `eucare.geometries` | Euclidean, hyperbolic, and spherical backends |
| `eucare.rendering` | Cairo-based PNG rendering |

## Testing

```bash
uv run pytest                     # all tests
uv run pytest -m "not slow"      # skip integration tests (~7s)
uv run pytest --cov=eucare       # with coverage report
```

## Documentation

```bash
uv pip install -e ".[docs]"
mkdocs serve                      # local dev server at http://127.0.0.1:8000
mkdocs build                      # build static site to site/
```

## Development

```bash
uv run ruff check eucare tests    # lint
pre-commit install                 # install git hooks
```

GitHub Actions CI runs tests, lint, and `mkdocs build --strict` on every push and pull request (`.github/workflows/ci.yml`). Contributor guidance and architecture notes for AI coding agents live in [AGENTS.md](AGENTS.md).

## License

See [LICENSE](LICENSE) for details.
