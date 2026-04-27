# Eucare

**Geometric tilings, Conway operators, and origami crease patterns.**

Eucare is a Python library for constructing, manipulating, and visualizing geometric tilings. It supports Euclidean, hyperbolic, and spherical geometries, provides a suite of Conway topological operators, and can generate origami crease patterns with computed folded states.

## Features

- **Half-edge data structure** — efficient DCEL representation for planar graphs
- **Tiling construction** — all Archimedean tilings, Platonic solids, hyperbolic tilings
- **Conway operators** — dual, ambo, truncate, kis, join, gyro, starify, and more
- **Three geometries** — Euclidean, Poincaré disk (hyperbolic), sphere
- **Origami pipeline** — reciprocal figures, shrink-rotate, crease assignment, folding with ILP face ordering
- **Multiple renderers** — Cairo (PNG), SVG, Matplotlib, 3D mesh (STL)

## Quick example

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

## Getting started

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Architecture](concepts/architecture.md)
- [API Reference](reference/index.md)
