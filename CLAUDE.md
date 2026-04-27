# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eucare is a Python library for constructing, manipulating, and visualizing geometric tilings and origami crease patterns. It supports Euclidean, hyperbolic, and spherical geometries, and can generate folded forms, reciprocal figures, and output for 3D printing.

## Environment Setup

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"
```

Optional extras: `ilp` (PuLP solver), `notebook` (Jupyter), `threed` (meshio for STL), `torch` (PyTorch + einops).

Additional optional requirements:
- `fancy` from github.com/imagirom/fancy
- `cplex` from IBM (free for academics)

Legacy conda setup is also available via `environment.yaml`.

## Running Tests

```bash
uv run pytest                    # all tests (143 tests, ~11s)
uv run pytest -m "not slow"     # skip slow integration tests (~7s)
uv run pytest tests/test_base.py # single test file
uv run pytest -k "test_dual"    # run tests matching pattern
```

Tests are organized in `tests/`:
- `test_base.py` — geometry utilities (unit vectors, areas, intersections)
- `test_half.py` — half-edge data structure operations
- `test_tilings.py` — tiling construction in Euclidean, spherical, and hyperbolic geometries
- `test_conway.py` — Conway operators (dual, kis, ambo, truncate, join, gyro, starify)
- `test_shrink_rotate.py` — shrink-rotate origami pipeline (reciprocal figures, crease assignment, folding, overlap)
- `test_alternating_flagstones.py` — alternating flagstone and related Conway operators

## Architecture

### Core Data Structure: Half-Edge Graph (`eucare/half.py`, ~1300 lines)

The central data structure is the **half-edge data structure** (DCEL), implemented as `HalfEdgeGraph` and its subclasses. This is the backbone of the entire library.

Key class hierarchy:
- `AttributeObject` → `IdObject` → `HalfEdge`, `Vertex`, `Face` — graph elements with dict-like attribute storage
- `HalfEdgeGraph` → `CyclicHalfedgeGraph` → `InAngleHEG` → `EuclideanPositionHEG` — increasingly specialized graph types
  - `HalfEdgeGraph`: topology only (vertices, halfedges, faces as sets; gluing, deletion, copying)
  - `InAngleHEG`: adds interior angles and edge lengths
  - `EuclideanPositionHEG`: adds 2D vertex positions, epsilon-based vertex merging, position recomputation

Half-edges store `rev` (reverse), `nex` (next), `pre` (previous), `orig`/`dest` vertices, and `face`. Vertices link to `any_outgoing` half-edge. Faces link to `any_side` half-edge.

### NEF Graph (`eucare/graph.py`)

An alternative graph representation using a NetworkX DiGraph where nodes, edges, and faces are all nodes in a directed graph. Navigation uses forward/backward traversal patterns (n2e, e2f, f2n, etc.). Less used than the half-edge structure.

### Tiling Construction Pipeline

1. **ProtoTiles** (`eucare/prototiles.py`): Define tile shapes via angles, edge lengths, and labels. `EuclideanProtoTile` and `RegularEuclideanTile` compute vertex positions.
2. **Tile Sets** (`eucare/example_tilesets.py`): Predefined Archimedean and other tilings using GomJau-Hogg notation.
3. **Instructions** (`eucare/instructions.py`): `HalfEdgeInstruction` objects (e.g., `GlueTileInstruction`) describe how to attach tiles to border edges. Stored as edge attributes and executed via `execute_edge_instruction`.
4. **Growth**: Tilings grow by iterating over border edges/vertices and executing their instructions.

### Conway Operators (`eucare/conway.py`)

Topological operators (dual, ambo, truncate, kis, join, gyro, starify, etc.) that transform tilings. Implemented as `TopologicalConwayOperator` applied to `HalfEdgeGraph` faces. Each operator defines a fundamental domain as a small half-edge graph with three marked vertices (v1, vf, v2) that gets substituted into each face triangle.

### Origami / Crease Patterns

- **Reciprocal Figures** (`eucare/reciprocal_figures.py`): Computes dual crease patterns via linear algebra on edge vectors rotated 90 degrees. Used for shrink-rotate tessellation origami.
- **Overlap** (`eucare/overlap.py`): Computes the folded state — crease assignments (MOUNTAIN/VALLEY), face stacking order via ILP (using PuLP), and overlap graphs for opaque rendering.
- **Cutting** (`eucare/cutting.py`): Cuts graphs along paths for unfolding.

### Geometry Backends (`eucare/geometries/`)

Pluggable geometry implementations: `EuclideanGeometry`, `PoincareDiskModel` (hyperbolic), `SphereModel`. Used by `InAngleHEG` for distance/angle calculations and position recomputation.

### Rendering

- **Cairo** (`eucare/rendering.py`): `CairoRenderer` for high-quality PNG output with face coloring, edge rendering, and insets.
- **SVG** (`eucare/svg.py`): SVG output via `svgwrite`.
- **Matplotlib** (`eucare/plotting.py`): Simple line/polygon plotting helpers.
- **3D** (`eucare/marching_cubes.py`): Marching cubes for generating meshes (used with `meshio` for STL export).

### Other Modules

- `eucare/conversions.py`: Convert between NetworkX graphs and `EuclideanPositionHEG` (via `EHEG_from_nx`).
- `eucare/colorization.py`: Graph coloring algorithms.
- `eucare/classifiers.py`: Classify faces by congruence (edge lengths + angles) for coloring.
- `eucare/layout.py`: Position optimization for half-edge graphs.
- `eucare/io.py`: File I/O for graph formats.
- `eucare/search_trees.py`: Spatial search structures.
- `eucare/image_to_graph.py`: Convert images to graph structures.

## Development Notes

- The library is used interactively via Jupyter notebooks (see `notebooks/`). Most workflows start in a notebook.
- There is no package installer (`setup.py` / `pyproject.toml`); import `eucare` directly from the repo root.
- `test.py` is a runnable script (not a test suite) that constructs a tiling and renders it — useful for smoke testing.
- The `HalfEdgeGraph.show()` method is the quickest way to visualize a graph during development.
