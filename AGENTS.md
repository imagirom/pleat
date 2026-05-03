# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, GitHub Copilot, and similar) when working in this repository.

## Project Overview

Eucare is a Python library for constructing, manipulating, and visualizing geometric tilings and origami crease patterns. It supports Euclidean, hyperbolic, and spherical geometries, and can generate folded forms, reciprocal figures, and output for 3D printing.

## Environment Setup

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"
```

Optional extras: `docs` (MkDocs), `notebook` (Jupyter), `threed` (meshio for STL), `torch` (PyTorch + einops).

Additional optional requirements (not on PyPI):
- `fancy` from github.com/imagirom/fancy
- `cplex` from IBM (free for academics)

## Running Tests

```bash
uv run pytest                        # all tests
uv run pytest -m "not slow"         # skip slow integration tests (~7s)
uv run pytest tests/test_base.py    # single test file
uv run pytest -k "test_dual"        # run tests matching pattern
uv run pytest --cov=eucare          # with coverage report (HTML in htmlcov/)
```

Tests are organized in `tests/`:
- `test_base.py` — geometry utilities (unit vectors, areas, intersections)
- `test_half.py` — half-edge data structure operations
- `test_tilings.py` — tiling construction in Euclidean, spherical, and hyperbolic geometries
- `test_conway.py` — Conway operators (dual, kis, ambo, truncate, join, gyro, starify)
- `test_shrink_rotate.py` — shrink-rotate origami pipeline (reciprocal figures, crease assignment, folding, overlap)
- `test_alternating_flagstones.py` — alternating flagstone and related Conway operators
- `test_geometries.py` — cross-backend invariants for the three geometry backends
- `test_classifiers.py` — equivalence classifiers used by colorization
- `test_utils.py`, `test_io.py`, `test_example_graphs.py`, `test_plotting.py` — small helpers

## Linting and CI

```bash
uv run --extra dev black --check eucare tests  # formatting check
uv run --extra dev black eucare tests          # auto-format
pre-commit install                    # enable pre-commit hooks (black + standard hygiene)
```

GitHub Actions CI runs on every push/PR (see `.github/workflows/ci.yml`):
- `test` job on Python 3.10/3.11/3.12 with coverage upload
- `lint` job (`black --check`)
- `docs` job (`mkdocs build --strict`)

## Documentation

```bash
uv pip install -e ".[docs]"
DISABLE_MKDOCS_2_WARNING=true mkdocs serve    # dev server at http://127.0.0.1:8000
DISABLE_MKDOCS_2_WARNING=true mkdocs build    # static site to site/
```

API docs are auto-generated from docstrings via mkdocstrings. Notebooks in `docs/notebooks/` are rendered via mkdocs-jupyter (execute: false — outputs must be pre-saved in the notebook). The `DISABLE_MKDOCS_2_WARNING=true` flag suppresses a spurious deprecation warning injected by the `properdocs` transitive dependency.

## Architecture

### Core Data Structure: Half-Edge Graph (`eucare/half.py`)

The central data structure is the **half-edge data structure** (DCEL). This is the backbone of the entire library.

Key class hierarchy:
- `AttributeObject` → `IdObject` → `HalfEdge`, `Vertex`, `Face` — graph elements with dict-like attribute storage (`obj['key'] = value`)
- `HalfEdgeGraph` → `InAngleHEG` → `GeometricHEG` → `EuclideanPositionHEG` — increasingly specialized graph types

Half-edges store `rev` (reverse), `nex` (next), `pre` (previous), `orig`/`dest` vertices, and `face`. Vertices link to `any_outgoing` half-edge. Faces link to `any_side` half-edge.

**Important**: Most methods mutate graphs in-place and return `None`. Use `.copy()` first if you need the original.

**Global state**: `IdObject.current_ids` is class-level mutable state. Call `IdObject.reset_ids()` before constructing independent graphs in tests.

### Tiling Construction Pipeline

1. **ProtoTiles** (`prototiles.py`): Define tile shapes via angles, edge lengths, and labels.
2. **Tile Sets** (`example_tilesets.py`): Predefined Archimedean and other tilings.
3. **Instructions** (`instructions.py`): `GlueTileInstruction` objects describe how to attach tiles to border edges.
4. **Growth** (`example_graphs.py`): `from_tiles(tiles, rings=N)` grows a tiling by executing instructions on border vertices.

### Conway Operators (`conway.py`)

Topological operators that transform tilings. Each operator defines a fundamental domain as a small half-edge graph with three marked vertices (v1, vf, v2) that gets substituted into each face triangle. Call pattern: `operator_fn()(graph, delete_on_border=True)`.

### Origami / Crease Patterns

- **Reciprocal Figures** (`reciprocal_figures.py`): Dual crease patterns via rotated edge vectors. `make_SRG(G)` is the main entry point.
- **Overlap** (`overlap.py`): Crease assignments (MOUNTAIN/VALLEY), face stacking order via ILP (PuLP), overlap graphs. `fold_complete(SRG)` runs the full pipeline.
- **Cutting** (`cutting.py`): Cuts graphs along halfplanes for unfolding.

### Geometry Backends (`geometries/`)

Pluggable geometry implementations: `EuclideanGeometry`, `PoincareDiskModel` (hyperbolic), `SphereModel`. Used by `GeometricHEG` for distance/angle calculations and position recomputation.

### Rendering

- **Cairo** (`rendering.py`): `CairoRenderer` for high-quality PNG output.
- **SVG** (`svg.py`): Vector output via svgwrite/svgpathtools.
- **Matplotlib** (`plotting.py`): Simple line/polygon plotting helpers.
- **3D** (`marching_cubes.py`): Marching cubes for mesh generation (STL export via meshio).

### Other Modules

- `conversions.py`: Convert between NetworkX graphs and `EuclideanPositionHEG`.
- `classifiers.py`: Classify faces by congruence (edge lengths + angles) for coloring.
- `io.py`: File I/O for `.heg` graph format (YAML-based).
- `search_trees.py`: BFS tree generators for faces and vertices.

## Development Notes

- The library is used interactively via Jupyter notebooks (see `notebooks/`).
- `test.py` at the root is a runnable smoke-test script (not part of the test suite).
- `HalfEdgeGraph.show()` is the quickest way to visualize a graph during development.
- Cyclic iterators (`Vertex.outgoing_iter()`, `Face.halfedge_iter()`) are `while True` loops — never modify graph topology during iteration.
