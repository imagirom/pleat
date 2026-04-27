# Installation

## With uv (recommended)

```bash
git clone https://github.com/rremme/eucare.git
cd eucare
uv venv --python 3.10
uv pip install -e ".[dev]"
```

## With pip

```bash
pip install -e ".[dev]"
```

## Optional extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, pytest-cov, ruff | Testing and linting |
| `docs` | mkdocs-material, mkdocstrings | Documentation |
| `notebook` | jupyter, ipywidgets, ipympl | Interactive notebooks |
| `threed` | meshio | 3D mesh export (STL) |
| `torch` | torch, einops | Optimization (e.g. flagstone fitting) |

Install multiple extras at once:

```bash
uv pip install -e ".[dev,docs,notebook]"
```

## External dependencies

These are not available on PyPI and must be installed separately:

- **[fancy](https://github.com/imagirom/fancy)** — configuration library (optional)
- **[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)** — ILP solver for face ordering in folded states (optional; PuLP falls back to GLPK)
