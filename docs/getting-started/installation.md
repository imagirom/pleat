# Installation

## With uv (recommended)

```bash
git clone https://github.com/rremme/eucare.git
cd eucare
uv venv --python 3.10
uv pip install -e ".[dev]"
```

For a full local setup with docs, notebooks, and optional feature stacks, install:

```bash
uv pip install -e ".[all]"
```

## With pip

```bash
pip install -e ".[dev]"
```

## Optional extras

| Extra | Packages | Purpose |
|-------|----------|---------|
| `dev` | pytest, pytest-cov, black, mypy, pre-commit, nbstripout | Testing, formatting, and commit hooks |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-jupyter, notebook runtime, image + intersecting-cylinders deps | Documentation, including executing the published notebooks during `mkdocs build` |
| `notebook` | jupyter, ipywidgets, ipympl, widgetsnbextension | Interactive notebooks |
| `threed` | meshio | 3D mesh export (STL) |
| `torch` | torch, einops | Optimization (e.g. flagstone fitting) |
| `image` | scikit-image, mahotas | Image-to-graph pipeline |
| `intersecting_cylinders` | rdp, plotly | Intersecting-cylinders module |
| `all` | Union of all optional extras | Convenience install for local development |

Install multiple extras at once:

```bash
uv pip install -e ".[dev,docs,notebook]"
```

Keeping focused extras is normal practice: it keeps installs smaller and makes feature-specific requirements explicit. The `all` extra is the convenience path when you do want everything available in one environment.

## External dependencies

These are not available on PyPI and must be installed separately:

- **[fancy](https://github.com/imagirom/fancy)** — configuration library (optional)
- **[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)** — ILP solver for face ordering in folded states (optional; PuLP falls back to GLPK)
