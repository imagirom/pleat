# Installation

## Using the library

```bash
pip install pleat
```

That's all you need to build tilings and crease patterns in your own code.
All example notebooks are readable — executed, with output — right here on
this documentation site, no installation required.

## Running the notebooks locally

The notebooks use sample graphs and images shipped in the repository, so to
run or modify them, clone the repo and install with all extras:

```bash
git clone https://github.com/imagirom/pleat.git
cd pleat
uv venv --python 3.14
uv pip install -e ".[all]"
jupyter lab docs/notebooks
```

If you prefer a lighter install, pick a subset from the
[extras table](#optional-extras) below instead — e.g.
`".[notebook,image,intersecting_cylinders]"` runs most notebooks while
skipping heavy dependencies such as torch (only needed for the
alternating-flagstones optimization).

## Contributing / development setup

```bash
git clone https://github.com/imagirom/pleat.git
cd pleat
uv venv --python 3.14
uv pip install -e ".[dev]"
```

For a full local setup with docs, notebooks, and optional feature stacks,
install `".[all]"` instead. (Plain `pip install -e ".[dev]"` works too if you
don't use uv.)

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

## Optional External dependencies

These are not available on PyPI and must be installed separately:

- **[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)** — ILP solver for face ordering in folded states. Free licenses for students and academics. (optional; PuLP has fallback solvers which work well enough for most crease patterns)
