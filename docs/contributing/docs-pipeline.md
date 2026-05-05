# Documentation Pipeline

Eucare's documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and served as a static site. This page explains the pipeline so you can add pages, fix docstrings,
or add example notebooks without having to reverse-engineer the setup.

## Setup

```bash
uv pip install -e ".[docs]"
```

The `docs` extra installs:

| Package | Role |
|---|---|
| `mkdocs-material` | Theme and Markdown extensions |
| `mkdocstrings[python]` | Auto-generates API pages from docstrings |
| `mkdocs-gen-files` | Runs `docs/gen_ref_pages.py` at build time to create `reference/*.md` |
| `mkdocs-literate-nav` | Reads the generated `reference/SUMMARY.md` to populate the API nav |
| `mkdocs-jupyter` | Renders and executes Jupyter notebooks as docs pages |
| `jupyter`, `ipywidgets`, `ipympl` | Runtime needed for the published notebook examples |
| `scikit-image`, `mahotas`, `rdp`, `plotly` | Optional stacks required by notebooks currently published in the docs |

## Build and Preview

```bash
# Local dev server — reloads on file change
DISABLE_MKDOCS_2_WARNING=true mkdocs serve

# One-shot static build into site/
DISABLE_MKDOCS_2_WARNING=true mkdocs build
```

The `DISABLE_MKDOCS_2_WARNING=true` flag suppresses a spurious deprecation warning
injected by `properdocs`, a transitive dependency that is a fork of MkDocs.

Notebook pages are executed during the build, so the published docs do not rely on committed cell outputs.

## How API Reference Pages Are Generated

`docs/gen_ref_pages.py` runs at build time (via the `gen-files` plugin). It:

1. Walks `eucare/*.py` (and `eucare/geometries/*.py`) and creates a stub page per module at
   `reference/<module>.md` containing only a `:::` autodoc directive.
2. Writes `reference/SUMMARY.md` listing all pages in the order they were discovered.

The `literate-nav` plugin reads `SUMMARY.md` and builds the **API Reference** nav section
from it automatically. The nav entry must point to the directory (`reference/`), not to a file,
so literate-nav can find and expand the generated summary.

Docstrings use **Google style**. mkdocstrings extracts them and renders sections
(`Args:`, `Returns:`, `Raises:`, `Example:`) as formatted HTML.

## Adding Example Notebooks

1. Add or move the notebook into `docs/notebooks/`.
2. Add an entry to the **Notebooks** section in `mkdocs.yml`:
   ```yaml
   - Notebooks:
       - My New Notebook: notebooks/My New Notebook.ipynb
   ```
3. Run `DISABLE_MKDOCS_2_WARNING=true mkdocs build` and fix any missing runtime dependencies before committing.

The docs build executes notebooks from source, so committed outputs in `docs/notebooks/` are optional. The pre-commit configuration deliberately skips output stripping for `docs/notebooks/`, while stripping outputs from notebooks elsewhere in the repository.

### Interactive widgets (ipywidgets)

mkdocs-jupyter includes RequireJS (`include_requirejs: true`) to partially support
widget rendering. However, **interactive widgets** (sliders, dropdowns) require a live
Python kernel and will not be functional on the static site — they fall back to their
`text/plain` representation. To show a meaningful static output, save a plain matplotlib
figure alongside the widget cell, or replace the widget with a static figure before
committing the notebook to `docs/notebooks/`.

## Notebook Outputs And Commits

Install the hooks once per clone:

```bash
pre-commit install
```

The repository uses `nbstripout` in pre-commit to remove cell outputs from notebooks on commit, except for notebooks under `docs/notebooks/`. That keeps exploratory notebooks lightweight while leaving the published docs notebooks free to carry checked-in outputs when that is useful.

### Notebook headings

Add a `# My Notebook Title` markdown cell as the very first cell of any new
notebook so the title appears correctly in both Jupyter and the docs.
