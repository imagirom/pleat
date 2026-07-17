# Improvements backlog

Ranked list of follow-up improvements to the `pleat` library and
documentation. Each item is rated **impact** (how much it helps users /
maintainers) × **effort** (rough cost to implement). High-impact /
low-effort first. Completed items are removed rather than checked off.

---

## P1 — High impact

### P1.1 — Finish type-hint pass · *high effort*

`mypy pleat` currently reports ~575 errors across 30 files (with untyped
third-party imports already ignored via pyproject overrides). The bulk comes
from two systemic patterns:

- The DCEL fields (`HalfEdge.rev/nex/pre/face`, `Vertex.any_outgoing`, …) are
  typed `X | None` but nearly all code assumes them non-`None` after graph
  construction. Needs either narrowing asserts at API boundaries or a
  rethink of the declared types.
- Functions with a `return_mappings`-style flag returning
  `X | tuple[X, dict]` poison downstream call sites. Fix with `@overload`
  on the few offenders (`HalfEdgeGraph.copy`, `EHEG_from_nx`, …).

Remaining annotation gaps (docstrings/types) as before: `half.py` mid-tier
methods, geometry classmethod overrides, `example_graphs.py` /
`example_tilesets.py` builder return types.

**Acceptance:** `mypy pleat` passes (current strictness); no new untyped
public API.

---

## P2 — Medium impact

### P2.1 — Split `pleat/half.py` (1800 LOC) · *medium effort*

`half.py` mixes the bare DCEL primitives (``AttributeObject``, ``IdObject``,
``HalfEdge``, ``Vertex``, ``Face``), the topological graph
(``HalfEdgeGraph``, ``CyclicHalfedgeGraph``, ``RegularNGon``), and three
specialisations adding angles / pluggable geometry / Euclidean positions
(``InAngleHEG``, ``GeometricHEG``, ``EuclideanPositionHEG``). Splitting into
``pleat/half/{primitives.py,graph.py,geometric.py}`` would make the module
easier to navigate, easier to type-check incrementally, and would expose a
clearer mental model in the API docs.

**Acceptance:** `from pleat.half import HalfEdge, HalfEdgeGraph,
EuclideanPositionHEG` continues to work via re-exports; `half/` package is
under 700 LOC per module.

### P2.2 — Split `pleat/overlap.py` (1078 LOC) · *medium effort*

`overlap.py` interleaves three concerns:

1. Geometric primitives — `line_segment_intersections`,
   `get_potential_intersections`, `intervals_overlapping`,
   `fast_group_closeby` / `faster_group_closeby_nx`.
2. The overlap-graph construction (`overlap_graph`, `remove_duplicates`).
3. The flat-foldability ILP and pipeline (`find_folded_face_order`,
   `infer_additional_over_under_pairs`, `fold_wireframe`,
   `face_order_to_clean_graph`, `color_creases`, `fold_complete`,
   `save_results`).

Pull (1) into `pleat/geometry_helpers.py` (or back into `pleat/base.py`)
and (3) into `pleat/folding.py`; keep `overlap.py` focused on graph
construction.

**Acceptance:** public symbols re-exported from `pleat.overlap` for
backwards compatibility; per-module LOC under 500.

---

## P3 — Low impact

### P3.1 — Replace heavy `save_results` in `overlap.py` · *medium effort*

`overlap.save_results` is ~90 lines of hard-to-test IO + plot code. Split
into pure-data and rendering halves; the rendering half can be covered by
mocking the renderer.

**Acceptance:** `overlap.py` total coverage ≥ 90% without changing public
behaviour.
