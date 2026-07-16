# Improvements backlog

Ranked list of follow-up improvements to the `pleat` library and
documentation. Each item is rated **impact** (how much it helps users /
maintainers) × **effort** (rough cost to implement). High-impact /
low-effort first. Completed items are removed rather than checked off.

---

## P1 — High impact

### P1.1 — Finish type-hint pass · *medium effort*

Type hints + Google-style docstrings exist across most public API. Remaining
gaps:

- `pleat/half.py` mid- and lower-tier methods: `delete_edge`, `glue_*`,
  `subdivide_*`, `recompute_positions`, `show`, `copy`, `check_consistency`,
  the entire `InAngleHEG` / `GeometricHEG` / `EuclideanPositionHEG` /
  `CyclicHalfedgeGraph` / `RegularNGon` overrides.
- `pleat/geometries/{euclidean,hyperbolic,spherical}.py` classmethod
  overrides (return types). The abstract base is fully documented and the
  overrides inherit semantics, so adding return types is mechanical.
- `pleat/classifiers.py` private overrides (`_get_index`,
  `_compare_representations`, `_represent_item`) — class-level docstrings
  document the contract; redundant per-method docstrings would be noise.
- `pleat/example_graphs.py`, `pleat/example_tilesets.py` builder return
  types — currently untyped because of branching factory signatures.

**Acceptance:** `mypy pleat` passes (current strictness); no new untyped
public API.

---

## P2 — Medium impact

### P2.1 — Full flat-foldability diagnostics · *low effort*

`pleat.flat_foldable` has `kawasaki_sum` / `max_kawasaki_sum`. Add
`maekawa_check` and an `is_flat_foldable(graph)` returning boolean +
per-vertex diagnostics.

**Acceptance:** `from pleat.flat_foldable import is_flat_foldable, maekawa_check`
works.

### P2.2 — `convert_to_euclidean` no-op fast path · *low effort*

`GeometricHEG.convert_to_euclidean()` always re-projects every vertex and
recomputes lengths, even when `geometry is EuclideanGeometry`. Add an early
return when already Euclidean.

**Acceptance:** unit test asserts that calling on a Euclidean graph leaves
positions byte-identical.

### P2.3 — `pleat.io` round-trip test for `.heg` · *low effort*

`io.py` lacks an explicit round-trip integration test for arbitrary tilings
(curved geometries especially).

**Acceptance:** `tests/test_io.py::test_roundtrip_curved` passes.

### P2.4 — `LenClassifier` / `congruency_classifier` accept faces directly · *low effort*

Notebook 07 wraps `LenClassifier()` in
`PreMapClassifier(..., lambda f: list(f.halfedge_iter()))` because `Face`
doesn't implement `__len__`. Either teach `LenClassifier` to fall back to
`f.order()` when given a `Face`, or add a top-level
`face_corner_count_classifier()` helper.

**Acceptance:** `colorize(G, face_corner_count_classifier())` works directly
on a graph.

### P2.5 — Split `pleat/half.py` (1800 LOC) · *medium effort*

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

### P2.6 — Split `pleat/overlap.py` (1078 LOC) · *medium effort*

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

### P2.7 — Make `Classifier.classify` and friends generic · *low effort*

`pleat/classifiers.py` defines a hierarchy of `Classifier` subclasses with
a contract spelled out in class docstrings but enforced only at the
`_get_index` level. Annotating `classify` and the subclass overrides with
`Generic[T]` (over the input type) and `Hashable` return types would let
mypy catch misuse and would document the API better than the class
docstrings alone.

**Acceptance:** `Classifier`, `RepresentationClassifier`,
`NestedClassifier` are typed `Generic[T]`; existing call sites unchanged.

### P2.8 — Drop `print_attribute_info` or move to a debug module · *trivial*

`pleat/utils.py::print_attribute_info` is a diagnostic helper used in two
notebooks. It writes to stdout and has no tests. Either drop it or move it
to a clearly-named `pleat/debug.py`.

**Acceptance:** `pleat/utils.py` no longer contains UI-style helpers.

### P2.9 — `pleat/image_to_graph.py` `threshold` and `edge_length_cutoff` · *low effort*

`image_to_graph(...)` raises `NotImplementedError` when either parameter is
left at its default `None`, but the function signature does not signal this.
Either make them required positional arguments, or implement the two
auto-estimation paths.

**Acceptance:** calling `image_to_graph(rgb)` without keyword args either
returns a sensible default or fails at parse time, not at runtime.

---

## P3 — Low impact

### P3.1 — Coverage of numba-jit code · *low effort*

`overlap.line_segment_intersections` is not measurable by `coverage.py` due
to numba JIT. Add `# pragma: no cover` on the jit body so the totals reflect
reality.

**Acceptance:** total coverage reported reflects what's actually testable.

### P3.2 — Replace heavy `save_results` in `overlap.py` · *medium effort*

`overlap.save_results` is ~90 lines of hard-to-test IO + plot code. Split
into pure-data and rendering halves; the rendering half can be covered by
mocking the renderer.

**Acceptance:** `overlap.py` total coverage ≥ 90% without changing public
behaviour.
