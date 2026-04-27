# Improvements backlog

Ranked list of follow-up improvements to the `eucare` library and
documentation, surfaced while writing the instructive notebook series. Each
item is rated **impact** (how much it helps users / maintainers) × **effort**
(rough cost to implement). High-impact / low-effort first.

Trivial one-liners are applied directly and **not** listed here.

---

## P1 — High impact

### P1.1 — Standardize crease-pattern render preset · *low effort*

Most SRG-related notebooks redefine the same dict:

```python
render_settings = dict(face_inset=0, render_vertices=False, render_faces=False)
```

Every notebook that omits one of these keys looks subtly different. Propose a
small `eucare.rendering.CREASE_PATTERN_PRESET` constant (or a helper
`crease_pattern_renderer(**overrides)`) and use it consistently in the new
notebook series.

**Acceptance:** importable `eucare.rendering.CREASE_PATTERN_PRESET`; all new
notebooks reference it; legacy notebooks left untouched.

### P1.2 — Replace `%pylab inline` in legacy notebooks · *medium effort*

`%pylab` pollutes the namespace and is deprecated. New notebooks use explicit
imports. Replacing in 30+ legacy notebooks is mechanical but verbose; defer
until the legacy notebooks are pruned (see `notebooks/README.md`).

**Acceptance:** zero `%pylab` occurrences in `docs/notebooks/`. Legacy left as-is.

### P1.3 — Finish type-hint pass · *medium effort*

Conway factories are now annotated. Remaining gaps:

- `eucare/half.py` (~99 public functions, especially `HalfEdgeGraph`,
  `InAngleHEG`, `GeometricHEG`, `EuclideanPositionHEG` methods).
- `eucare/geometries/{euclidean,hyperbolic,spherical,base}.py` classmethods
  returning `np.ndarray`.
- `eucare/example_tilesets.py`, `eucare/example_graphs.py`,
  `eucare/io.py`, `eucare/rendering.py`, `eucare/cutting.py`,
  `eucare/overlap.py` top-level helpers.

**Acceptance:** `mypy eucare` passes (current strictness); no new untyped
public API.

---

## P2 — Medium impact

### P2.1 — Expose Kawasaki / Maekawa helpers · *low effort*

Five+ notebooks reimplement variants of:

```python
def kawasaki_sum(v):
    return sum(((-1) ** i) * h['in_angle'] for i, h in enumerate(v.outgoing_iter()))
```

Promote a clean version to `eucare.reciprocal_figures` (or a new
`eucare.foldability` module) along with `is_flat_foldable(graph)` returning
boolean + per-vertex diagnostics.

**Acceptance:** `from eucare.reciprocal_figures import kawasaki_sum, maekawa_check`
works; documented in `04_Folding_and_Overlap.ipynb`.

### P2.2 — Deprecate / remove `fancy`-dependent code paths · *medium effort*

The `fancy` library is no longer maintained or available on PyPI. Notebooks
that depend on it are flagged DISCARD in `notebooks/README.md`. Audit
`eucare/` for any remaining `fancy` imports and either drop those code paths
or guard them behind `try/except ImportError`.

**Acceptance:** `grep -r 'import fancy\|from fancy' eucare/` returns nothing.

### P2.3 — `convert_to_euclidean` no-op fast path · *low effort*

`GeometricHEG.convert_to_euclidean()` always re-projects every vertex and
recomputes lengths, even when `geometry is EuclideanGeometry`. Add an early
return when already Euclidean.

**Acceptance:** unit test asserts that calling on a Euclidean graph leaves
positions byte-identical.

### P2.4 — Mountain / Valley color constants · *low effort*

Notebooks pick different reds/blues/grays. Standardize:

```python
# eucare/rendering.py
MOUNTAIN_COLOR = '#cc2222'
VALLEY_COLOR = '#2266cc'
FLAT_COLOR = '#888888'
```

**Acceptance:** constants importable; new notebooks use them; SVG export uses
them by default.

### P2.5 — `eucare.io` round-trip test for `.heg` · *low effort*

`io.py` is at 95% coverage but lacks an explicit round-trip integration test
for arbitrary tilings (curved geometries especially). Worth adding after
P2.3.

**Acceptance:** `tests/test_io.py::test_roundtrip_curved` passes.

---

## P3 — Low impact

### P3.1 — Coverage of numba-jit code · *low effort*

`overlap.line_segment_intersections` (lines 85–134) is not measurable by
`coverage.py` due to numba JIT. Two options:

- Add `# pragma: no cover` on the jit body so the totals reflect reality.
- Or run a parallel non-jit shadow function for testing.

**Acceptance:** total coverage reported reflects what's actually testable.

### P3.2 — `notebooks/` housekeeping commit · *low effort*

After this round of curation, hard-delete the DISCARD entries listed in
`notebooks/README.md` (about 13 files). Should be a separate commit so it can
be reviewed and reverted easily.

**Acceptance:** `notebooks/README.md` table only lists files that exist in
the directory.

### P3.3 — Replace heavy `save_results` in `overlap.py` · *medium effort*

`overlap.save_results` (lines 729–821) is ~93 lines of hard-to-test IO + plot
code. Split into pure-data and rendering halves; the rendering half can be
covered by mocking the renderer.

**Acceptance:** `overlap.py` total coverage ≥ 90% without changing public
behaviour.

---

## Trivial improvements applied immediately

The following were applied in the same commit as this file:

- `eucare.rendering.CREASE_PATTERN_PRESET` and
  `eucare.rendering.MOUNTAIN_COLOR / VALLEY_COLOR / FLAT_COLOR` constants
  (P1.1, P2.4).
- `eucare.reciprocal_figures.kawasaki_sum` helper (P2.1, partial — only the
  sum, not the full `is_flat_foldable` API).
- `eucare.rendering.multi_show(graphs, titles=...)` — render multiple graphs
  side-by-side with titles, used throughout the curated notebook series.

---

## Surfaced while writing the styling / modifications notebooks

### P2.6 — `overlap.fold_complete(progress=False)` flag · *trivial*

`fold_complete` (and the helpers it calls) prints a tqdm progress bar
unconditionally. The notebook series silences it via the
`TQDM_DISABLE=1` environment variable, but a proper `progress: bool = True`
parameter (forwarded into the relevant `tqdm` calls) would be cleaner.

**Acceptance:** `overlap.fold_complete(SRG, progress=False)` is silent
without env-var hacks.

### P2.7 — `LenClassifier` / `congruency_classifier` accept faces directly · *low effort*

In notebook 07 we currently wrap `LenClassifier()` in
`PreMapClassifier(..., lambda f: list(f.halfedge_iter()))` because `Face`
doesn't implement `__len__`. Either teach `LenClassifier` to fall back to
`f.order()` when given a `Face`, or add a top-level
`face_corner_count_classifier()` helper.

**Acceptance:** `colorize(G, face_corner_count_classifier())` works directly
on a graph.
