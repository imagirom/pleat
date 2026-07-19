# Sink folds via local ray casting

Date: 2026-07-19

## Goal

Add crease-pattern manipulation built on a strictly local ray caster over the
existing DCEL (`pleat.half`):

1. Cast a ray from a point on the graph, mirroring its direction at every edge
   it crosses, returning the crossings. Optionally cast both ways, so a ray
   that runs off the edge of the paper still yields its full trajectory.
2. Materialize such a ray as new vertices and edges in the graph.
3. An **open sink fold**: cast, materialize, invert the crease assignments
   inside, infer the rim assignment, and reject invalid sinks.

A **closed sink fold** (same construction, different crease assignment) is
explicitly out of scope for this spec.

"Strictly local" is a hard requirement: every step of the ray needs only its
last intersection point, its direction, and the face it is currently in. No
global geometry queries, no point-in-polygon over the whole graph.

## Scope and non-goals

- Euclidean geometry only. The reflection maths assumes a flat plane; the
  entry points assert `EuclideanPositionHEG` (or a Euclidean-backed
  `GeometricHEG`).
- No closed sink fold.
- No global flat-foldability (layer ordering stays with
  `pleat.overlap.fold_complete`).

## 1. The vertex-fan rule

The interesting case in the ray caster is a ray that hits a vertex exactly.
Per the requirement, this is resolved as if the ray had passed at distance
`ε` to the left of the vertex (side configurable). Naively offsetting the ray
is unnecessary: **`ε` cancels out**, and the rule is purely angular.

### Derivation

A ray arrives at vertex `v` with unit direction `d`, travelling inside face
`f`. Offset the ray laterally to `L = {v + ε·n̂ + t·d}` with `n̂` the left
normal of `d`. Reflections about lines *through `v`* fix `v`, so instead of
reflecting the ray, unfold the paper: reflect each successive sector into the
plane and keep the ray straight. Every reflected copy of `v` coincides with
`v`, so in unfolded coordinates the ray is the straight line `L`.

A ray leaving `v` in unfolded direction `u`, at angle `θ` measured
counter-clockwise from `d`, meets `L` at

```
s = ε / (d × u)                  (distance from v along u)
t = ε · (d·u) / (d × u) = ε·cot θ (position along the ray)
```

Hence:

- `s > 0` iff `d × u > 0`, i.e. **`u` is strictly left of `d`**: `θ ∈ (0, π)`.
  Only left-side edges are crossed.
- `cot` is strictly decreasing on `(0, π)`, so the crossings occur in order of
  **decreasing `θ`**.

`ε` scales `s` and `t` but orders neither, so it drops out of the combinatorics
entirely. It never appears in the implementation.

The ray arrives from within `f`, so `f`'s unfolded sector contains `θ = π`.
Its two boundary rays therefore straddle `π`, and only the one at smaller `θ`
lies in `(0, π)`. That is `f`'s **clockwise** boundary half-edge, so the ray
leaves `f` there.

### Algorithm

Angles are tracked in *unfolded* coordinates, which differ from actual angles
after the first reflection; the update is a running subtraction of the actual
sector angles (`in_angle`, already stored on half-edges).

```
θ   ← ccw_angle(d, direction of f's clockwise boundary half-edge at v)
crossed ← []
while θ > 0:
    h ← current clockwise boundary half-edge at v
    d ← reflect(d, direction of h)      # actual direction, not unfolded
    crossed.append(h)
    step clockwise around v to the next face
    θ ← θ - (in_angle of that sector)
# the ray leaves v into the current face with direction d
return crossed, d, current_face
```

Termination: `θ < π` initially and every sector angle is positive, so the loop
runs at most `deg(v)` times.

`side='right'` mirrors the whole rule: walk counter-clockwise, cross edges with
`θ ∈ (-π, 0)`.

The rule is **tolerance-free**. It reflects off however many edges the fan
requires, which is the case a single "pick one edge to reflect off" rule gets
wrong: after reflecting once, the ray frequently meets a further edge incident
to the same vertex in immediate succession.

## 2. Ray casting — `pleat/ray_casting.py`

### Start primitive

A point on an edge: `(halfedge, t)` with `t ∈ [0, 1]`. `t = 0` and `t = 1` are
allowed and mean `halfedge.orig` / `halfedge.dest`, so starting at a node is a
special case rather than a separate entry point.

### Data

```python
@dataclass
class RayHit:
    halfedges: list[HalfEdge]     # crossed here, in crossing order
    t: float                      # parameter along halfedges[0]
    position: np.ndarray
    vertex: Vertex | None         # set iff the hit is exactly at a vertex
    direction_in: np.ndarray
    direction_out: np.ndarray
    face: Face | None             # face entered after this hit

@dataclass
class RayPath:
    hits: list[RayHit]            # ordered backward end → forward end
    closed: bool
    ends: tuple[str, str]         # each 'closed' | 'border' | 'max_steps'
```

`len(halfedges) > 1` only at a vertex fan. A `RayHit` at a vertex contributes a
single corner, not one hit per crossed edge — otherwise materializing would
produce zero-length rim edges.

### Stepping

State is `(point, direction, face)`. One step:

1. Intersect the ray with each half-edge of `face` (`face.halfedge_iter()`),
   take the nearest strictly-forward crossing. This is the only place that
   touches geometry, and it touches only the current face.
2. If the crossing lies within `vertex_tol` of the half-edge's `orig` or
   `dest`, snap to that vertex and apply the §1 fan rule.
3. Otherwise reflect the direction about the half-edge and continue into
   `h.rev.face`.

Implemented as a generator so §3 can mutate the graph between steps.

### Termination

- **closed** — the ray returns to its start point (within `vertex_tol`).
- **border** — the next face is `None`.
- **max_steps** — safety cap, default 10 000.

### Both ways

`both_ways: bool = True`. If the forward ray does not close, cast a second ray
from the same start point in direction `-d` and prepend its hits reversed, so
`hits` runs from the backward end to the forward end.

The backward ray must use the **mirrored side**: a forward ray with
`side='left'` retraces as a backward ray with `side='right'`. A ray passing at
`+ε` laterally while travelling along `d` is, travelling along `-d` down the
same offset line, passing at `-ε`. Reusing the same side would trace a
different path rather than the continuation of this one.

The backward pass is skipped when the forward ray closes: a closed path already
covers the whole trajectory, and casting backwards would retrace it.

`ends` reports how each end terminated. A path is `closed` iff the forward ray
returned to the start point.

### Tolerances

`vertex_tol` (absolute distance, default derived from the graph's mean edge
length) is the *only* tolerance in the caster: it decides vertex snapping and
closure. The fan rule contributes none.

## 3. Materializing — `add_ray_creases`

Casting and building interleave, one segment at a time. Building the whole path
first and materializing afterwards is wrong: if the ray crosses the same edge
twice, the half-edge recorded on the first pass is stale by the time it is
needed, because `subdivide_edge` has split it.

Per step:

1. `subdivide_edge(h)` at the hit, unless the hit is at an existing vertex.
2. `subdivide_face(current_face, v_prev, v_cur)` to lay the rim segment.
3. The next face is `crossed_halfedge.rev.face`. After a split this is
   automatically the correct sub-face, so no geometric face lookup is ever
   needed — including when the ray re-enters a face it has already cut.

The start point is subdivided first, yielding `v₀`; closure then reduces to
"arrived back at `v₀`", an identity check rather than a distance check.

New half-edges are tagged so the caller can find the rim afterwards.

With `both_ways`, the forward pass is materialized first and the backward pass
then runs on the already-modified graph. This needs no special handling: the
backward ray may well cross edges the forward pass has split, and the
`h.rev.face` rule resolves those the same way it resolves a ray re-entering a
face it has already cut.

Returns the list of rim half-edges in traversal order, plus the `RayPath`.

## 4. Local flat-foldability — extends `pleat/flat_foldable.py`

The existing module has `kawasaki_sum`, `maekawa_check`, and
`is_locally_flat_foldable`; its docstring already reserves big-little-big for
this file.

### Folded crease positions

Given sector angles `a_1 … a_2n` around a vertex, define

```
ψ_0 = 0,   ψ_k = ψ_{k-1} + (-1)^k · a_k
```

`ψ_k` is where crease `k` lands on the line in the folded state. Two
consequences:

- `ψ_{2n}` **is** the Kawasaki sum. Kawasaki is the statement that the cycle
  closes, i.e. the boundary case of this same construction, not an independent
  condition.
- Every remaining local condition — taco-taco, taco-tortilla, equivalently
  every comparison made by the crimp recursion — depends only on **which `ψ_k`
  coincide**. Nothing else about the geometry is continuous.

So the tolerance question collapses to a single clustering of `{ψ_k}`.
Downstream of it, the test is exact discrete combinatorics. This matters for
symmetric vertices (two creases at right angles, three at 60°) where floating
noise otherwise flips the verdict.

### Diagnostic instead of a residual

There is no scalar residual for MV validity — it is genuinely a discrete
predicate once the geometry is non-degenerate. The honest analogue is the
robustness of the clustering:

```
margin = smallest inter-cluster gap in {ψ_k}
```

A large `margin` means the verdict is numerically robust. A `margin` near the
tolerance means the vertex is genuinely ambiguous, and the API says so rather
than returning a confident boolean. Symmetric vertices land here legitimately:
a 4-fold symmetric vertex admits all four 3M-1V assignments.

### The test

```python
def local_assignment_valid(v, tol=...) -> tuple[bool, float]:
    """Return (valid, margin) for the crease assignment at vertex v."""
```

Even degree and `|ψ_{2n}| <= tol` (Kawasaki) are checked first, then the crimp
recursion:

```python
def _crimp_ok(angles, mv):                  # cyclic lists
    if len(angles) == 2:
        return True
    return any(_crimp_ok(*_crimp(angles, mv, i))
               for i in _weakly_minimal(angles, tol)
               if mv[i] != mv[i + 1])       # big-little-big
```

`_crimp(angles, mv, i)` deletes the two creases bounding sector `i` and merges
`a_{i-1} - a_i + a_{i+1}` into one sector.

The recursion backtracks over **weakly** minimal sectors rather than requiring
a strict minimum. Requiring strictness stalls on ties; greedily picking one
weakly-minimal sector is unsound, because with a tie the big-little-big lemma
does not force the bounding creases to differ, so a valid assignment can be
rejected. Vertex degrees are ≤ ~12, so exhaustive backtracking is free and
provably correct.

`is_locally_flat_foldable` is updated to call this in place of its current
Maekawa-only check. Maekawa is implied by the crimp reduction, but
`maekawa_check` stays as a public cheap necessary condition.

## 5. Open sink — `pleat/sink.py`

```python
def open_sink(G, halfedge, t, direction, side='left', strict=True, ...) -> list[HalfEdge]:
```

1. Cast and materialize the rim (§2, §3).
2. Invert `CREASE_ASSIGNMENT` on every crease strictly inside the rim. The
   inside is found by flood fill over faces, seeded from the faces on the
   inward side of the rim and bounded by rim half-edges. Rim-adjacent, so
   still local — no polygon containment test over the graph.
3. The rim of an open sink is uniformly `MOUNTAIN` or uniformly `VALLEY`.
   Set the whole rim to `MOUNTAIN` and run §4 at every interior rim node; if
   any node fails, flip the whole rim to `VALLEY` and retest.
4. If neither works, raise `InvalidSinkError` naming the failing vertices and
   their `margin`. If a node reports a `margin` near the tolerance, log that
   the verdict is degenerate.

### Open rims

The rim does not have to close. If the ray reaches the border, `both_ways`
casts backwards from the start point; if that reaches the border too, the rim
is a path from border to border rather than a cycle, and the sink is still
well-defined. Nothing in steps 2–4 assumes a cycle:

- The flood fill is bounded by rim half-edges **and** border half-edges, and
  still terminates because the enclosed region is finite.
- The rim endpoints lie on the border and carry no local flat-foldability
  condition, so step 3 tests interior rim nodes only.

Only `max_steps` is a genuine failure, and it raises.

### `strict`

`strict: bool = True` controls what happens when the crease assignment is
absent or cannot be made valid:

- **Missing assignments.** With `strict=False`, edges lacking
  `CREASE_ASSIGNMENT` are left alone by the inversion in step 2, and rim nodes
  where not every incident crease is assigned are skipped in step 3. This makes
  `open_sink` usable as a pure geometric operation on a pattern whose creases
  have not been assigned yet. With `strict=True`, a missing assignment on an
  edge that step 2 or 3 needs raises `InvalidSinkError`.
- **Unsatisfiable assignments.** With `strict=False`, step 4 logs a warning
  naming the failing vertices and leaves the rim at `MOUNTAIN` instead of
  raising, so the caller still gets the geometry and can inspect or repair the
  result. With `strict=True` it raises.

`strict=False` never changes the geometry produced — steps 1 and 2 are
unaffected — only whether a crease problem aborts the operation.

The degree-4 case falls out of §4 rather than being special-cased: at a new rim
node the radial crease is split into `c_in` (inverted) and `c_out`, so
`c_in = -c_out`; Maekawa then forces the two rim edges equal, and
big-little-big forces the odd crease to be the radial half bounding the
smallest sector, giving rim `= c_out`. Nodes where the ray passed through an
existing vertex are handled by the same general test with no extra code.

`InvalidSinkError` subclasses `ValueError`.

## Testing

`tests/test_ray_casting.py`:

- Reflection round-trip on a regular triangular grid: a ray cast from a known
  point returns to it after the expected number of crossings.
- A ray aimed exactly at a vertex, from both `side='left'` and `side='right'`,
  giving mirrored paths.
- A ray whose vertex fan crosses more than one edge (the multi-reflection
  case), checked against a manually computed direction.
- A ray that reaches the border, reporting `ends[1] == 'border'`.
- `both_ways` on a ray that hits the border in both directions: `hits` runs
  border to border, `ends == ('border', 'border')`.
- `both_ways` on a closing ray casts no backward pass — the hit count matches
  the one-way result.
- A ray crossing the same edge twice, verifying `add_ray_creases` produces a
  consistent graph (`check_consistency`).

`tests/test_flat_foldable.py` (extend):

- `ψ` prefix sum reproduces `kawasaki_sum` at its final index.
- 4-fold symmetric vertex: all four 3M-1V assignments valid, `margin` small.
- Generic degree-4 vertex with a unique smallest sector: exactly the
  big-little-big-consistent assignments valid.
- A vertex where greedy tie-breaking would reject a valid assignment.

`tests/test_sink.py`:

- Sinking the apex of a preliminary-base-like pattern yields a graph passing
  `is_locally_flat_foldable`.
- The rim comes out uniform, and matches the hand-derived M/V.
- A deliberately invalid sink raises `InvalidSinkError` naming the bad
  vertices, and with `strict=False` warns and still returns the geometry.
- `open_sink` on a pattern with no `CREASE_ASSIGNMENT` at all: raises with
  `strict=True`, produces the same geometry with `strict=False`.
- A sink whose ray reaches the border in both directions: open rim, interior
  nodes still locally flat-foldable.
- `IdObject.reset_ids()` in fixtures, per the project's test convention.

## Open questions

None blocking. Deferred by decision: closed sink folds, non-Euclidean
geometries.
