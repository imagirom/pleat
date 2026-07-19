# Sink Folds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strictly local ray caster over pleat's DCEL and build an open sink fold on top of it.

**Architecture:** A ray is stepped through the crease pattern one face at a time, carrying only `(point, direction, face)`. Crossing a crease transmits the direction by `d - 2(d·û)û`; hitting a vertex is resolved by an angular fan walk in which the epsilon offset cancels out. Materializing the ray reuses `subdivide_edge`/`subdivide_face` and derives the next face from `h.rev.face`, so no global geometry query is ever needed. The open sink casts, materializes, inverts the interior crease assignment, and picks the uniform rim assignment by testing local flat-foldability at every rim node.

**Tech Stack:** Python 3.10–3.14, numpy, pytest, black. No new dependencies.

Spec: `docs/superpowers/specs/2026-07-19-sink-folds-design.md`.

## Global Constraints

- Euclidean geometry only. Public entry points assume vertex positions in `v["pos"]` as 2-D numpy arrays and `in_angle`/`length` present on half-edges (call `G.recompute_lengths_and_angles()` if unsure).
- Never mutate graph topology while iterating a cyclic iterator (`Vertex.outgoing_iter`, `Face.halfedge_iter`) — materialize with `list(...)` first.
- `in_angle` on a half-edge `e` is the interior angle at `e.dest` between `e` and `e.nex`.
- `Vertex.outgoing_iter()` is counter-clockwise; `reverse_outgoing_iter()` is clockwise (`h.rev.nex`).
- Crease assignment lives in `h[CREASE_ASSIGNMENT]` with `MOUNTAIN = 1`, `VALLEY = -1`, `BORDER = 0`, all from `pleat.overlap`. Assignments are set on **both** half-edges of an edge.
- Tests live in `tests/`, use plain pytest functions, and rely on the autouse `reset_ids` fixture in `tests/conftest.py`.
- Run `uv run --extra dev black pleat tests` before each commit; CI enforces `black --check`.
- Every task ends with a commit. Do not include a Claude co-author trailer.

---

### Task 1: Vector primitives

**Files:**
- Create: `pleat/ray_casting.py`
- Test: `tests/test_ray_casting.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `cross2(a, b) -> float`, `signed_angle(a, b) -> float`, `transmit(d, u) -> np.ndarray`, `halfedge_direction(h) -> np.ndarray`, `DegenerateRayError(ValueError)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ray_casting.py`:

```python
"""Tests for the local ray caster over crease patterns."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.ray_casting import (
    DegenerateRayError,
    cross2,
    halfedge_direction,
    signed_angle,
    transmit,
)

SQRT_HALF = np.sqrt(0.5)


def test_cross2():
    assert cross2(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cross2(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(-1.0)


def test_signed_angle_is_ccw_and_in_minus_pi_to_pi():
    east, north = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert signed_angle(east, north) == pytest.approx(np.pi / 2)
    assert signed_angle(north, east) == pytest.approx(-np.pi / 2)
    assert signed_angle(east, east) == pytest.approx(0.0)


def test_transmit_through_perpendicular_crease_goes_straight():
    # a rim segment travelling east crosses a vertical (north-south) crease
    d = np.array([1.0, 0.0])
    u = np.array([0.0, 1.0])
    np.testing.assert_allclose(transmit(d, u), d, atol=1e-12)


def test_transmit_through_45_degree_crease_turns_by_90_degrees():
    # the square-preliminary-base check from the spec
    d = np.array([1.0, 0.0])
    u = np.array([SQRT_HALF, SQRT_HALF])
    np.testing.assert_allclose(transmit(d, u), np.array([0.0, -1.0]), atol=1e-12)


def test_transmit_is_not_mirroring_across_the_crease():
    # mirroring would send the ray back into the face it came from
    d = np.array([1.0, 0.0])
    u = np.array([SQRT_HALF, SQRT_HALF])
    mirrored = 2 * np.dot(d, u) * u - d
    assert not np.allclose(transmit(d, u), mirrored)


def test_transmit_preserves_length_and_is_an_involution():
    d = np.array([0.6, -0.8])
    u = np.array([1.0, 2.0])
    out = transmit(d, u)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    np.testing.assert_allclose(transmit(out, u), d, atol=1e-12)


def test_transmit_normalises_the_crease_direction():
    d = np.array([1.0, 0.0])
    np.testing.assert_allclose(
        transmit(d, np.array([0.0, 5.0])), transmit(d, np.array([0.0, 1.0])), atol=1e-12
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pleat.ray_casting'`

- [ ] **Step 3: Write minimal implementation**

Create `pleat/ray_casting.py`:

```python
"""Cast rays through a crease pattern, transmitting at every crease.

A ray carries only its current point, direction, and face, so every step is
local: no global geometry query is ever made.  Crossing a crease transmits the
direction by ``d - 2(d.u)u``; hitting a vertex is resolved by an angular fan
walk in which the epsilon offset cancels out (see
``docs/superpowers/specs/2026-07-19-sink-folds-design.md``).
"""

from __future__ import annotations

import numpy as np

from .half import HalfEdge


class DegenerateRayError(ValueError):
    """The ray hit a configuration this module deliberately does not resolve."""


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Return the scalar cross product ``a x b`` of two 2-D vectors."""
    return float(a[0] * b[1] - a[1] * b[0])


def signed_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Return the counter-clockwise angle from *a* to *b*, in ``(-pi, pi]``."""
    return float(np.arctan2(cross2(a, b), np.dot(a, b)))


def transmit(d: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Transmit direction *d* across a crease of direction *u*.

    Keeps the component crossing the crease and flips the component along it.
    This is *not* the mirror image ``2(d.u)u - d``, which flips the crossing
    component and would send the ray back into the face it came from.
    """
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    return np.asarray(d, dtype=float) - 2 * np.dot(d, u) * u


def halfedge_direction(h: HalfEdge) -> np.ndarray:
    """Return the vector from ``h.orig`` to ``h.dest`` in position space."""
    return h.dest["pos"] - h.orig["pos"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: PASS, 7 passed

- [ ] **Step 5: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/ray_casting.py tests/test_ray_casting.py
git commit -m "feat(ray_casting): vector primitives for crease transmission"
```

---

### Task 2: The vertex fan rule

**Files:**
- Modify: `pleat/ray_casting.py`
- Test: `tests/test_ray_casting.py`

**Interfaces:**
- Consumes: `signed_angle`, `transmit`, `halfedge_direction`, `DegenerateRayError` from Task 1.
- Produces: `fan_at_vertex(v, d, face, side="left") -> tuple[list[HalfEdge], np.ndarray, Face | None]` — the half-edges transmitted through in order, the outgoing direction, and the face the ray leaves into (`None` if it ran off the paper).

The walk and its justification are in spec §1. Two facts pinned down against the real data structure:

- `f`'s **clockwise** boundary at `v` is the outgoing half-edge `g` with `g.face is f`; its **counter-clockwise** boundary is the outgoing `g` with `g.rev.face is f`.
- `g.rev["in_angle"]` is the sector angle at `v` of the next face **clockwise**; `g.pre["in_angle"]` is the sector angle at `v` of the next face **counter-clockwise**.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ray_casting.py`:

```python
from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.ray_casting import fan_at_vertex


def _grid():
    """Unit square grid; central vertex has degree 4 with four 90-degree sectors."""
    G = from_tiles(platonic(n=4), rings=2)
    G.recompute_lengths_and_angles()
    return G


def _outgoing_towards(v, target):
    """Return the outgoing half-edge at *v* pointing at *target* (a position)."""
    return next(
        h for h in v.outgoing_iter() if np.allclose(h.dest["pos"], target, atol=1e-9)
    )


def test_fan_at_degree_4_vertex_transmits_through_exactly_one_crease():
    G = _grid()
    v = G.central_vertex()  # at (-0.5, 0.5), creases pointing W, S, E, N
    west = _outgoing_towards(v, [-1.5, 0.5])
    sw_face = west.face  # the sector spanning 180..270 degrees

    d = np.array([SQRT_HALF, SQRT_HALF])  # arriving north-east, so -d is in the SW face
    crossed, d_out, face_out = fan_at_vertex(v, d, sw_face)

    assert crossed == [west]
    # crossing a horizontal crease flips the horizontal component
    np.testing.assert_allclose(d_out, np.array([-SQRT_HALF, SQRT_HALF]), atol=1e-12)
    assert face_out is west.rev.face


def test_fan_transmits_through_several_creases_at_one_vertex():
    """Hand-computed 3-crossing case from the spec.

    A 135-degree crease is added into the north-west square, so walking
    clockwise from the south-west face the sectors are 45, 45, 90 degrees.
    Arriving along 60 degrees gives theta = 120, 165, 120, 210 -- three
    crossings before theta leaves (0, pi).
    """
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])
    north = _outgoing_towards(v, [-0.5, 1.5])
    nw_face = north.face
    far_corner = next(
        w for w in nw_face.vertex_iter() if np.allclose(w["pos"], [-1.5, 1.5], atol=1e-9)
    )
    G.subdivide_face(nw_face, v, far_corner)  # the 135-degree crease
    G.recompute_lengths_and_angles()
    diagonal = _outgoing_towards(v, [-1.5, 1.5])

    d = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
    crossed, d_out, face_out = fan_at_vertex(v, d, west.face)

    assert crossed == [west, diagonal, north]
    np.testing.assert_allclose(
        d_out, np.array([np.cos(np.pi / 6), np.sin(np.pi / 6)]), atol=1e-12
    )
    assert face_out is north.rev.face


def test_fan_grazing_a_corner_transmits_through_nothing():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])

    # theta_1 = 180 - 200 = -20 degrees, outside (0, pi): the offset ray misses
    d = np.array([np.cos(np.deg2rad(200)), np.sin(np.deg2rad(200))])
    crossed, d_out, face_out = fan_at_vertex(v, d, west.face)

    assert crossed == []
    np.testing.assert_allclose(d_out, d, atol=1e-12)
    assert face_out is west.face


def test_fan_side_right_mirrors_side_left():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])
    south = _outgoing_towards(v, [-0.5, -0.5])

    d = np.array([SQRT_HALF, SQRT_HALF])
    crossed, d_out, _ = fan_at_vertex(v, d, west.face, side="right")

    assert crossed == [south]
    # crossing a vertical crease flips the vertical component
    np.testing.assert_allclose(d_out, np.array([SQRT_HALF, -SQRT_HALF]), atol=1e-12)


def test_fan_raises_when_the_ray_arrives_along_a_crease():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])

    d = np.array([1.0, 0.0])  # exactly anti-parallel to the west crease
    with pytest.raises(DegenerateRayError):
        fan_at_vertex(v, d, west.face)


def test_fan_raises_when_it_wraps_the_whole_vertex():
    from pleat.example_graphs import rosette

    G = rosette(8)  # eight equal 45-degree sectors: theta oscillates forever
    G.recompute_lengths_and_angles()
    v = G.central_vertex()
    g = next(h for h in v.outgoing_iter() if h.face is not None)
    # aim so that theta_1 is small enough that theta never leaves (0, pi)
    axis = halfedge_direction(g)
    angle = np.arctan2(axis[1], axis[0]) - np.deg2rad(20)
    d = np.array([np.cos(angle), np.sin(angle)])

    with pytest.raises(DegenerateRayError):
        fan_at_vertex(v, d, g.face)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ray_casting.py -v -k fan`
Expected: FAIL — `ImportError: cannot import name 'fan_at_vertex'`

- [ ] **Step 3: Write minimal implementation**

Append to `pleat/ray_casting.py`:

```python
def fan_at_vertex(
    v: Vertex,
    d: np.ndarray,
    face: Face,
    side: str = "left",
) -> tuple[list[HalfEdge], np.ndarray, Face | None]:
    """Resolve a ray that hits vertex *v* head-on, as if it passed at distance eps.

    The ray arrives with direction *d* travelling inside *face*.  It is treated
    as passing an infinitesimal distance to the *side* of *v*, transmitting
    through every crease it would meet in quick succession.  The epsilon
    cancels out, so the rule is exact and takes no tolerance.

    Args:
        v: The vertex the ray hits.
        d: Unit direction of travel on arrival.
        face: The face the ray is travelling in on arrival.
        side: ``"left"`` or ``"right"`` -- which side of *v* the ray passes.

    Returns:
        ``(crossed, d_out, face_out)``: the half-edges transmitted through in
        order, the outgoing direction, and the face the ray leaves into
        (``None`` if it ran off the paper).

    Raises:
        DegenerateRayError: if the ray arrives exactly along a crease, or if it
            wraps the whole vertex (see the spec -- resolving that needs the
            holonomy of the full loop rather than this local walk).
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    s = 1.0 if side == "left" else -1.0

    # the boundary half-edge of `face` at v that the offset ray reaches first
    if s > 0:
        g = next(h for h in v.outgoing_iter() if h.face is face)
    else:
        g = next(h for h in v.outgoing_iter() if h.rev.face is face)

    theta = s * signed_angle(d, halfedge_direction(g))
    if abs(theta) < 1e-12:
        raise DegenerateRayError(f"ray arrives at {v} along a crease")

    d = np.asarray(d, dtype=float)
    crossed: list[HalfEdge] = []
    sign = 1.0
    degree = v.order()

    while 0 < theta < np.pi:
        if len(crossed) >= degree:
            raise DegenerateRayError(
                f"ray wraps the whole vertex {v}; resolving this needs the "
                "holonomy of the full loop, which is out of scope"
            )
        d = transmit(d, halfedge_direction(g))
        crossed.append(g)
        face = g.rev.face if s > 0 else g.face
        if face is None:
            return crossed, d, None
        theta += sign * (g.rev["in_angle"] if s > 0 else g.pre["in_angle"])
        sign = -sign
        g = g.rev.nex if s > 0 else g.pre.rev

    return crossed, d, face
```

Add `Face` and `Vertex` to the `from .half import ...` line at the top of the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: PASS, 13 passed

- [ ] **Step 5: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/ray_casting.py tests/test_ray_casting.py
git commit -m "feat(ray_casting): vertex fan rule"
```

---

### Task 3: Stepping a ray through faces

**Files:**
- Modify: `pleat/ray_casting.py`
- Test: `tests/test_ray_casting.py`

**Interfaces:**
- Consumes: everything from Tasks 1–2.
- Produces: `RayHit` dataclass with fields `halfedges: list[HalfEdge]`, `t: float`, `position: np.ndarray`, `vertex: Vertex | None`, `direction_in: np.ndarray`, `direction_out: np.ndarray`, `face: Face | None`; and `first_crossing(face, p, d, vertex_tol) -> tuple[HalfEdge, float] | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ray_casting.py`:

```python
from pleat.ray_casting import first_crossing


def test_first_crossing_finds_the_forward_edge_of_the_face():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])
    sw_face = west.face  # the unit square with corners (-0.5,0.5)..(-1.5,-0.5)

    p = np.array([-1.0, 0.0])  # its centre
    h, t = first_crossing(sw_face, p, np.array([1.0, 0.0]), vertex_tol=1e-9)

    assert np.allclose(h.orig["pos"], [-0.5, -0.5]) or np.allclose(
        h.dest["pos"], [-0.5, -0.5]
    )
    crossing = h.orig["pos"] + t * (h.dest["pos"] - h.orig["pos"])
    np.testing.assert_allclose(crossing, np.array([-0.5, 0.0]), atol=1e-12)


def test_first_crossing_ignores_edges_behind_the_ray():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])

    p = np.array([-1.0, 0.0])
    forward, _ = first_crossing(west.face, p, np.array([1.0, 0.0]), vertex_tol=1e-9)
    backward, _ = first_crossing(west.face, p, np.array([-1.0, 0.0]), vertex_tol=1e-9)
    assert forward is not backward


def test_first_crossing_returns_t_within_the_unit_interval():
    G = _grid()
    v = G.central_vertex()
    west = _outgoing_towards(v, [-1.5, 0.5])
    p = np.array([-1.0, 0.0])
    _, t = first_crossing(west.face, p, np.array([0.3, 1.0]), vertex_tol=1e-9)
    assert 0.0 <= t <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ray_casting.py -v -k first_crossing`
Expected: FAIL — `ImportError: cannot import name 'first_crossing'`

- [ ] **Step 3: Write minimal implementation**

Append to `pleat/ray_casting.py`:

```python
@dataclass
class RayHit:
    """One point at which a ray meets the graph.

    ``halfedges`` lists the creases transmitted through here, in order; it has
    more than one entry only at a vertex fan, and may be empty when the ray
    merely grazes a corner.  ``t`` is the parameter along ``halfedges[0]``.
    """

    halfedges: list[HalfEdge]
    t: float
    position: np.ndarray
    vertex: Vertex | None
    direction_in: np.ndarray
    direction_out: np.ndarray
    face: Face | None


def first_crossing(
    face: Face,
    p: np.ndarray,
    d: np.ndarray,
    vertex_tol: float,
) -> tuple[HalfEdge, float] | None:
    """Return the first crossing of the ray ``p + t*d`` with the boundary of *face*.

    Only *face* is inspected, which is what keeps the caster local.

    Returns:
        ``(halfedge, s)`` where *s* is the parameter along the half-edge, or
        ``None`` if the ray leaves through no edge (which should not happen in
        a well-formed face).
    """
    best: tuple[HalfEdge, float] | None = None
    best_t = np.inf
    for h in list(face.halfedge_iter()):
        a = h.orig["pos"]
        e = h.dest["pos"] - a
        denom = cross2(d, e)
        if abs(denom) < 1e-15:
            continue  # parallel to this edge
        t = cross2(a - p, e) / denom
        s = cross2(p - a, d) / cross2(e, d)
        edge_len = np.linalg.norm(e)
        # allow a hair of slack so leaving a vertex does not re-detect it
        if t <= vertex_tol or t >= best_t:
            continue
        if s < -vertex_tol / edge_len or s > 1 + vertex_tol / edge_len:
            continue
        best, best_t = (h, float(np.clip(s, 0.0, 1.0))), t
    return best
```

Add `from dataclasses import dataclass` to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: PASS, 16 passed

- [ ] **Step 5: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/ray_casting.py tests/test_ray_casting.py
git commit -m "feat(ray_casting): per-face crossing search and RayHit"
```

---

### Task 4: `cast_ray`, one direction

**Files:**
- Modify: `pleat/ray_casting.py`
- Test: `tests/test_ray_casting.py`

**Interfaces:**
- Consumes: `RayHit`, `first_crossing`, `fan_at_vertex`.
- Produces: `RayPath` dataclass with `hits: list[RayHit]`, `closed: bool`, `ends: tuple[str, str]`; and `cast_ray(G, halfedge, t, direction, side="left", both_ways=True, vertex_tol=None, max_steps=10_000) -> RayPath`. In this task `both_ways` is accepted and ignored; Task 5 implements it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ray_casting.py`:

```python
from pleat.ray_casting import cast_ray


def test_cast_ray_straight_across_a_grid_reaches_the_border():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])

    # start at the midpoint of a vertical edge, heading east: every crease it
    # meets is vertical, so it transmits straight through and never turns
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False)

    assert not path.closed
    assert path.ends[1] == "border"
    ys = [hit.position[1] for hit in path.hits]
    assert np.allclose(ys, ys[0])  # it really did travel in a straight line


def test_cast_ray_records_the_starting_point_as_its_first_hit():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False)

    assert path.hits[0].halfedges[0] in (north, north.rev)
    np.testing.assert_allclose(path.hits[0].position, np.array([-0.5, 1.0]), atol=1e-12)


def test_cast_ray_respects_max_steps():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])
    path = cast_ray(G, north, 0.5, np.array([1.0, 0.0]), both_ways=False, max_steps=2)

    assert path.ends[1] == "max_steps"
    assert len(path.hits) == 3  # the start plus two steps
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ray_casting.py -v -k cast_ray`
Expected: FAIL — `ImportError: cannot import name 'cast_ray'`

- [ ] **Step 3: Write minimal implementation**

Append to `pleat/ray_casting.py`:

```python
@dataclass
class RayPath:
    """A cast ray: its crossings, whether it closed, and how each end stopped."""

    hits: list[RayHit]
    closed: bool
    ends: tuple[str, str]


def default_vertex_tol(G) -> float:
    """Return a vertex-snapping tolerance scaled to the graph's edge lengths."""
    lengths = [h["length"] for h in G.halfedges if "length" in h]
    return 1e-9 * (float(np.mean(lengths)) if lengths else 1.0)


def _point_on(h: HalfEdge, t: float) -> np.ndarray:
    return h.orig["pos"] + t * (h.dest["pos"] - h.orig["pos"])


def _walk(G, start_he, start_t, direction, side, vertex_tol, max_steps):
    """Yield ``RayHit``s from the start point onwards, ending with a stop reason.

    The generator's return value (via ``StopIteration.value``) is the reason:
    ``'closed'``, ``'border'``, or ``'max_steps'``.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    p = _point_on(start_he, start_t)

    # the ray sets off into whichever side of the start edge it points at
    face = start_he.face if cross2(halfedge_direction(start_he), d) > 0 else start_he.rev.face

    yield RayHit(
        halfedges=[start_he],
        t=start_t,
        position=p,
        vertex=None,
        direction_in=d,
        direction_out=d,
        face=face,
    )

    for _ in range(max_steps):
        if face is None:
            return "border"
        found = first_crossing(face, p, d, vertex_tol)
        if found is None:
            return "border"
        h, s = found
        d_in = d
        edge = halfedge_direction(h)
        edge_len = np.linalg.norm(edge)

        vertex = None
        if s * edge_len <= vertex_tol:
            vertex = h.orig
        elif (1 - s) * edge_len <= vertex_tol:
            vertex = h.dest

        if vertex is not None:
            p = vertex["pos"]
            crossed, d, face = fan_at_vertex(vertex, d, face, side=side)
        else:
            p = _point_on(h, s)
            crossed, d, face = [h], transmit(d, edge), h.rev.face

        yield RayHit(
            halfedges=crossed,
            t=s,
            position=p,
            vertex=vertex,
            direction_in=d_in,
            direction_out=d,
            face=face,
        )
        if _closes(crossed, s, vertex, start_he, start_t, vertex_tol):
            return "closed"
    return "max_steps"


def _closes(crossed, s, vertex, start_he, start_t, vertex_tol) -> bool:
    """Return True if this hit is back at the ray's starting point."""
    if vertex is not None:
        return vertex in (start_he.orig, start_he.dest) and start_t in (0.0, 1.0)
    for h in crossed:
        if h is start_he:
            here = s
        elif h is start_he.rev:
            here = 1 - s
        else:
            continue
        if abs(here - start_t) * np.linalg.norm(halfedge_direction(start_he)) <= vertex_tol:
            return True
    return False


def cast_ray(
    G,
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    both_ways: bool = True,
    vertex_tol: float | None = None,
    max_steps: int = 10_000,
) -> RayPath:
    """Cast a ray from a point on an edge, transmitting at every crease it meets.

    Args:
        G: The crease pattern.
        halfedge: The edge the ray starts on.
        t: Parameter along *halfedge*; ``0`` and ``1`` mean its endpoints.
        direction: Initial direction of travel.
        side: Which side of a vertex the ray passes when it hits one head-on.
        both_ways: Cast backwards too if the forward ray does not close.
        vertex_tol: Distance below which a crossing snaps to a vertex.
        max_steps: Safety cap on the number of crossings per direction.

    Returns:
        A :class:`RayPath` whose hits run from the backward end to the forward
        end.
    """
    if vertex_tol is None:
        vertex_tol = default_vertex_tol(G)

    hits: list[RayHit] = []
    walker = _walk(G, halfedge, t, direction, side, vertex_tol, max_steps)
    try:
        while True:
            hits.append(next(walker))
    except StopIteration as stop:
        forward_reason = stop.value

    return RayPath(
        hits=hits, closed=forward_reason == "closed", ends=("start", forward_reason)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: PASS, 19 passed

- [ ] **Step 5: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/ray_casting.py tests/test_ray_casting.py
git commit -m "feat(ray_casting): cast_ray in one direction"
```

---

### Task 5: Casting both ways

**Files:**
- Modify: `pleat/ray_casting.py:cast_ray`
- Test: `tests/test_ray_casting.py`

**Interfaces:**
- Consumes: `_walk`, `RayPath`.
- Produces: `cast_ray(..., both_ways=True)` returning hits ordered backward-end → forward-end with `ends` describing both.

The backward pass uses the **mirrored side**: a forward ray with `side="left"` retraces as a backward ray with `side="right"`. A ray passing at `+eps` laterally while travelling along `d` is, travelling along `-d` down the same offset line, passing at `-eps`. Reusing the same side would trace a different path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ray_casting.py`:

```python
def test_both_ways_extends_a_border_to_border_ray():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])
    d = np.array([1.0, 0.0])

    one_way = cast_ray(G, north, 0.5, d, both_ways=False)
    two_way = cast_ray(G, north, 0.5, d, both_ways=True)

    assert two_way.ends == ("border", "border")
    assert len(two_way.hits) > len(one_way.hits)
    xs = [hit.position[0] for hit in two_way.hits]
    assert xs == sorted(xs)  # ordered backward end -> forward end
    ys = [hit.position[1] for hit in two_way.hits]
    assert np.allclose(ys, ys[0])


def test_both_ways_does_not_cast_backwards_when_the_ray_closes():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])
    # a ray at 45 degrees across a square grid closes on itself
    d = np.array([SQRT_HALF, SQRT_HALF])

    closed = cast_ray(G, north, 0.5, d, both_ways=True)
    if closed.closed:
        assert closed.ends == ("closed", "closed")
        assert cast_ray(G, north, 0.5, d, both_ways=False).hits == closed.hits
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ray_casting.py -v -k both_ways`
Expected: FAIL — `assert ('start', 'border') == ('border', 'border')`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `cast_ray` after the `vertex_tol` default with:

```python
    def run(direction, side_):
        collected: list[RayHit] = []
        walker = _walk(G, halfedge, t, direction, side_, vertex_tol, max_steps)
        try:
            while True:
                collected.append(next(walker))
        except StopIteration as stop:
            return collected, stop.value

    forward, forward_reason = run(direction, side)
    if forward_reason == "closed":
        return RayPath(hits=forward, closed=True, ends=("closed", "closed"))
    if not both_ways:
        return RayPath(hits=forward, closed=False, ends=("start", forward_reason))

    other = "right" if side == "left" else "left"
    backward, backward_reason = run(-np.asarray(direction, dtype=float), other)
    # both passes emit the shared start point first; drop the duplicate
    hits = list(reversed(backward[1:])) + forward
    return RayPath(hits=hits, closed=False, ends=(backward_reason, forward_reason))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: PASS, 21 passed

- [ ] **Step 5: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/ray_casting.py tests/test_ray_casting.py
git commit -m "feat(ray_casting): cast both ways when the ray does not close"
```

---

### Task 6: Materializing a ray as creases

**Files:**
- Modify: `pleat/ray_casting.py`
- Test: `tests/test_ray_casting.py`

**Interfaces:**
- Consumes: `_walk`, `cast_ray`, `default_vertex_tol`.
- Produces: `RAY_CREASE = "ray_crease"` attribute key, and `add_ray_creases(G, halfedge, t, direction, side="left", both_ways=True, vertex_tol=None, max_steps=10_000) -> tuple[list[HalfEdge], RayPath]` returning the new rim half-edges in traversal order.

Casting and building **interleave**: each step is materialized before the next is computed, so a ray that crosses an already-split edge always sees the current graph. `crossed.rev.face` gives the correct sub-face after a split, so no geometric face lookup is needed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ray_casting.py`:

```python
from pleat.ray_casting import RAY_CREASE, add_ray_creases


def test_add_ray_creases_keeps_the_graph_consistent():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])

    before = G.order()
    rim, path = add_ray_creases(G, north, 0.5, np.array([1.0, 0.0]))

    G.check_consistency()
    assert G.order() > before  # vertices were inserted along the way
    assert rim, "expected at least one rim half-edge"
    assert all(h[RAY_CREASE] for h in rim)


def test_add_ray_creases_rim_is_a_connected_path():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])
    rim, _ = add_ray_creases(G, north, 0.5, np.array([1.0, 0.0]))

    for a, b in zip(rim, rim[1:]):
        assert a.dest is b.orig


def test_add_ray_creases_handles_a_ray_crossing_the_same_edge_twice():
    G = _grid()
    v = G.central_vertex()
    north = _outgoing_towards(v, [-0.5, 1.5])
    # a 45-degree ray revisits edges; the interleaved build must cope
    add_ray_creases(G, north, 0.5, np.array([SQRT_HALF, SQRT_HALF]))
    G.check_consistency()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ray_casting.py -v -k add_ray_creases`
Expected: FAIL — `ImportError: cannot import name 'add_ray_creases'`

- [ ] **Step 3: Write minimal implementation**

Append to `pleat/ray_casting.py`:

```python
RAY_CREASE = "ray_crease"


def _vertex_at(G, h: HalfEdge, t: float, vertex_tol: float) -> Vertex:
    """Return the vertex at parameter *t* on *h*, subdividing the edge if needed."""
    edge_len = np.linalg.norm(halfedge_direction(h))
    if t * edge_len <= vertex_tol:
        return h.orig
    if (1 - t) * edge_len <= vertex_tol:
        return h.dest
    position = _point_on(h, t)
    _, v = G.subdivide_edge(h)
    v["pos"] = position
    return v


def add_ray_creases(
    G,
    halfedge: HalfEdge,
    t: float,
    direction: np.ndarray,
    side: str = "left",
    both_ways: bool = True,
    vertex_tol: float | None = None,
    max_steps: int = 10_000,
) -> tuple[list[HalfEdge], "RayPath"]:
    """Cast a ray and add it to *G* as new vertices and creases.

    Casting and building interleave, one segment at a time, so a ray that
    re-crosses an edge it has already split always sees the current graph.

    Returns:
        ``(rim, path)`` -- the new half-edges in traversal order, each tagged
        with :data:`RAY_CREASE`, and the :class:`RayPath` that was cast.
    """
    if vertex_tol is None:
        vertex_tol = default_vertex_tol(G)

    path = cast_ray(
        G,
        halfedge,
        t,
        direction,
        side=side,
        both_ways=both_ways,
        vertex_tol=vertex_tol,
        max_steps=max_steps,
    )

    rim: list[HalfEdge] = []
    previous: Vertex | None = None
    for hit in path.hits:
        if hit.vertex is not None:
            here = hit.vertex
        else:
            here = _vertex_at(G, hit.halfedges[0], hit.t, vertex_tol)
        if previous is not None and previous is not here:
            face = next(iter(previous.common_faces_iter(here)), None)
            if face is not None:
                h12, _ = G.subdivide_face(face, previous, here, **{RAY_CREASE: True})
                rim.append(h12)
        previous = here

    G.recompute_lengths_and_angles()
    return rim, path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ray_casting.py -v`
Expected: PASS, 24 passed

- [ ] **Step 5: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/ray_casting.py tests/test_ray_casting.py
git commit -m "feat(ray_casting): materialise a cast ray as creases"
```

---

### Task 7: Local flat-foldability of a crease assignment

**Files:**
- Modify: `pleat/flat_foldable.py`
- Test: `tests/test_flat_foldable.py`

**Interfaces:**
- Consumes: `pleat.overlap.CREASE_ASSIGNMENT`.
- Produces: `folded_crease_angles(v) -> np.ndarray` (the alternating prefix sum `psi`, length `deg(v)`), and `local_assignment_valid(v, tol=1e-9) -> tuple[bool, float]` returning `(valid, margin)`.

`psi_k = psi_{k-1} + (-1)^k * a_k` is where crease `k` lands in the folded state, and `psi_{2n}` **is** the Kawasaki sum, so Kawasaki is the statement that the cycle closes rather than a separate condition. Every remaining condition depends only on which `psi_k` coincide, so tolerance enters exactly once. `margin` is the smallest inter-cluster gap; a `margin` near `tol` means the vertex is genuinely ambiguous, not that the verdict is wrong.

The crimp recursion backtracks over **weakly** minimal sectors. Requiring a strict minimum stalls on ties; greedily taking one weakly-minimal sector is unsound, because with a tie big-little-big does not force the bounding creases to differ, so a valid assignment gets rejected. Degrees are at most about twelve, so exhaustive backtracking is free.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_flat_foldable.py`:

```python
from pleat.example_graphs import rosette
from pleat.flat_foldable import folded_crease_angles, local_assignment_valid


def _degree_4_vertex():
    G = EuclideanPositionHEG(other=rosette(n=4))
    G.recompute_lengths_and_angles()
    return G, next(v for v in G.vertices if not v.on_border())


def test_folded_crease_angles_final_entry_is_the_kawasaki_sum():
    from pleat.flat_foldable import kawasaki_sum

    G, v = _degree_4_vertex()
    psi = folded_crease_angles(v)
    assert len(psi) == v.order()
    # psi is the running alternating sum, so its last entry is exactly the
    # Kawasaki sum -- Kawasaki is "the cycle closes", not a separate condition
    assert psi[-1] == pytest.approx(kawasaki_sum(v), abs=1e-9)


def test_symmetric_degree_4_vertex_accepts_every_3_to_1_assignment():
    G, v = _degree_4_vertex()
    creases = list(v.outgoing_iter())
    assert len(creases) == 4

    for odd in range(4):
        for i, h in enumerate(creases):
            value = VALLEY if i == odd else MOUNTAIN
            h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = value
        valid, margin = local_assignment_valid(v)
        assert valid, f"odd crease {odd} should be valid"
        assert margin >= 0.0


def test_symmetric_degree_4_vertex_rejects_a_2_to_2_assignment():
    G, v = _degree_4_vertex()
    creases = list(v.outgoing_iter())
    for i, h in enumerate(creases):
        value = VALLEY if i < 2 else MOUNTAIN
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = value
    valid, _ = local_assignment_valid(v)
    assert not valid


def test_symmetric_vertex_reports_a_small_margin():
    G, v = _degree_4_vertex()
    creases = list(v.outgoing_iter())
    for h in creases:
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = MOUNTAIN
    creases[0][CREASE_ASSIGNMENT] = creases[0].rev[CREASE_ASSIGNMENT] = VALLEY
    _, margin = local_assignment_valid(v)
    # all four sectors are equal, so the folded creases collapse into two
    # clusters -- the vertex is symmetric, and the margin says so
    assert margin < np.pi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flat_foldable.py -v`
Expected: FAIL — `ImportError: cannot import name 'folded_crease_angles'`

- [ ] **Step 3: Write minimal implementation**

Append to `pleat/flat_foldable.py`:

```python
def folded_crease_angles(v: Vertex) -> np.ndarray:
    """Return the folded positions ``psi`` of the creases around *v*.

    ``psi_k = psi_{k-1} + (-1)^k * a_k`` over the sector angles ``a_k`` in
    counter-clockwise order.  ``psi[-1]`` is the Kawasaki sum, so Kawasaki's
    theorem is the statement that this cycle closes rather than an independent
    condition.  The same alternating sum drives the vertex fan rule in
    :mod:`pleat.ray_casting`.
    """
    angles = np.abs(np.array([h["in_angle"] for h in v.incoming_iter()]))
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    return np.cumsum(angles * (-1.0) ** np.arange(len(angles)))


def _cluster_margin(values: np.ndarray, tol: float) -> float:
    """Return the smallest gap between distinct clusters of *values*."""
    gaps = np.diff(np.sort(values))
    distinct = gaps[gaps > tol]
    return float(distinct.min()) if len(distinct) else 0.0


def _crimp(angles: list[float], mv: list[int], i: int) -> tuple[list[float], list[int]]:
    """Fold sector *i* inside: drop creases ``i`` and ``i+1``, merge three sectors.

    Both lists are rotated so the crimp sits at the front, which keeps sectors
    and creases aligned without any modular index juggling.
    """
    n = len(angles)
    k = (i - 1) % n
    a = angles[k:] + angles[:k]  # a[0], a[1], a[2] = sectors i-1, i, i+1
    m = mv[k:] + mv[:k]  # m[0], m[1], m[2] = creases i-1, i, i+1
    merged = a[0] - a[1] + a[2]
    return [merged] + a[3:], [m[0]] + m[3:]


def _crimp_ok(angles: list[float], mv: list[int], tol: float) -> bool:
    """Return True if a sequence of crimps folds this vertex flat.

    ``angles[i]`` is the sector between creases ``mv[i]`` and ``mv[i + 1]``
    cyclically.  Backtracks over every weakly-minimal sector: with a tie,
    big-little-big does not force the bounding creases to differ, so committing
    to one candidate would reject valid assignments.
    """
    n = len(angles)
    if n <= 2:
        return True
    smallest = min(angles)
    for i in range(n):
        if angles[i] > smallest + tol:
            continue
        if mv[i] == mv[(i + 1) % n]:
            continue  # big-little-big: the bounding creases must differ
        new_angles, new_mv = _crimp(angles, mv, i)
        if new_angles[0] < -tol:
            continue  # the crimp would need more paper than there is
        new_angles[0] = max(new_angles[0], 0.0)
        if _crimp_ok(new_angles, new_mv, tol):
            return True
    return False


def local_assignment_valid(v: Vertex, tol: float = 1e-9) -> tuple[bool, float]:
    """Check the crease assignment at interior vertex *v* folds flat locally.

    Returns:
        ``(valid, margin)``.  *margin* is the smallest gap between distinct
        folded crease positions; a margin near *tol* means the vertex is
        symmetric enough that the verdict depends on tie-breaking, not that it
        is wrong.

    Requires ``pleat.overlap.CREASE_ASSIGNMENT`` on every half-edge at *v*.
    """
    from .overlap import CREASE_ASSIGNMENT

    angles = [abs(h["in_angle"]) for h in v.incoming_iter()]
    mv = [h[CREASE_ASSIGNMENT] for h in v.incoming_iter()]
    if len(angles) % 2 != 0:
        return False, 0.0

    psi = folded_crease_angles(v)
    margin = _cluster_margin(psi, tol)
    if abs(psi[-1]) > max(tol, 1e-8) * len(angles):
        return False, margin  # Kawasaki: the cycle does not close
    return _crimp_ok(angles, mv, max(tol, 1e-8)), margin
```

Then replace the Maekawa branch of `is_locally_flat_foldable` — the lines

```python
        if all(CREASE_ASSIGNMENT in h.attributes for h in v.outgoing_iter()) and not maekawa_check(v):
            violations[v] = "Maekawa's theorem violated (|#mountains - #valleys| != 2)"
```

with

```python
        if all(CREASE_ASSIGNMENT in h.attributes for h in v.outgoing_iter()):
            valid, margin = local_assignment_valid(v)
            if not valid:
                violations[v] = f"crease assignment does not fold flat (margin {margin:.3e})"
```

`maekawa_check` stays public as a cheap necessary condition.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_flat_foldable.py -v`
Expected: PASS — including the pre-existing `test_shrink_rotate_pattern_is_locally_flat_foldable`, which now exercises the crimp recursion on a real pattern.

If that pre-existing test fails, the crimp recursion is wrong, not the pattern — debug against it before continuing.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -m "not slow" -q`
Expected: no new failures.

- [ ] **Step 6: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/flat_foldable.py tests/test_flat_foldable.py
git commit -m "feat(flat_foldable): local flat-foldability via folded crease positions"
```

---

### Task 8: Open sink fold

**Files:**
- Create: `pleat/sink.py`
- Test: `tests/test_sink.py`

**Interfaces:**
- Consumes: `add_ray_creases`, `RAY_CREASE`, `RayPath` from `pleat.ray_casting`; `local_assignment_valid` from `pleat.flat_foldable`; `CREASE_ASSIGNMENT`, `MOUNTAIN`, `VALLEY` from `pleat.overlap`.
- Produces: `InvalidSinkError(ValueError)` and `open_sink(G, halfedge, t, direction, side="left", strict=True, **cast_kwargs) -> list[HalfEdge]`.

The rim of an open sink is uniformly `MOUNTAIN` or uniformly `VALLEY`, so step 3 is a two-candidate test rather than a search. The degree-4 case falls out of Task 7 rather than being special-cased.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sink.py`:

```python
"""Tests for sink folds."""

from __future__ import annotations

import numpy as np
import pytest

from pleat.example_graphs import from_tiles
from pleat.example_tilesets import platonic
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from pleat.sink import InvalidSinkError, open_sink

SQRT_HALF = np.sqrt(0.5)


def _grid():
    G = from_tiles(platonic(n=4), rings=2)
    G.recompute_lengths_and_angles()
    return G


def _assign_all(G, value=MOUNTAIN):
    for h in G.halfedges:
        h[CREASE_ASSIGNMENT] = value


def _start(G):
    v = G.central_vertex()
    north = next(
        h for h in v.outgoing_iter() if np.allclose(h.dest["pos"], [-0.5, 1.5], atol=1e-9)
    )
    return north


def test_open_sink_without_assignment_raises_when_strict():
    G = _grid()
    with pytest.raises(InvalidSinkError):
        open_sink(G, _start(G), 0.5, np.array([SQRT_HALF, SQRT_HALF]), strict=True)


def test_open_sink_without_assignment_still_builds_geometry_when_not_strict():
    G = _grid()
    before = G.order()
    rim = open_sink(G, _start(G), 0.5, np.array([SQRT_HALF, SQRT_HALF]), strict=False)
    G.check_consistency()
    assert G.order() > before
    assert rim


def test_open_sink_rim_is_uniform():
    G = _grid()
    _assign_all(G)
    rim = open_sink(G, _start(G), 0.5, np.array([SQRT_HALF, SQRT_HALF]), strict=False)
    values = {h[CREASE_ASSIGNMENT] for h in rim}
    assert len(values) == 1
    assert values <= {MOUNTAIN, VALLEY}


def test_open_sink_inverts_the_interior():
    G = _grid()
    _assign_all(G, MOUNTAIN)
    rim = open_sink(G, _start(G), 0.5, np.array([SQRT_HALF, SQRT_HALF]), strict=False)
    inverted = [h for h in G.halfedges if h.get(CREASE_ASSIGNMENT) == VALLEY]
    assert inverted, "expected the interior of the sink to be inverted"


def test_open_sink_geometry_is_the_same_with_and_without_strict():
    a, b = _grid(), _grid()
    _assign_all(a)
    rim_a = open_sink(a, _start(a), 0.5, np.array([SQRT_HALF, SQRT_HALF]), strict=False)
    rim_b = open_sink(b, _start(b), 0.5, np.array([SQRT_HALF, SQRT_HALF]), strict=False)
    assert len(rim_a) == len(rim_b)
    assert a.order() == b.order()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sink.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pleat.sink'`

- [ ] **Step 3: Write minimal implementation**

Create `pleat/sink.py`:

```python
"""Sink folds on crease patterns.

An open sink pushes a folded point through the model.  In the crease pattern
this is a rim traced by a ray (see :mod:`pleat.ray_casting`), every crease
strictly inside the rim inverted, and the rim itself assigned uniformly
mountain or valley -- which of the two is inferred from the node-wise local
flat-foldability conditions.
"""

from __future__ import annotations

import logging

from .flat_foldable import local_assignment_valid
from .half import Face, HalfEdge
from .overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY
from .ray_casting import RAY_CREASE, add_ray_creases

logger = logging.getLogger(__name__)


class InvalidSinkError(ValueError):
    """The requested sink fold does not yield a valid crease pattern."""


def _interior_faces(rim: list[HalfEdge]) -> set[Face]:
    """Flood fill the faces enclosed by *rim*, bounded by the rim and the border."""
    blocked = {h for h in rim} | {h.rev for h in rim}
    seen: set[Face] = set()
    stack = [h.face for h in rim if h.face is not None]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        for h in list(f.halfedge_iter()):
            if h in blocked or h.rev.face is None:
                continue
            if h.rev.face not in seen:
                stack.append(h.rev.face)
    return seen


def _rim_nodes(rim: list[HalfEdge]):
    """Yield the interior vertices along *rim* (its endpoints carry no condition)."""
    for h in rim:
        for v in (h.orig, h.dest):
            if not v.on_border():
                yield v


def open_sink(
    G,
    halfedge: HalfEdge,
    t: float,
    direction,
    side: str = "left",
    strict: bool = True,
    **cast_kwargs,
) -> list[HalfEdge]:
    """Add an open sink fold to *G*, starting from a point on *halfedge*.

    Casts a ray to trace the rim, adds it to the graph, inverts every crease
    strictly inside it, and infers whether the rim is mountain or valley.

    Args:
        G: The crease pattern.
        halfedge: The edge the rim starts on.
        t: Parameter along *halfedge*; ``0`` and ``1`` mean its endpoints.
        direction: Initial direction of the rim.
        side: Which side of a vertex the ray passes when it hits one head-on.
        strict: If True, raise when the crease assignment is missing or cannot
            be made valid.  If False, warn and carry on -- the geometry is
            unaffected either way, so this makes ``open_sink`` usable on a
            pattern whose creases have not been assigned yet.
        **cast_kwargs: Forwarded to :func:`pleat.ray_casting.add_ray_creases`.

    Returns:
        The rim half-edges in traversal order.

    Raises:
        InvalidSinkError: if the ray hit the step cap, or (when *strict*) the
            crease assignment is missing or no uniform rim assignment works.
    """
    rim, path = add_ray_creases(
        G, halfedge, t, direction, side=side, **cast_kwargs
    )
    if "max_steps" in path.ends:
        raise InvalidSinkError("the sink rim did not terminate within max_steps")

    interior = _interior_faces(rim)
    rim_edges = {h for h in rim} | {h.rev for h in rim}
    inner = [
        h
        for f in interior
        for h in list(f.halfedge_iter())
        if h not in rim_edges and not h.on_border()
    ]

    missing = [h for h in inner if CREASE_ASSIGNMENT not in h.attributes]
    if missing and strict:
        raise InvalidSinkError(
            f"{len(missing)} creases inside the sink have no assignment; "
            "pass strict=False to sink anyway"
        )
    for h in inner:
        if CREASE_ASSIGNMENT in h.attributes:
            h[CREASE_ASSIGNMENT] = -h[CREASE_ASSIGNMENT]

    nodes = list(_rim_nodes(rim))
    for candidate in (MOUNTAIN, VALLEY):
        for h in rim:
            h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = candidate
        failures = []
        for v in nodes:
            if not all(CREASE_ASSIGNMENT in h.attributes for h in v.outgoing_iter()):
                continue  # incompletely assigned; nothing to check here
            valid, margin = local_assignment_valid(v)
            if not valid:
                failures.append(v)
            elif margin < 1e-6:
                logger.info("sink rim node %s is degenerate (margin %.3e)", v, margin)
        if not failures:
            return rim

    message = f"no uniform rim assignment folds flat; failing vertices: {failures}"
    if strict:
        raise InvalidSinkError(message)
    logger.warning("%s; leaving the rim as MOUNTAIN", message)
    for h in rim:
        h[CREASE_ASSIGNMENT] = h.rev[CREASE_ASSIGNMENT] = MOUNTAIN
    return rim
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sink.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -m "not slow" -q`
Expected: no new failures.

- [ ] **Step 6: Format and commit**

```bash
uv run --extra dev black pleat tests
git add pleat/sink.py tests/test_sink.py
git commit -m "feat(sink): open sink fold"
```

---

### Task 9: Export and document

**Files:**
- Modify: `pleat/__init__.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `pleat.ray_casting` and `pleat.sink` importable as `pleat.<name>`, consistent with the other modules.

`pleat/__init__.py` eagerly imports every public submodule as `import pleat.<name>`, grouped by topic with comment headers and alphabetical within each group.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sink.py`:

```python
def test_modules_are_importable_from_the_package():
    import pleat

    assert hasattr(pleat, "ray_casting")
    assert hasattr(pleat, "sink")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sink.py -k importable -v`
Expected: FAIL — `AssertionError` on `hasattr(pleat, "ray_casting")`

- [ ] **Step 3: Add the exports**

In `pleat/__init__.py`, add to the "Topology and origami pipeline" group:

```python
import pleat.ray_casting
```

and to the same group (alphabetical, after `pleat.prototiles` if present, else at the end of its group):

```python
import pleat.sink
```

- [ ] **Step 4: Document the modules in AGENTS.md**

In the "Other Modules" section of `AGENTS.md`, add:

```markdown
- `ray_casting.py`: Cast a ray through a crease pattern, transmitting at every crease (`d - 2(d·û)û`) and resolving head-on vertex hits by an angular fan walk. `add_ray_creases()` materialises the ray as new vertices and creases. Strictly local — a step only ever inspects the current face.
- `sink.py`: Open sink folds. `open_sink()` traces a rim with `ray_casting`, inverts the creases inside it, and infers the uniform rim assignment from node-wise local flat-foldability.
```

And in the "Origami / Crease Patterns" section, extend the `flat_foldable.py` note to mention `local_assignment_valid()`.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -m "not slow" -q`
Expected: all pass.

```bash
uv run --extra dev black pleat tests
git add pleat/__init__.py AGENTS.md tests/test_sink.py
git commit -m "docs: document ray_casting and sink modules"
```

---

## Notes for the implementer

- **The test oracle in Task 2 is hand-computed.** If `test_fan_transmits_through_several_creases_at_one_vertex` fails, re-derive rather than adjusting the expected value: `theta` should run 120° → 165° → 120° → 210°, and the direction 60° → 120° → −30° → 30°.
- **Task 7 changes existing behaviour.** `is_locally_flat_foldable` becomes stricter. `test_shrink_rotate_pattern_is_locally_flat_foldable` is the canary — it must keep passing.
- **Task 8's `_interior_faces` seeds from `h.face` for every rim half-edge.** Whether that is the inside or the outside depends on the rim's orientation, and the rim is traced consistently, so it is the same side throughout. If the flood fill escapes to the whole graph, the seed side is wrong — seed from `h.rev.face` instead and add a test pinning the orientation.
- The `RayHit.t` for a vertex hit refers to `halfedges[0]`, which at a fan is only the first crease transmitted through. Consumers should branch on `hit.vertex is not None` before using `t`.
- **Task 7 assumes `angles[i]` is the sector between crease `i` and crease `i+1`.** That holds if `h["in_angle"]` for an incoming half-edge is the sector immediately counter-clockwise of `h`, matching `incoming_iter`'s counter-clockwise order. If the crimp recursion rejects patterns it should accept, an off-by-one in this alignment is the first thing to check — try `angles = angles[1:] + angles[:1]`.
