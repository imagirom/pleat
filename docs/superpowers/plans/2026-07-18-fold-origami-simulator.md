# FOLD export/import + "Open in Origami Simulator" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FOLD (v1.2) crease-pattern import/export to pleat and let a crease pattern be opened in [Origami Simulator](https://origamisimulator.org/) from a notebook, a script, or the online docs.

**Architecture:** Promote `pleat/io.py` to a subpackage `pleat/io/` and add `io/fold.py`. FOLD serialization maps pleat's half-edge graph to/from FOLD's vertex/edge/face arrays. The Origami Simulator launcher writes a self-contained HTML page (an OS iframe + a `postMessage` handshake carrying the FOLD as JSON) and opens it with the stdlib `webbrowser`; a lighter inline-button variant renders in browser Jupyter and the built docs.

**Tech Stack:** Python (numpy), stdlib `json`/`tempfile`/`webbrowser`, the existing `pleat.half` DCEL, `pleat.overlap` crease constants. No new dependencies.

## Global Constraints

- Python `>= 3.10` (project floor). Use `from __future__ import annotations`.
- No new runtime dependencies — stdlib only for the launcher.
- FOLD target version: `file_spec` = `1.2`, `file_creator` = `"pleat"`.
- Crease model (from `pleat.overlap`): `CREASE_ASSIGNMENT = "crease_assignment"`, `MOUNTAIN = 1`, `VALLEY = -1`, unassigned = `0`; assignment is stored on a half-edge and mirrored on its `.rev`.
- FOLD `edges_assignment` letters used: `"M"`, `"V"`, `"B"` (border), `"U"` (unassigned). `edges_foldAngle`: `M → -180.0`, `V → +180.0`, else `null`.
- Scope: **Euclidean 2D crease patterns only.** Non-Euclidean geometries stay on `.heg`; this change adds FOLD *alongside* `.heg`, it does not retire it.
- Deterministic output: order vertices/edges/faces by their `["id"]` attribute so serialization is reproducible.
- Pre-release project: no back-compat shims. The `io/__init__.py` re-exports are the package's public surface, not a shim.

## File Structure

```
pleat/io/                 (was pleat/io.py — promoted to a package)
  __init__.py             re-exports every public name (heg + circlepack + fold)
  heg.py                  MOVED verbatim from io.py: .heg YAML format
  circlepack.py           MOVED verbatim from io.py: CirclePack .p format
  fold.py                 NEW: FOLD serialization + Origami Simulator launcher
pleat/half.py             +1 convenience method on the graph class
pleat/__init__.py         +2 top-level re-exports (headline launcher entry points)
tests/test_fold.py        NEW: FOLD round-trip + FOLD validity + HTML content
tests/test_io.py          UNCHANGED (must keep passing after the split)
docs/notebooks/Saving_and_Exporting.ipynb   +section
```

---

### Task 1: Promote `pleat/io.py` to a subpackage (mechanical split)

Pure refactor, no behavior change. The existing `tests/test_io.py` is the safety net.

**Files:**
- Delete: `pleat/io.py`
- Create: `pleat/io/__init__.py`, `pleat/io/heg.py`, `pleat/io/circlepack.py`
- Test: `tests/test_io.py` (existing — must still pass unchanged)

**Interfaces:**
- Produces: `pleat.io.{graph_to_dict, dict_to_graph, save_graph, load_graph, CirclePackData, parse_p_file, write_p_file, load_circlepack, save_circlepack}` — all importable from `pleat.io` exactly as before.

- [ ] **Step 1: Baseline the safety net**

Run: `python -m pytest tests/test_io.py -q`
Expected: PASS (this is the behavior we must preserve).

- [ ] **Step 2: Create `pleat/io/heg.py`**

Move the `.heg` YAML section of the old `io.py` here verbatim — the module docstring, imports (`os`, `copy.copy`, `numpy as np`, `yaml`, `import pleat`, and `from ..geometries import EuclideanGeometry, PoincareDiskModel`, `from ..half import EuclideanPositionHEG, Face, HalfEdge, HalfEdgeGraph, Vertex, rotate_by`), and these functions unchanged: `graph_to_dict`, `dict_to_graph`, `save_graph`, `load_graph`.

Note the relative-import depth changes from `.geometries`/`.half` to `..geometries`/`..half` (one level deeper now).

- [ ] **Step 3: Create `pleat/io/circlepack.py`**

Move the CirclePack `.p` section here verbatim: the section-comment banner, `from dataclasses import dataclass`, numpy import, `from ..geometries import EuclideanGeometry, PoincareDiskModel`, `from ..half import EuclideanPositionHEG, Face, HalfEdge, Vertex, rotate_by`, and these unchanged: `CirclePackData`, `parse_p_file`, `write_p_file`, `_build_heg_from_data`, `_r_eucl_from_x_and_center`, `load_circlepack`, `_graph_to_circlepack_data`, `save_circlepack`. (`load_circlepack` calls `_build_heg_from_data` — they stay together here.) Deferred imports inside functions (`from ..circle_packing import ...`) keep working; just verify the `.circle_packing` → `..circle_packing` depth.

- [ ] **Step 4: Create `pleat/io/__init__.py`**

```python
"""File I/O for pleat graphs: the ``.heg`` half-edge format, the CirclePack
``.p`` format, and the FOLD crease-pattern format."""

from __future__ import annotations

from .circlepack import (
    CirclePackData,
    load_circlepack,
    parse_p_file,
    save_circlepack,
    write_p_file,
)
from .heg import dict_to_graph, graph_to_dict, load_graph, save_graph

__all__ = [
    "graph_to_dict",
    "dict_to_graph",
    "save_graph",
    "load_graph",
    "CirclePackData",
    "parse_p_file",
    "write_p_file",
    "load_circlepack",
    "save_circlepack",
]
```

(FOLD names get added to this file in Task 3.)

- [ ] **Step 5: Delete `pleat/io.py`**

```bash
git rm pleat/io.py
```

- [ ] **Step 6: Verify the split preserved behavior**

Run: `python -m pytest tests/test_io.py -q && python -c "import pleat; pleat.io.load_graph('graphs/irregular2.heg').check_consistency(); print('ok')"`
Expected: PASS, then `ok`. (The second check mirrors `tests/test_intersecting_cylinders.py`'s use of `pleat.io.load_graph`.)

- [ ] **Step 7: Commit**

```bash
git add pleat/io/ tests/
git commit -m "refactor: promote pleat.io to a subpackage (heg + circlepack)"
```

---

### Task 2: FOLD serialization — `graph_to_fold` / `fold_to_graph` / `save_fold` / `load_fold`

**Files:**
- Create: `pleat/io/fold.py`
- Test: `tests/test_fold.py`

**Interfaces:**
- Consumes: `pleat.overlap.{CREASE_ASSIGNMENT, MOUNTAIN, VALLEY}`; `pleat.half.{EuclideanPositionHEG, Vertex, HalfEdge, Face}`.
- Produces:
  - `graph_to_fold(G, *, title: str | None = None) -> dict`
  - `fold_to_graph(fold: dict) -> EuclideanPositionHEG`
  - `save_fold(path: str, G, *, overwrite: bool = False) -> None` (appends `.fold`)
  - `load_fold(path: str) -> EuclideanPositionHEG`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fold.py`:

```python
"""Tests for pleat.io.fold: FOLD round-trip, FOLD validity, and OS launcher HTML."""

from __future__ import annotations

import numpy as np

from pleat.example_graphs import rosette
from pleat.half import EuclideanPositionHEG
from pleat.io.fold import fold_to_graph, graph_to_fold, load_fold, save_fold
from pleat.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY

VALID_ASSIGNMENTS = {"M", "V", "B", "F", "U"}


def _creased_rosette():
    """A hexagonal rosette (6 triangles) with its interior spokes creased M/V."""
    G = EuclideanPositionHEG(other=rosette(n=6))
    interior = [h for h in G.halfedges if not h.on_border() and not h.rev.on_border()]
    for i, h in enumerate(interior):
        a = MOUNTAIN if i % 2 == 0 else VALLEY
        h[CREASE_ASSIGNMENT] = a
        h.rev[CREASE_ASSIGNMENT] = a
    return G


def test_graph_to_fold_is_valid_fold():
    G = _creased_rosette()
    fold = graph_to_fold(G)
    assert fold["file_spec"] == 1.2
    assert fold["file_creator"] == "pleat"
    assert fold["frame_classes"] == ["creasePattern"]
    n_v = len(fold["vertices_coords"])
    n_e = len(fold["edges_vertices"])
    assert len(fold["edges_assignment"]) == n_e
    assert len(fold["edges_foldAngle"]) == n_e
    for a in fold["edges_assignment"]:
        assert a in VALID_ASSIGNMENTS
    for ang in fold["edges_foldAngle"]:
        assert ang is None or -180.0 <= ang <= 180.0
    for u, v in fold["edges_vertices"]:
        assert 0 <= u < n_v and 0 <= v < n_v
    # every interior spoke is creased, the outer hexagon is border
    assert fold["edges_assignment"].count("B") == 6
    assert set(fold["edges_assignment"]) >= {"M", "V", "B"}


def test_fold_roundtrip_preserves_topology_and_creases():
    G = _creased_rosette()
    fold = graph_to_fold(G)
    G2 = fold_to_graph(fold)
    G2.check_consistency()
    assert (len(G.vertices), len(G.halfedges), len(G.faces)) == (
        len(G2.vertices),
        len(G2.halfedges),
        len(G2.faces),
    )
    # crease assignments survive (as a multiset over undirected edges)
    def crease_multiset(g):
        seen, out = set(), []
        for h in g.halfedges:
            if h in seen:
                continue
            seen.add(h)
            seen.add(h.rev)
            out.append(h.attributes.get(CREASE_ASSIGNMENT, 0))
        return sorted(out)

    assert crease_multiset(G) == crease_multiset(G2)


def test_save_load_fold_roundtrip(tmp_path):
    G = _creased_rosette()
    path = str(tmp_path / "rose")
    save_fold(path, G)
    assert (tmp_path / "rose.fold").exists()
    G2 = load_fold(str(tmp_path / "rose.fold"))
    G2.check_consistency()
    assert len(G.faces) == len(G2.faces)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fold.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pleat.io.fold'`.

- [ ] **Step 3: Implement the serialization**

Create `pleat/io/fold.py`:

```python
"""FOLD (v1.2) crease-pattern I/O and an Origami Simulator launcher.

FOLD spec: https://github.com/edemaine/fold/blob/main/doc/spec.md
Scope: Euclidean 2D crease patterns. See docs/superpowers/specs for the design.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..half import EuclideanPositionHEG, Face, HalfEdge, Vertex
from ..overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY

_ASSIGN_TO_LETTER = {MOUNTAIN: "M", VALLEY: "V"}
_LETTER_TO_ASSIGN = {"M": MOUNTAIN, "V": VALLEY}
_FOLD_ANGLE = {"M": -180.0, "V": 180.0}


def _coords2d(pos) -> list[float]:
    """Return a plain ``[x, y]`` from a Euclidean position (2-vector or complex)."""
    if np.iscomplexobj(pos) and np.ndim(pos) == 0:
        c = complex(pos)
        return [c.real, c.imag]
    arr = np.asarray(pos, dtype=float).ravel()
    return [float(arr[0]), float(arr[1])]


def graph_to_fold(G, *, title: str | None = None) -> dict:
    """Serialise a Euclidean crease-pattern graph to a FOLD v1.2 dict.

    Undirected edges are the rev-pairs of ``G.halfedges``. Each edge's
    assignment comes from :data:`CREASE_ASSIGNMENT` (M/V), or ``"B"`` when either
    side is a border half-edge, or ``"U"`` otherwise. Faces are ``G.faces`` (the
    outer region is not a Face in pleat), each emitted as its CCW vertex loop.
    """
    verts = sorted(G.vertices, key=lambda v: v["id"])
    vidx = {v: i for i, v in enumerate(verts)}

    vertices_coords = [_coords2d(v["pos"]) for v in verts]

    edges_vertices: list[list[int]] = []
    edges_assignment: list[str] = []
    edges_foldAngle: list[float | None] = []
    seen: set = set()
    for h in sorted(G.halfedges, key=lambda h: h["id"]):
        if h in seen:
            continue
        seen.add(h)
        seen.add(h.rev)
        edges_vertices.append([vidx[h.orig], vidx[h.dest]])
        if h.on_border() or h.rev.on_border():
            letter = "B"
        else:
            letter = _ASSIGN_TO_LETTER.get(h.attributes.get(CREASE_ASSIGNMENT, 0), "U")
        edges_assignment.append(letter)
        edges_foldAngle.append(_FOLD_ANGLE.get(letter))

    faces_vertices = [
        [vidx[v] for v in sorted_face_vertices(f)]
        for f in sorted(G.faces, key=lambda f: f["id"])
    ]

    fold = {
        "file_spec": 1.2,
        "file_creator": "pleat",
        "file_classes": ["singleModel"],
        "frame_classes": ["creasePattern"],
        "frame_attributes": ["2D"],
        "vertices_coords": vertices_coords,
        "edges_vertices": edges_vertices,
        "edges_assignment": edges_assignment,
        "edges_foldAngle": edges_foldAngle,
        "faces_vertices": faces_vertices,
    }
    if title is not None:
        fold["file_title"] = title
    return fold


def sorted_face_vertices(f: Face) -> list[Vertex]:
    """Return the face's boundary vertices in CCW order."""
    return list(f.vertex_iter())


def fold_to_graph(fold: dict) -> EuclideanPositionHEG:
    """Reconstruct a Euclidean half-edge graph from a FOLD dict.

    Requires ``faces_vertices`` (needs oriented faces to rebuild the DCEL).
    Interior edges are twinned across their two faces; boundary edges get a
    border twin (``face=None``) linked into the outer cycle. ``vertices_coords``
    restores positions and ``edges_assignment`` restores M/V creases.
    """
    coords = fold["vertices_coords"]
    faces_vertices = fold.get("faces_vertices")
    if not faces_vertices:
        raise ValueError(
            "FOLD frame has no faces_vertices; cannot reconstruct a face-based "
            "half-edge graph (only creasePattern/foldedForm frames with faces "
            "are supported)."
        )

    G = EuclideanPositionHEG()
    verts = [Vertex() for _ in coords]
    for v, c in zip(verts, coords):
        xy = [float(c[0]), float(c[1])] if len(c) >= 2 else [float(c[0]), 0.0]
        v["pos"] = np.array(xy)
    G.add_vertices(verts)

    # 1. interior half-edges from each face loop
    he: dict[tuple[int, int], HalfEdge] = {}
    all_halfedges: list[HalfEdge] = []
    for face_vs in faces_vertices:
        n = len(face_vs)
        loop = []
        for k in range(n):
            i, j = face_vs[k], face_vs[(k + 1) % n]
            h = HalfEdge(orig=verts[i], dest=verts[j])
            he[(i, j)] = h
            loop.append(h)
        f = Face(any_side=loop[0])
        for k in range(n):
            h = loop[k]
            h.nex = loop[(k + 1) % n]
            h.pre = loop[(k - 1) % n]
            h.face = f
            verts[face_vs[k]].any_outgoing = h
        all_halfedges.extend(loop)
        G.add_halfedges(loop)
        G.add_face(f)

    # 2. twin interior edges; create border twins for unmatched (boundary) edges
    border: list[HalfEdge] = []
    for (i, j), h in list(he.items()):
        if (j, i) in he:
            h.rev = he[(j, i)]
        elif h.rev is None:
            b = HalfEdge(orig=verts[j], dest=verts[i], face=None)
            b.rev = h
            h.rev = b
            he[(j, i)] = b
            border.append(b)
            verts[j].any_outgoing = verts[j].any_outgoing or b

    # 3. link the border cycle(s): one outgoing border half-edge per boundary vertex
    border_out = {b.orig: b for b in border}
    for b in border:
        nxt = border_out[b.dest]
        b.nex = nxt
        nxt.pre = b
    if border:
        G.add_halfedges(border)

    # 4. restore crease assignments
    assignment = fold.get("edges_assignment")
    edges_vertices = fold["edges_vertices"]
    if assignment:
        for (i, j), a in zip((tuple(e) for e in edges_vertices), assignment):
            val = _LETTER_TO_ASSIGN.get(a)
            if val is None:
                continue
            he[(i, j)][CREASE_ASSIGNMENT] = val
            he[(j, i)][CREASE_ASSIGNMENT] = val

    G.check_consistency()
    return G


def save_fold(path: str, G, *, overwrite: bool = False) -> None:
    """Write *G* to a ``.fold`` JSON file (appends ``.fold`` if missing)."""
    if not path.endswith(".fold"):
        path += ".fold"
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}. Set overwrite=True to overwrite.")
    with open(path, "w") as fh:
        json.dump(graph_to_fold(G), fh)


def load_fold(path: str) -> EuclideanPositionHEG:
    """Load a ``.fold`` file into a Euclidean half-edge graph."""
    with open(path) as fh:
        return fold_to_graph(json.load(fh))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fold.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pleat/io/fold.py tests/test_fold.py
git commit -m "feat: FOLD v1.2 crease-pattern import/export (pleat.io.fold)"
```

---

### Task 3: Origami Simulator launcher — HTML template, `open_in_origami_simulator`, `origami_simulator_button`

**Files:**
- Modify: `pleat/io/fold.py` (append launcher functions)
- Modify: `pleat/io/__init__.py` (re-export the new names)
- Modify: `pleat/half.py` (one convenience method on the graph class)
- Modify: `pleat/__init__.py` (two top-level re-exports)
- Test: `tests/test_fold.py` (append)

**Interfaces:**
- Consumes: `graph_to_fold` (Task 2).
- Produces:
  - `origami_simulator_html(G, *, embed: bool = True) -> str`
  - `open_in_origami_simulator(G) -> str` (returns the temp file path; also opens the browser)
  - `origami_simulator_button(G) -> _OrigamiSimulatorButton` (has `_repr_html_`)
  - graph method `EuclideanPositionHEG.open_in_origami_simulator(self)`
  - top-level `pleat.open_in_origami_simulator`, `pleat.origami_simulator_button`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fold.py`:

```python
def test_origami_simulator_html_embeds_fold_and_importfold():
    G = _creased_rosette()
    html = __import__("pleat.io.fold", fromlist=["origami_simulator_html"]).origami_simulator_html(G)
    assert "importFold" in html
    assert "origamisimulator.org" in html
    assert '"edges_assignment"' in html  # the FOLD JSON is embedded
    assert "</script" not in html.split('"edges_assignment"')[0][-4000:] or "<\\/" in html  # no raw </script injection


def test_origami_simulator_button_repr_html():
    from pleat.io.fold import origami_simulator_button

    btn = origami_simulator_button(_creased_rosette())
    html = btn._repr_html_()
    assert "importFold" in html
    assert "<button" in html
    assert "origamisimulator.org" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fold.py -k origami -q`
Expected: FAIL with `AttributeError`/`ImportError` for `origami_simulator_html` / `origami_simulator_button`.

- [ ] **Step 3: Append the launcher to `pleat/io/fold.py`**

Add these imports at the top of the file (with the existing ones):

```python
import tempfile
import uuid
import webbrowser
```

Append at the end of `pleat/io/fold.py`:

```python
_OS_URL = "https://origamisimulator.org/"


def _fold_json(G) -> str:
    """FOLD as a JSON string safe to embed inside an HTML <script> tag."""
    return json.dumps(graph_to_fold(G)).replace("</", "<\\/")


def origami_simulator_html(G, *, embed: bool = True) -> str:
    """A self-contained HTML page that loads *G* into Origami Simulator.

    ``embed=True`` (default) embeds OS in a full-viewport iframe (click-free) with
    a "pop out to full tab" button. The handshake replies to whichever OS window
    announces itself ready (iframe or popped-out tab), so both paths import the
    same FOLD. Written to a temp file and opened by :func:`open_in_origami_simulator`.
    """
    fold_json = _fold_json(G)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>pleat &rarr; Origami Simulator</title>
<style>
  html,body{{margin:0;height:100%}}
  #os{{border:0;width:100vw;height:100vh;display:block}}
  #pop{{position:fixed;top:8px;right:8px;z-index:9;font:14px sans-serif;
       padding:6px 10px;cursor:pointer}}
</style></head><body>
<button id="pop">Open in full tab &#8599;</button>
<iframe id="os" src="{_OS_URL}"></iframe>
<script>
  const FOLD = {fold_json};
  const OS_URL = "{_OS_URL}";
  function post(win){{ win.postMessage({{op:'importFold', fold: FOLD}}, '*'); }}
  window.addEventListener('message', function(e){{
    if (e.data && e.data.from === 'OrigamiSimulator' && e.data.status === 'ready') post(e.source);
  }});
  document.getElementById('pop').addEventListener('click', function(){{
    window.open(OS_URL, 'origami_simulator');   // its 'ready' arrives via the listener
  }});
</script></body></html>"""


def open_in_origami_simulator(G) -> str:
    """Open *G* in Origami Simulator in the system browser (click-free).

    Writes a temp HTML page (see :func:`origami_simulator_html`) and opens it with
    ``webbrowser``. Returns the temp file path and prints a ``file://`` URL (a
    manual Ctrl+click fallback if auto-open fails).

    Note: with a remote kernel (SSH/cluster) the browser opens on the server; use
    :func:`save_fold` and drag the file into origamisimulator.org instead.
    """
    fd, path = tempfile.mkstemp(prefix="pleat-os-", suffix=".html")
    with os.fdopen(fd, "w") as fh:
        fh.write(origami_simulator_html(G))
    url = "file://" + path
    print(f"Opening Origami Simulator: {url}")
    webbrowser.open(url)
    return path


class _OrigamiSimulatorButton:
    """A displayable button; on click opens vanilla OS in a new tab with the CP.

    Renders in browser Jupyter and the built docs (its `_repr_html_` output is
    captured at docs-build time). Each button only answers the OS window it
    itself opened, so multiple buttons on one page don't cross-post.
    """

    def __init__(self, G):
        self._html = _origami_simulator_button_html(G)

    def _repr_html_(self) -> str:
        return self._html


def _origami_simulator_button_html(G) -> str:
    fold_json = _fold_json(G)
    uid = uuid.uuid4().hex[:8]
    return f"""<button id="pleat-os-{uid}" style="font:14px sans-serif;padding:6px 10px;cursor:pointer">
Open in Origami Simulator &#8599;</button>
<script>
(function(){{
  const FOLD = {fold_json};
  const btn = document.getElementById("pleat-os-{uid}");
  let sim = null;
  window.addEventListener('message', function(e){{
    if (e.source === sim && e.data && e.data.from === 'OrigamiSimulator' && e.data.status === 'ready')
      e.source.postMessage({{op:'importFold', fold: FOLD}}, '*');
  }});
  btn.addEventListener('click', function(){{ sim = window.open("{_OS_URL}", 'origami_simulator'); }});
}})();
</script>"""


def origami_simulator_button(G) -> _OrigamiSimulatorButton:
    """Return a displayable "Open in Origami Simulator" button for *G*.

    Use in a notebook cell (browser Jupyter or the online docs). On click it opens
    Origami Simulator in a new tab and imports the crease pattern.
    """
    return _OrigamiSimulatorButton(G)
```

- [ ] **Step 4: Re-export from `pleat/io/__init__.py`**

Add after the `from .heg import ...` line:

```python
from .fold import (
    fold_to_graph,
    graph_to_fold,
    load_fold,
    open_in_origami_simulator,
    origami_simulator_button,
    origami_simulator_html,
    save_fold,
)
```

And extend `__all__` with:

```python
    "graph_to_fold",
    "fold_to_graph",
    "save_fold",
    "load_fold",
    "origami_simulator_html",
    "open_in_origami_simulator",
    "origami_simulator_button",
```

- [ ] **Step 5: Add the convenience method to the graph class**

In `pleat/half.py`, find `class EuclideanPositionHEG` and add this method (it defers the import to avoid a cycle, mirroring how `io` imports from `half`):

```python
    def open_in_origami_simulator(self) -> str:
        """Open this crease pattern in Origami Simulator (see pleat.io.fold)."""
        from .io.fold import open_in_origami_simulator

        return open_in_origami_simulator(self)
```

- [ ] **Step 6: Add top-level re-exports in `pleat/__init__.py`**

After `import pleat.io`, add:

```python
from pleat.io.fold import open_in_origami_simulator, origami_simulator_button
```

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_fold.py -q && python -c "import pleat; print(pleat.open_in_origami_simulator, pleat.origami_simulator_button)"`
Expected: PASS (5 tests), then the two function reprs print.

- [ ] **Step 8: Commit**

```bash
git add pleat/io/fold.py pleat/io/__init__.py pleat/half.py pleat/__init__.py tests/test_fold.py
git commit -m "feat: open crease patterns in Origami Simulator (iframe launcher + inline button)"
```

---

### Task 4: Documentation

**Files:**
- Modify: `docs/notebooks/Saving_and_Exporting.ipynb`

**Interfaces:**
- Consumes: everything from Tasks 2–3.

- [ ] **Step 1: Read the notebook's existing structure**

Run: `python -c "import json; nb=json.load(open('docs/notebooks/Saving_and_Exporting.ipynb')); print(len(nb['cells'])); [print(i, c['cell_type'], ''.join(c['source'])[:70].replace(chr(10),' ')) for i,c in enumerate(nb['cells'])]"`
Expected: prints the cell list so you can see how a CP (`cp`) is built earlier in the notebook and match its variable name / style.

- [ ] **Step 2: Add a markdown cell and a code cell**

Add near the end (use `NotebookEdit`, or edit the JSON). Markdown cell:

```markdown
## FOLD & Origami Simulator

[FOLD](https://github.com/edemaine/fold) is the standard origami interchange
format. `save_fold` writes a `.fold` file (crease pattern with M/V/B assignments
and fold angles); `load_fold` reads one back. You can also open a crease pattern
straight in [Origami Simulator](https://origamisimulator.org/):

- `cp.open_in_origami_simulator()` — opens it in your browser (works from a
  notebook or a script; needs a local kernel — with a remote kernel, `save_fold`
  and drag the file in instead).
- `origami_simulator_button(cp)` — renders a button (works here in the online
  docs): click it to open the pattern in Origami Simulator.
```

Code cell (match the CP variable used earlier in the notebook — replace `cp` if it differs):

```python
from pleat.io.fold import save_fold, origami_simulator_button

save_fold("example", cp, overwrite=True)      # writes example.fold
origami_simulator_button(cp)                    # a clickable button in the docs
```

- [ ] **Step 3: Verify the notebook executes (this is what the docs build does)**

Run: `jupyter nbconvert --to notebook --execute --stdout docs/notebooks/Saving_and_Exporting.ipynb > /dev/null && echo OK`
Expected: `OK` (no execution error — `execute: true` in `mkdocs.yml` runs this at build).

- [ ] **Step 4: Commit**

```bash
git add docs/notebooks/Saving_and_Exporting.ipynb
git commit -m "docs: FOLD export and Open-in-Origami-Simulator section"
```

---

## Self-Review

**Spec coverage:**
- FOLD v1.2 export mapping (vertices/edges/assignment/foldAngle/faces) → Task 2 `graph_to_fold`. ✓
- FOLD import / DCEL reconstruction → Task 2 `fold_to_graph`. ✓
- `.fold` files → Task 2 `save_fold`/`load_fold`. ✓
- `io` subpackage split → Task 1. ✓
- Kernel-side `webbrowser` + temp-HTML iframe + pop-out + printed `file://` path → Task 3 `origami_simulator_html`/`open_in_origami_simulator`. ✓
- Online-docs / inline button surface → Task 3 `origami_simulator_button` + Task 4. ✓
- Convenience method + re-exports → Task 3 Steps 5–6. ✓
- Tests: round-trip, FOLD validity, HTML content → Tasks 2–3. ✓
- `.heg` retained (not retired); Euclidean-only scope → enforced by keeping Task 1 a pure move and FOLD living alongside. ✓
- Deferred (foldedForm/faceOrders, importSVG, results.show() wiring) → not in any task, as intended. ✓

**Placeholder scan:** No TBD/TODO; every code and command step is concrete.

**Type consistency:** `graph_to_fold`/`fold_to_graph`/`save_fold`/`load_fold`/`origami_simulator_html`/`open_in_origami_simulator`/`origami_simulator_button` names match across the module, `__init__` re-exports, tests, and the graph method. Assignment constants (`MOUNTAIN`/`VALLEY`) and `CREASE_ASSIGNMENT` used consistently. `_fold_json` shared by both the iframe page and the button.
