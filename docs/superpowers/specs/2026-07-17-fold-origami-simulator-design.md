# FOLD export/import + "Open in Origami Simulator"

**Date:** 2026-07-17
**Status:** Design approved, pending spec review. Implementation to happen in a dedicated worktree.

## Goal

Let pleat crease patterns be opened in [Origami Simulator](https://origamisimulator.org/)
directly from a Jupyter notebook or a plain Python script, and add FOLD as a
first-class crease-pattern interchange format (import + export).

## Background: how the reference button works

The "Simulate in Origami Simulator" button on
[erikdemaine.org/fonts/maze](https://erikdemaine.org/fonts/maze/) uses a
**browser-to-browser `postMessage` handshake** — no server, no file hosting
(`maze.js`):

1. Opener opens Origami Simulator (OS): `window.open('https://origamisimulator.org/')`.
2. OS, once loaded, posts back `{from:'OrigamiSimulator', status:'ready'}` — to
   `window.parent` if it is embedded in an iframe, else to `window.opener`.
3. Opener waits for `ready`, then posts the crease pattern.

Origami Simulator's own `js/importer.js` listens for exactly two ops:

- `{op:'importFold', fold:<FOLD object>}` — sets FOLD data directly.
- `{op:'importSVG', svg:<string>, filename, vertTol}` — parses SVG by stroke colour.

The maze uses `importSVG` only because its renderer emits SVG. We use
**`importFold`**: pleat has the full half-edge topology + M/V assignments, so it
can hand OS an unambiguous FOLD object — no colour round-trip (which even pleat's
own `svg.load_svg` has to reverse heuristically).

Note: `?model=` in the OS URL only selects *built-in demos* (it reads the
`data-url` of an `<a class="demo">`), **not** arbitrary URLs. So the only
automatic import path is `postMessage` from a parent/opener page. OS sets no
`X-Frame-Options`/CSP `frame-ancestors` (verified), so it can be embedded in an
iframe.

## Launch mechanism (must work in browser Jupyter, JupyterLab, VS Code, and scripts)

Rendering *inside* notebook output is not portable: VS Code's notebook webview
sandboxes HTML/JS and blocks external iframes and `Javascript` display, and the
three front-ends have different CSP. The one mechanism that works identically
everywhere is **kernel-side**:

> `graph_to_fold(cp)` → write a self-contained temp `.html` that embeds OS in a
> full-page iframe plus a handshake script carrying the FOLD JSON →
> `webbrowser.open("file://…")`.

`open_in_origami_simulator(cp)` step by step:

1. Build the FOLD dict from the graph.
2. Write a temp HTML file containing a full-page `<iframe
   src="https://origamisimulator.org/">` and an inline `<script>` with the FOLD
   embedded as JSON plus a `message` listener.
3. `webbrowser.open("file:///…/pleat-os-XXXX.html")`.
4. The default browser loads the local page; the iframe loads OS.
5. OS finishes, sees it is framed (`window.parent !== window`), posts
   `{from:'OrigamiSimulator', status:'ready'}` to the parent (our page).
6. Our listener receives `ready`, posts `{op:'importFold', fold}` into the iframe.
7. OS imports the FOLD and folds the CP.

**Cross-platform:** the only OS-specific step is #3, and it uses Python's stdlib
`webbrowser.open` (**not** a direct `xdg-open` call) — portable, dispatching to
`xdg-open`/`$BROWSER` on Linux, the `open` command on macOS, and `os.startfile`
on Windows. Steps 4–7 are pure browser behaviour, identical on every OS. A
`file://` page posting to an `https://` iframe is fine: cross-origin
`postMessage` with `'*'` is allowed, and `file→https` is not mixed content.

The kernel is local in the target environments, so `webbrowser.open` launches the
user's **real default browser**, sidestepping every notebook-webview CSP. The
temp page is an ordinary top-level page, so the iframe loads with no popup blocker
and no user gesture. Click-free on execute.

This is **one code path** — identical for the notebook call and the programmatic
call. No `get_ipython` branching, no ipywidgets, no popup.

`open_in_origami_simulator` also **prints the `file://` path** (a manual
Ctrl+click fallback if auto-open fails).

**Vanilla OS vs embedded.** The maze opens *vanilla* OS in a new tab because it is
a live https page reacting to a user *click* (the gesture lets `window.open`
succeed, and the page stays alive to run the handshake). A Python-launched,
click-free flow cannot open a *separate tab* (auto `window.open` is popup-blocked)
and has no other data channel (`?model=` loads built-in demos only; cross-origin
`localStorage` is inaccessible). The iframe therefore loads the **real, unmodified
origamisimulator.org** — just framed; at full viewport it is visually
indistinguishable from vanilla OS. The temp page additionally renders a small
**"pop out to full tab ↗"** button that, on click (gesture now available), opens
vanilla OS in a new tab and re-sends the FOLD — covering the few OS features that
dislike being framed (fullscreen, VR, native save dialogs).

**Known limitation:** if the kernel is remote (SSH/cluster), `webbrowser` opens a
browser on the server. Documented; `save_fold` + manual drag-in is the fallback.

## Online-documentation surface

The docs are built with `mkdocs-jupyter` (`execute: true`) and served over https
from GitHub Pages. Verified against the built `site/`: notebooks are executed at
build time so their display output is captured, inline `<script>` tags and Jupyter
widget scripts are preserved in the static HTML, and there is no CSP. This is the
maze's native scenario — a live https page plus a real user click.

So a notebook cell that emits an **"Open in Origami Simulator" button** (an
`IPython.display.HTML` output) renders a *working* button in the online docs: a
reader clicks it and **vanilla OS opens in a new tab** with the CP loaded (the
click is a genuine gesture, so `window.open` is not blocked). For the docs a
button is preferred over an auto-loading iframe, so a page of results does not
spin up a WebGL OS instance per result at page load.

This is a second delivery surface over the same FOLD payload + handshake core:

- `open_in_origami_simulator(cp)` — temp file + `webbrowser` (live kernels: VS
  Code / Lab / browser / scripts; click-free embedded iframe + pop-out). Headline
  "open it now" call.
- `origami_simulator_button(cp)` — returns a displayable (button + handshake +
  embedded FOLD) for **browser Jupyter and the online docs**; opens vanilla OS in
  a new tab on click. Does not run in VS Code's live webview (that is covered by
  the function above; and the docs are static, so it is moot there).

## FOLD ↔ pleat mapping (crease pattern only)

Target: FOLD **v1.2** (current spec), `frame_classes:["creasePattern"]`.

pleat crease model (from `pleat.overlap`): `CREASE_ASSIGNMENT="crease_assignment"`
attribute on half-edges, `MOUNTAIN=1 / VALLEY=-1 / unassigned=0`, mirrored on
`.rev`. Border = half-edge with no incident face.

**Export `graph_to_fold(G) -> dict`:**

| FOLD key | Source |
|---|---|
| `file_spec` | `1.2` |
| `file_creator` | `"pleat"` |
| `file_classes` | `["singleModel"]` |
| `frame_classes` | `["creasePattern"]` |
| `frame_attributes` | `["2D"]` |
| `vertices_coords` | `v["pos"]` → `[float(x), float(y)]` |
| `edges_vertices` | rev-paired half-edges, deduped → `[u, v]` vertex indices |
| `edges_assignment` | `1→"M"`, `-1→"V"`, `0`/missing→`"U"`, no-face→`"B"` |
| `edges_foldAngle` | `"M"→-180`, `"V"→180`, else `null` (FOLD convention: M negative) |
| `faces_vertices` | each face's CCW vertex loop; outer/deleted face skipped |

Vertices, edges, faces are assigned integer indices in a stable order.

**Import `fold_to_graph(fold) -> EuclideanPositionHEG`:** rebuild the DCEL from
`faces_vertices`, reusing the adjacency→half-edge construction pattern already in
`io._build_heg_from_data` (create half-edges per directed face side, pair by
`rev`, link `nex`/`pre`, assign faces, drop the outer face). Restore `pos` from
`vertices_coords` and `CREASE_ASSIGNMENT` from `edges_assignment` (M→1, V→-1,
B/U/F→unset). If the FOLD frame has no `faces_vertices` (a `graph`/`linkage`
class), raise a clear error — reconstructing a face-based DCEL needs faces.

Scope: Euclidean 2D crease patterns. Non-Euclidean geometries stay on `.heg`
(see below).

## `.heg` decision

FOLD is **not** a complete replacement for `.heg` and this feature does not retire
it. `.heg` is pleat's generic half-edge (DCEL) dump backing the 199-file `graphs/`
corpus, which includes hyperbolic (Poincaré, complex-valued positions) and other
non-Euclidean tilings stored pre-creasing. FOLD's `vertices_coords` carry no
geometry-model marker, so those would degrade to meaningless flat coords; and
FOLD reconstruction wants a clean oriented 2-complex, which faceless/mid-construction
graphs lack. A `pleat:*`-extended FOLD could subsume `.heg` later, but that is a
separate, riskier migration and is out of scope here. FOLD is added *alongside*
`.heg`, covering the Euclidean crease-pattern subset.

## API

FOLD serialization is I/O and belongs with `.heg`/CirclePack. `io.py` (604 lines,
two formats) is at the size where a subpackage pays off, so **promote `pleat/io.py`
to a subpackage** `pleat/io/`:

```
pleat/io/
  __init__.py     — re-exports the public names (see below)
  heg.py          — existing .heg YAML: graph_to_dict, dict_to_graph,
                    save_graph, load_graph
  circlepack.py   — existing CirclePack .p: CirclePackData, parse_p_file,
                    write_p_file, load_circlepack, save_circlepack, and the
                    HEG<->CirclePackData bridge helpers
  fold.py         — new: FOLD serialization + Origami Simulator launcher
```

The split of the existing `io.py` into `heg.py` + `circlepack.py` is mechanical
(no logic changes). `io/__init__.py` re-exports every currently-public name, so
existing imports (`pleat.io.load_graph`, `pleat.io.save_graph`,
`pleat.io.load_circlepack`, `from pleat.io import graph_to_dict, ...` in tests,
`pleat.io.load_graph("graphs/irregular2.heg")` in
`tests/test_intersecting_cylinders.py`) keep working unchanged. This is the
package's public surface, not a back-compat shim.

`pleat/io/fold.py` (module-function style, matching `heg.py`/`circlepack.py`; note
`G.save` already means render-to-image, so FOLD I/O is module functions, not graph
methods):

- `graph_to_fold(G) -> dict`
- `fold_to_graph(fold: dict) -> EuclideanPositionHEG`
- `save_fold(path, G, *, overwrite=False)` — appends `.fold`, writes JSON
- `load_fold(path) -> EuclideanPositionHEG`
- `origami_simulator_html(G) -> str` — pure function: the self-contained page
  (full-viewport iframe + pop-out button + handshake JS + embedded FOLD JSON).
  Testable without a browser.
- `open_in_origami_simulator(G)` — temp file + `webbrowser.open`; prints the
  `file://` path. Headline entry point; usable both in a notebook cell and a
  script. (The browser launcher is FOLD-coupled and small, so it is co-located in
  `fold.py` rather than a separate module.)
- `origami_simulator_button(G)` — returns a small object with `_repr_html_`
  (button + handshake + embedded FOLD) for inline display in browser Jupyter and
  the online docs; opens vanilla OS in a new tab on click. Shares the handshake
  core with `origami_simulator_html`.

One convenience method on the graph class: `G.open_in_origami_simulator()`
(delegates to the module function) — this is the "after results are shown" UX and
reads better than the fully-qualified call. `to_fold`/`from_fold` stay module
functions to avoid clashing with the image-oriented `G.save`.

Re-export the FOLD public names from both `pleat/io/__init__.py` and
`pleat/__init__.py` (so `pleat.open_in_origami_simulator(cp)` works).

### Handshake HTML template

Full-viewport `<iframe src="https://origamisimulator.org/">` plus a script that:

- holds the FOLD object as embedded JSON,
- listens for `message` where `e.data.from === 'OrigamiSimulator' && e.data.status === 'ready'`,
- on ready, `iframe.contentWindow.postMessage({op:'importFold', fold}, '*')`,
- guards the race (post once ready; if the iframe was already ready, a short
  retry/interval covers it — same shape as the maze's `ready`/`onReady` flag).

## Testing

One meaningful automated check (no browser needed):

- **Round-trip:** build a small CP with M/V/B assignments →
  `graph_to_fold → fold_to_graph` preserves vertex/edge/face counts and the
  per-edge M/V/B assignment.
- **FOLD validity:** `file_spec == 1.2`; `edges_assignment ⊆ {M,V,B,F,U}`;
  every `edges_foldAngle` in `[-180, 180]` or null; `edges_vertices` indices in
  range.
- **HTML:** `origami_simulator_html(G)` and `origami_simulator_button(G)`'s
  `_repr_html_` each contain the FOLD JSON and the string `importFold`.

## Docs

Add a "FOLD & Origami Simulator" section to
`docs/notebooks/Saving_and_Exporting.ipynb`: `save_fold`,
`open_in_origami_simulator(cp)` / `cp.open_in_origami_simulator()` (with the
remote-kernel caveat), and an `origami_simulator_button(cp)` cell — the latter
renders a working button in the built online docs, so readers can open a CP in
Origami Simulator straight from the documentation.

## Deferred (not in this change)

- `foldedForm` frame + `faceOrders` from the ILP layer-ordering solve.
- `.heg` retirement / `pleat:*`-extended FOLD.
- Auto-wiring a button into `results.show()` (the inline `origami_simulator_button`
  is in scope; wiring it into the results panels automatically is not).
- `importSVG` path (superseded by `importFold`; pleat's colored SVG export
  already exists if ever needed as a fallback).
