"""FOLD (v1.2) crease-pattern I/O and an Origami Simulator launcher.

FOLD spec: https://github.com/edemaine/fold/blob/main/doc/spec.md
Scope: Euclidean 2D crease patterns. See docs/superpowers/specs for the design.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
import webbrowser

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

    faces_vertices = [[vidx[v] for v in f.vertex_iter()] for f in sorted(G.faces, key=lambda f: f["id"])]

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
        for e, a in zip(edges_vertices, assignment):
            val = _LETTER_TO_ASSIGN.get(a)
            if val is None:
                continue
            i, j = int(e[0]), int(e[1])
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


# ===========================================================================
# Origami Simulator launcher
# ===========================================================================
#
# Origami Simulator (https://origamisimulator.org/) imports a crease pattern via
# a postMessage handshake: it announces ``{from:'OrigamiSimulator',
# status:'ready'}`` to its parent frame (when embedded) or opener (when popped
# out), and then accepts ``{op:'importFold', fold:<FOLD object>}``. We embed the
# FOLD as JSON in a self-contained page and reply to whichever OS window reports
# ready, so both the iframe and the popped-out tab import the same pattern.

# The empty ``?model=`` query is load-bearing: it makes Origami Simulator skip
# loading its default demo (the waterbomb), which would otherwise finish loading
# *after* our importFold and clobber it. This mirrors erikdemaine.org's maze tool.
_OS_URL = "https://origamisimulator.org/?model="


def _fold_json(G) -> str:
    """FOLD as a JSON string safe to embed inside an HTML <script> tag."""
    return json.dumps(graph_to_fold(G)).replace("</", "<\\/")


def origami_simulator_html(G) -> str:
    """A self-contained HTML page that loads *G* into Origami Simulator.

    OS is embedded in a full-viewport iframe (click-free) with a "pop out to full
    tab" button. The handshake replies to whichever OS window announces itself
    ready (iframe or popped-out tab), so both paths import the same FOLD. Written
    to a temp file and opened by :func:`open_in_origami_simulator`.
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

    Renders in browser Jupyter and the built docs (its ``_repr_html_`` output is
    captured at docs-build time). Each button only answers the OS window it
    itself opened, so multiple buttons on one page do not cross-post.
    """

    def __init__(self, G) -> None:
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
