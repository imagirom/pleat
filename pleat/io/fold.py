"""FOLD (v1.2) crease-pattern import/export.

FOLD spec: https://github.com/edemaine/fold/blob/main/doc/spec.md
Scope: Euclidean 2D crease patterns. See docs/superpowers/specs for the design.
Opening a pattern in Origami Simulator lives in :mod:`pleat.origami_simulator`.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..half import EuclideanPositionHEG, Face, HalfEdge, Vertex
from ..overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY, color_creases

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

    Raises:
        ValueError: if *G* does not have real 2D vertex positions -- FOLD cannot
            represent the hyperbolic (complex Poincaré-disk) or spherical (3D)
            coordinates pleat uses for curved tilings.
    """
    verts = sorted(G.vertices, key=lambda v: v["id"])
    if verts:
        sample = np.asarray(verts[0]["pos"])
        if np.iscomplexobj(sample):
            raise ValueError(
                "FOLD export requires a Euclidean 2D crease pattern, but this graph has "
                "complex (hyperbolic / Poincaré-disk) vertex positions."
            )
        if sample.ravel().size != 2:
            raise ValueError(
                f"FOLD export requires 2D vertex positions, but this graph has "
                f"{sample.ravel().size}D positions (e.g. a spherical tiling)."
            )
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

    # Emit faces clockwise (pleat's vertex_iter is CCW, hence the reversal). Origami
    # Simulator's importFold colors the face's CCW side as the *back* (white) and shows
    # it up; feeding CW puts the front (coloured) side up, matching pleat's rendering.
    # This only affects which side is coloured, not the fold -- mountain/valley is
    # driven by edges_foldAngle, independent of winding. fold_to_graph reverses back.
    faces_vertices = [[vidx[v] for v in f.vertex_iter()][::-1] for f in sorted(G.faces, key=lambda f: f["id"])]

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

    # 1. interior half-edges from each face loop. graph_to_fold emits faces clockwise
    #    (for Origami Simulator); reverse back to pleat's CCW convention here.
    he: dict[tuple[int, int], HalfEdge] = {}
    for raw_face in faces_vertices:
        face_vs = raw_face[::-1]
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

    color_creases(G)  # set edge["color_key"] from CREASE_ASSIGNMENT
    G.check_consistency()
    return G


def save_fold(path: str, G, *, overwrite: bool = False) -> None:
    """Write *G* to a ``.fold`` JSON file (appends ``.fold`` if missing)."""
    if not path.endswith(".fold"):
        path += ".fold"
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}. Set overwrite=True to overwrite.")
    fold = graph_to_fold(G)  # build first, so a failure leaves no partial file
    with open(path, "w") as fh:
        json.dump(fold, fh)


def load_fold(path: str) -> EuclideanPositionHEG:
    """Load a ``.fold`` file into a Euclidean half-edge graph."""
    with open(path) as fh:
        return fold_to_graph(json.load(fh))
