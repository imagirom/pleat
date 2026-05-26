"""Reciprocal figure construction.

The reciprocal figure of a planar graph ``G`` places a point on each face
such that the line connecting the points of two adjacent faces is
perpendicular to the shared primal edge. It is the geometric prerequisite
for building a flat-foldable shrink-rotate crease pattern.

The implementation uses a least-squares solve constrained to the null
space induced by Kirchhoff-style edge cycle conditions, then optimises the
remaining degrees of freedom so the dual points approximate the primal face
centroids.
"""

from __future__ import annotations

import logging
from copy import copy


import numpy as np
from numpy.typing import NDArray
import scipy as sc

from ..base import rotation_matrix
from ..conway import dual_graph
from ..half import Face, GeometricHEG, HalfEdge, HalfEdgeGraph, Vertex
from ..utils import invert_mapping, random_directed_set

logger = logging.getLogger(__name__)


def reciprocal_figure(
    G: GeometricHEG,
    reciprocal_pos_key: str = "reciprocal_pos",
    rcond: float = 1e-7,
) -> GeometricHEG:
    """Compute the reciprocal figure of *G* and return it as a face graph.

    Stores reciprocal positions on faces of *G* under
    *reciprocal_pos_key*. Returns the dual graph (one vertex per face of
    *G*, one face per vertex of *G*) with positioned vertices, mapped
    back to *G* via ``'pre'`` attributes on vertices, halfedges, and
    faces.
    """
    # Step 1: Choose direction for every interior edge.
    directed_edges = random_directed_set([e for e in G.halfedges if not (e.on_border() or e.rev.on_border())])

    # Step 2: Construct array of all vectors of the directed edges.
    edge_vectors = np.stack([e.orig["pos"] - e.dest["pos"] for e in directed_edges])

    dual_vectors = edge_vectors @ rotation_matrix(np.pi / 2)
    dual_directions = dual_vectors / np.linalg.norm(dual_vectors, axis=1, keepdims=True)
    edges_to_ids = {e: i for i, e in enumerate(directed_edges)}

    # Step 3: Formulate constraints as linear problem A x = 0.
    # Every interior vertex contributes one row.
    interior_vertices = [v for v in G.vertices if not v.on_border()]

    rows = []
    n_edges = len(directed_edges)
    for v in interior_vertices:
        row = np.zeros(n_edges, dtype=np.float32)
        for e in v.outgoing_iter():
            if e in directed_edges:
                row[edges_to_ids[e]] = -1
            else:
                row[edges_to_ids[e.rev]] = 1
        rows.append(row)
    B = np.stack(rows)
    A = (B[:, None, :] * dual_directions.T[:None]).reshape(-1, n_edges)
    U = sc.linalg.null_space(A, rcond=rcond)
    assert U.shape[1] > 0, "G does not have a reciprocal figure!"

    # Step 4: Least-squares fit of the remaining degrees of freedom so dual
    # vertices approximate primal face centroids.
    to_process = set(G.faces)
    anchor = to_process.pop()
    coefficients: dict[Face, NDArray[np.float32]] = {anchor: np.zeros(n_edges, dtype=np.float32)}
    border: set[Face] = {anchor}
    while border:
        new_border: set[Face] = set()
        for f in border:
            for e in f.halfedge_iter():
                f2 = e.rev.face
                if f2 is None:
                    continue
                if f2 not in coefficients:
                    if e in directed_edges:
                        coefficients[f2] = copy(coefficients[f])
                        coefficients[f2][edges_to_ids[e]] = -1
                    elif e.rev in directed_edges:
                        coefficients[f2] = copy(coefficients[f])
                        coefficients[f2][edges_to_ids[e.rev]] = 1
                    else:
                        continue
                    new_border.add(f2)
        border = new_border
    assert set(coefficients.keys()) == set(G.faces)

    faces = G.faces
    n_faces = len(faces)
    D2P = np.stack([coefficients[f] for f in faces])

    M = np.moveaxis(np.dot(D2P, np.moveaxis(U[:, :, None] * dual_directions[:, None], 0, 1)), 1, 2)

    # Two extra columns parametrise a global xy offset of the dual graph.
    xy_columns = np.zeros((n_faces, 2, 2), dtype=np.float32)
    xy_columns[:, 0, 0] = 1
    xy_columns[:, 1, 1] = 1
    M = np.concatenate([xy_columns, M], axis=-1)

    face_centers = np.stack([f.midpoint() for f in faces])

    M = M.reshape(n_faces * 2, -1)
    face_centers = face_centers.reshape(n_faces * 2)

    logger.info("optimizing rotation centers using %d degrees of freedom..", M.shape[-1])
    sol = sc.optimize.lsq_linear(M, face_centers, lsq_solver="exact")
    assert sol["success"], f"{sol['message']}"
    sol = sol["x"]

    dual_vertices = M @ sol
    dual_vertices = dual_vertices.reshape(-1, 2)

    if reciprocal_pos_key is not None:
        for i, f in enumerate(faces):
            f[reciprocal_pos_key] = dual_vertices[i]

    # Step 5: Make the reciprocal figure into a face graph.
    D, (v_map, e_map, f_map) = G.copy(return_mappings=True)
    face2reciprocalpos = {f_map[f]: dual_vertices[i] for i, f in enumerate(faces)}

    D = dual_graph()(D)
    for v in D.vertices:
        v["pos"] = face2reciprocalpos[v["pre_conway"]]

    inv_v_map = invert_mapping(f_map)
    for v in D.vertices:
        v["pre"] = inv_v_map[v["pre_conway"]]
    inv_e_map = invert_mapping(e_map)
    for e in D.halfedges:
        if "pre_conway" in e.attributes:
            e["pre"] = inv_e_map[e["pre_conway"]]
    inv_f_map = invert_mapping(v_map)
    for f in D.faces:
        f["pre"] = inv_f_map[f["pre_conway"]]

    return D
