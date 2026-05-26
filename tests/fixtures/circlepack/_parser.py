"""Minimal parser for CirclePack .p files (FLOWERS / RADII / CENTERS).

Adapted from the notebooks-private/CirclePack.ipynb prototype.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eucare.half import EuclideanPositionHEG, Face, HalfEdge, Vertex, rotate_by


@dataclass
class CirclePackData:
    nodecount: int
    geometry: str | None
    alpha: int | None  # 1-indexed
    beta: int | None
    gamma: int | None
    flowers: dict[int, list[int]]  # 0-indexed
    radii: np.ndarray | None  # shape (N,)
    centers: np.ndarray | None  # shape (N, 2)


def parse_p_file(path: str) -> CirclePackData:
    """Parse a CirclePack .p file into a structured dataclass."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    assert lines[-1] == "END", lines[-1]
    lines = lines[:-1]

    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    name = ""
    for line in lines:
        if ":" in line:
            if current is not None:
                sections[name] = current
            head, tail = line.split(":", 1)
            name = head.strip()
            current = []
            tail = tail.strip()
            if tail:
                current.append(tail)
        else:
            if current is None:
                raise ValueError(f"Unexpected line without section: {line}")
            current.append(line)
    if current is not None:
        sections[name] = current

    nodecount = int(sections["NODECOUNT"][0])
    geometry = sections.get("GEOMETRY", [None])[0]

    abg = sections.get("ALPHA/BETA/GAMMA")
    alpha = beta = gamma = None
    if abg:
        parts = abg[0].split()
        alpha = int(parts[0])
        beta = int(parts[1])
        gamma = int(parts[2])

    flowers: dict[int, list[int]] = {}
    if "FLOWERS" in sections:
        # Each line: "center degree   n1 n2 ... nk"
        for line in sections["FLOWERS"]:
            parts = line.split()
            center = int(parts[0]) - 1
            # parts[1] is the degree, then comes a list of neighbor indices
            degree = int(parts[1])
            neighbors = [int(x) - 1 for x in parts[2:]]
            assert len(neighbors) == degree + 1, f"{line}, expected {degree+1} got {len(neighbors)}"
            flowers[center] = neighbors

    radii: np.ndarray | None = None
    if "RADII" in sections:
        nums = []
        for line in sections["RADII"]:
            nums.extend(float(x) for x in line.split())
        radii = np.array(nums)
        assert len(radii) == nodecount, f"{len(radii)} != {nodecount}"

    centers: np.ndarray | None = None
    if "CENTERS" in sections:
        nums: list[float] = []
        for line in sections["CENTERS"]:
            nums.extend(float(x) for x in line.split())
        centers = np.array(nums).reshape(-1, 2)
        assert len(centers) == nodecount, f"{len(centers)} != {nodecount}"

    return CirclePackData(
        nodecount=nodecount,
        geometry=geometry,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        flowers=flowers,
        radii=radii,
        centers=centers,
    )


def build_heg_from_flowers(data: CirclePackData) -> tuple[EuclideanPositionHEG, dict[int, Vertex]]:
    """Build an EuclideanPositionHEG with the combinatorial structure from FLOWERS.

    Returns (graph, idx_to_vertex). Positions are stubbed (all zeros).
    """
    G = EuclideanPositionHEG()
    idx_to_v: dict[int, Vertex] = {}
    for i in range(data.nodecount):
        v = Vertex()
        v["pos"] = np.array([0.0, 0.0])
        idx_to_v[i] = v
    G.add_vertices(list(idx_to_v.values()))

    # Build half-edges from flowers.
    # flowers[i] is a list of neighbor indices in CCW order. For interior vertices
    # the list is closed (last == first); for boundary vertices it is open.
    h_lookup: dict[Vertex, dict[Vertex, HalfEdge]] = {}
    for i, neighbors in data.flowers.items():
        v = idx_to_v[i]
        h_lookup[v] = {}
        # Strip trailing duplicate for interior (closed) flowers
        unique = neighbors[:-1] if (neighbors and neighbors[0] == neighbors[-1]) else neighbors
        for j in unique:
            w = idx_to_v[j]
            h = HalfEdge(orig=v, dest=w)
            h_lookup[v][w] = h
            v.any_outgoing = h
        G.add_halfedges(list(h_lookup[v].values()))

    # rev links
    for v in h_lookup:
        for w, h in h_lookup[v].items():
            h.rev = h_lookup[w][v]

    # nex / pre links: walk the flower in given order. For each pair (h_curr, h_next)
    # of consecutive outgoing edges at v, h_curr.rev.nex == h_next.
    for i, neighbors in data.flowers.items():
        v = idx_to_v[i]
        outgoing = [h_lookup[v][idx_to_v[j]] for j in (neighbors[:-1] if neighbors[0] == neighbors[-1] else neighbors)]
        for h_revnex, h, h_prerev in rotate_by(outgoing, (0, 1, 2)):
            h.rev.nex = h_revnex
            h.pre = h_prerev.rev

    # Build faces. For each half-edge not yet assigned a face, walk h.nex around
    # until we return; that's a face.
    from copy import copy

    unassigned = copy(G.halfedges)
    while unassigned:
        h_start = next(iter(unassigned))
        f = Face(any_side=h_start)
        G.add_face(f)
        for k in f.halfedge_iter():
            k.face = f
            unassigned.discard(k)

    G.check_consistency()

    # Identify the outer face combinatorially: any boundary vertex v in the flower
    # data (flowers[v][0] != flowers[v][-1]) has its half-edge to the last flower
    # neighbor with reverse-face equal to the outer face.
    outer_face: Face | None = None
    for i, neighbors in data.flowers.items():
        if not neighbors:
            continue
        if neighbors[0] != neighbors[-1]:
            v = idx_to_v[i]
            first_w = idx_to_v[neighbors[0]]
            h = h_lookup[v][first_w]
            outer_face = h.rev.face
            break
    if outer_face is None:
        # Closed surface (no boundary). Pick the largest face as a fallback (sphere case).
        outer_face = max(G.faces, key=lambda f: f.order())
    G.delete_face(outer_face)
    return G, idx_to_v
