"""File I/O for the `.heg` half-edge graph format (YAML-based) and the
CirclePack `.p` format."""

from __future__ import annotations

import os
from copy import copy
from dataclasses import dataclass

import numpy as np
import yaml

import pleat as ec

from .geometries import EuclideanGeometry, PoincareDiskModel
from .half import EuclideanPositionHEG, Face, HalfEdge, HalfEdgeGraph, Vertex, rotate_by


def graph_to_dict(
    G: HalfEdgeGraph,
    attributes_to_save: tuple[str, ...] = ("pos", "length", "in_angle", "color_key"),
) -> dict:
    """Serialise a half-edge graph to a JSON/YAML-friendly nested dict.

    Vertices, half-edges, and faces are each given an opaque string label
    (``v0``, ``h0``, ``f0``, ...).  Cross-references are stored by label.
    Numpy arrays in attributes are converted to plain Python lists, and numpy
    scalars to ``float``.

    Args:
        G: The graph to serialise.
        attributes_to_save: Attribute keys to copy onto each element.  Other
            attributes are dropped.

    Returns:
        ``{'vertices': ..., 'halfedges': ..., 'faces': ...}``, suitable for
        :func:`yaml.dump`.
    """
    vertex_labels = {v: f"v{i}" for i, v in enumerate(G.vertices)}
    halfedge_labels = {h: f"h{i}" for i, h in enumerate(G.halfedges)}
    face_labels = {f: f"f{i}" for i, f in enumerate(G.faces)}

    labels = {None: None}
    labels.update(vertex_labels)
    labels.update(halfedge_labels)
    labels.update(face_labels)

    def represent_attributes(obj):
        result = {}
        for attr in attributes_to_save:
            if attr in obj.attributes:
                value = obj[attr]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if np.isscalar(value):
                    value = float(value)
                result[attr] = value
        return result

    def add_attributes(func):
        def wrapped(obj):
            result = func(obj)
            attrs = represent_attributes(obj)
            if attrs:
                result["attributes"] = attrs
            return result

        return wrapped

    @add_attributes
    def represent_vertex(v):
        return dict(any_outgoing=labels[v.any_outgoing])

    @add_attributes
    def represent_halfedge(h):
        return dict(
            orig=labels[h.orig],
            dest=labels[h.dest],
            rev=labels[h.rev],
            nex=labels[h.nex],
            pre=labels[h.pre],
            face=labels[h.face],
        )

    @add_attributes
    def represent_face(f):
        return dict(any_side=labels[f.any_side])

    vertex_dict = {label: represent_vertex(v) for v, label in vertex_labels.items()}
    halfedge_dict = {label: represent_halfedge(h) for h, label in halfedge_labels.items()}
    face_dict = {label: represent_face(f) for f, label in face_labels.items()}

    graph_dict = dict(vertices=vertex_dict, halfedges=halfedge_dict, faces=face_dict)
    return graph_dict


def dict_to_graph(graph_dict: dict) -> ec.half.EuclideanPositionHEG:
    """Inverse of :func:`graph_to_dict`: reconstruct a graph from its serialised dict.

    The returned graph is always an :class:`EuclideanPositionHEG` regardless of
    the source graph's class (TODO: persist the class).
    """

    def unwrap_attributes(obj_dict):
        result = {}
        for key, value in obj_dict.pop("attributes", {}).items():
            if isinstance(value, list):
                try:
                    value = np.array(value, dtype=np.float64)
                except Exception:
                    pass
            result[key] = value
        return result

    lookup = {None: None}
    # create the halfedges
    for label in graph_dict["halfedges"]:
        lookup[label] = HalfEdge()

    vs = set()
    for label, v_dict in graph_dict["vertices"].items():
        attrs = unwrap_attributes(v_dict)
        v_dict["any_outgoing"] = lookup[v_dict["any_outgoing"]]
        v = Vertex(**v_dict)
        v.attributes = attrs
        lookup[label] = v
        vs.add(v)

    fs = set()
    for label, f_dict in graph_dict["faces"].items():
        attrs = unwrap_attributes(f_dict)
        f_dict["any_side"] = lookup[f_dict["any_side"]]
        f = Face(**f_dict)
        f.attributes = attrs
        lookup[label] = f
        fs.add(f)

    hs = set()
    for label, h_dict in graph_dict["halfedges"].items():
        attrs = unwrap_attributes(h_dict)
        h = lookup[label]
        for key in ["orig", "dest", "rev", "nex", "pre", "face"]:
            setattr(h, key, lookup[h_dict.pop(key, None)])
        h.attributes = attrs
        hs.add(h)

    # TODO make it so the class can be specified
    G = ec.half.EuclideanPositionHEG()
    G.vertices = vs
    G.faces = fs
    G.halfedges = hs
    return G


def save_graph(
    filename: str,
    graph: HalfEdgeGraph,
    overwrite: bool = False,
    extra_attributes_to_save: str | tuple[str, ...] | None = None,
    attributes_to_save: tuple[str, ...] = ("pos", "length", "in_angle", "color_key"),
) -> None:
    """Save *graph* to a ``.heg`` (YAML) file.

    Args:
        filename: Output path; ``.heg`` is appended if missing.
        graph: The graph to save.
        overwrite: If False and the file already exists, raise instead of overwriting.
        extra_attributes_to_save: Convenience: attribute key(s) to save in
            addition to *attributes_to_save*.
        attributes_to_save: Attribute keys to persist (see :func:`graph_to_dict`).
    """
    if not filename.endswith(".heg"):
        filename += ".heg"
    if not overwrite:
        assert not os.path.exists(filename), f"File exists: {filename}. Set overwrite=True to overwrite."
    if extra_attributes_to_save is not None:
        if isinstance(extra_attributes_to_save, str):
            extra_attributes_to_save = (extra_attributes_to_save,)
        attributes_to_save = tuple(extra_attributes_to_save) + tuple(attributes_to_save)
    graph_dict = graph_to_dict(graph, attributes_to_save=attributes_to_save)
    with open(filename, "w") as f:
        f.write(yaml.dump(graph_dict))


def load_graph(filename: str) -> ec.half.EuclideanPositionHEG:
    """Load a graph previously saved by :func:`save_graph`."""
    with open(filename, "r") as f:
        lines = f.read()
    graph_dict = yaml.load(lines, Loader=yaml.SafeLoader)
    return dict_to_graph(graph_dict)


# ===========================================================================
# CirclePack `.p` format
# ===========================================================================
#
# A `.p` file is the format read/written by Ken Stephenson's CirclePack tool.
# It encodes a triangulated disk with optional circle radii and centers.
# Layout (whitespace-tolerant; sections separated by blank lines):
#
#     NODECOUNT:  N
#     GEOMETRY:   euclidean | hyperbolic | spherical
#     ALPHA/BETA/GAMMA:  a b g                 (1-indexed anchor triple)
#     PACKNAME:   foo.p                         (optional, ignored)
#     FLOWERS:
#       i deg   n_1 n_2 ... n_{deg+1}          (1-indexed; interior = closed
#                                               cycle with last == first;
#                                               boundary = open list)
#     RADII:
#       r_1 r_2 ...                            (euclidean radii or hyperbolic
#                                               x-radii, depending on GEOMETRY)
#     CENTERS:
#       x_1 y_1 x_2 y_2 ...                    (one (x, y) per vertex)
#     END


@dataclass
class CirclePackData:
    """Structured contents of a CirclePack `.p` file.

    All vertex indices are 0-indexed internally even though the file format
    stores them 1-indexed. Use :func:`parse_p_file` to read and
    :func:`write_p_file` to write this dataclass verbatim.
    """

    nodecount: int
    geometry: str | None
    alpha: int | None  # 1-indexed anchor (None if no ALPHA/BETA/GAMMA in file)
    beta: int | None
    gamma: int | None
    flowers: dict[int, list[int]]  # 0-indexed: center -> CCW neighbor list
    radii: np.ndarray | None  # shape (N,) — euclidean radii or hyperbolic x-radii
    centers: np.ndarray | None  # shape (N, 2)


def parse_p_file(path: str) -> CirclePackData:
    """Parse a CirclePack `.p` file into a :class:`CirclePackData` dataclass."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if lines[-1] != "END":
        raise ValueError(f"{path}: expected final line 'END', got {lines[-1]!r}")
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
        for line in sections["FLOWERS"]:
            parts = line.split()
            center = int(parts[0]) - 1
            degree = int(parts[1])
            neighbors = [int(x) - 1 for x in parts[2:]]
            if len(neighbors) != degree + 1:
                raise ValueError(f"FLOWERS line {line!r}: expected {degree + 1} neighbors, got {len(neighbors)}")
            flowers[center] = neighbors

    radii: np.ndarray | None = None
    if "RADII" in sections:
        nums: list[float] = []
        for line in sections["RADII"]:
            nums.extend(float(x) for x in line.split())
        radii = np.array(nums)
        if len(radii) != nodecount:
            raise ValueError(f"RADII has {len(radii)} entries, expected {nodecount}")

    centers: np.ndarray | None = None
    if "CENTERS" in sections:
        nums = []
        for line in sections["CENTERS"]:
            nums.extend(float(x) for x in line.split())
        centers = np.array(nums).reshape(-1, 2)
        if len(centers) != nodecount:
            raise ValueError(f"CENTERS has {len(centers)} entries, expected {nodecount}")

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


def write_p_file(path: str, data: CirclePackData, *, overwrite: bool = False) -> None:
    """Write a :class:`CirclePackData` verbatim to a CirclePack `.p` file.

    Vertex labels and the per-vertex neighbor list order in ``data.flowers``
    are preserved exactly, so :func:`parse_p_file` ∘ :func:`write_p_file`
    round-trips losslessly (modulo float precision).
    """
    if not overwrite and os.path.exists(path):
        raise FileExistsError(f"File exists: {path}. Set overwrite=True to overwrite.")

    sections: list[str] = []
    sections.append(f"NODECOUNT: {data.nodecount}")
    if data.geometry is not None:
        sections.append(f"GEOMETRY: {data.geometry}")
    if data.alpha is not None and data.beta is not None and data.gamma is not None:
        sections.append(f"ALPHA/BETA/GAMMA: {data.alpha} {data.beta} {data.gamma}")

    flower_lines = ["FLOWERS:"]
    for i, neighbors in data.flowers.items():
        degree = len(neighbors) - 1
        flower_lines.append(f"{i + 1} {degree}   " + " ".join(str(n + 1) for n in neighbors))
    sections.append("\n".join(flower_lines))

    if data.radii is not None:
        radii_lines = ["RADII:"]
        rs = np.asarray(data.radii).ravel()
        for j in range(0, len(rs), 4):
            chunk = rs[j : j + 4]
            radii_lines.append("   ".join(f"{float(r):.16e}" for r in chunk))
        sections.append("\n".join(radii_lines))

    if data.centers is not None:
        center_lines = ["CENTERS:"]
        cs = np.asarray(data.centers).reshape(-1, 2)
        for j in range(0, len(cs), 2):
            chunk = cs[j : j + 2]
            parts = [f"{float(c[0]):.16e} {float(c[1]):.16e}" for c in chunk]
            center_lines.append("  ".join(parts))
        sections.append("\n".join(center_lines))

    sections.append("END")
    text = "\n\n".join(sections) + "\n"
    with open(path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# HEG <-> CirclePackData bridge
# ---------------------------------------------------------------------------


def _build_heg_from_data(
    data: CirclePackData,
) -> tuple[EuclideanPositionHEG, dict[int, Vertex]]:
    """Build an EuclideanPositionHEG with the combinatorial structure from FLOWERS.

    Returns ``(graph, idx_to_vertex)``. Vertex positions and radii are not set
    by this helper; :func:`load_circlepack` populates them afterwards.

    The outer face is identified by the convention that for a boundary vertex
    ``v`` with open flower ``[w_0, w_1, ...]``, the outer face is
    ``(v -> w_0).rev.face`` — that is, the outer face lies on the right of the
    edge from ``v`` to its first flower neighbor.
    """
    G = EuclideanPositionHEG()
    idx_to_v: dict[int, Vertex] = {}
    for i in range(data.nodecount):
        v = Vertex()
        v["pos"] = np.array([0.0, 0.0])
        idx_to_v[i] = v
    G.add_vertices(list(idx_to_v.values()))

    h_lookup: dict[Vertex, dict[Vertex, HalfEdge]] = {}
    for i, neighbors in data.flowers.items():
        v = idx_to_v[i]
        h_lookup[v] = {}
        # Interior flowers close on themselves (last == first); strip that duplicate.
        unique = neighbors[:-1] if (neighbors and neighbors[0] == neighbors[-1]) else neighbors
        for j in unique:
            w = idx_to_v[j]
            h = HalfEdge(orig=v, dest=w)
            h_lookup[v][w] = h
            v.any_outgoing = h
        G.add_halfedges(list(h_lookup[v].values()))

    for v in h_lookup:
        for w, h in h_lookup[v].items():
            h.rev = h_lookup[w][v]

    for i, neighbors in data.flowers.items():
        v = idx_to_v[i]
        is_closed = neighbors[0] == neighbors[-1]
        unique = neighbors[:-1] if is_closed else neighbors
        outgoing = [h_lookup[v][idx_to_v[j]] for j in unique]
        for h_revnex, h, h_prerev in rotate_by(outgoing, (0, 1, 2)):
            h.rev.nex = h_revnex
            h.pre = h_prerev.rev

    unassigned = copy(G.halfedges)
    while unassigned:
        h_start = next(iter(unassigned))
        f = Face(any_side=h_start)
        G.add_face(f)
        for k in f.halfedge_iter():
            k.face = f
            unassigned.discard(k)

    G.check_consistency()

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
        # Closed surface — pick the largest face as a fallback (sphere case).
        outer_face = max(G.faces, key=lambda f: f.order())
    G.delete_face(outer_face)

    return G, idx_to_v


def _r_eucl_from_x_and_center(x: float, c: complex) -> float:
    """Recover the euclidean radius of a Poincaré-disk circle from its x-radius and euclidean center.

    Inverts ``x = 4 r / ((1 + r)^2 - |c|^2)``. Picks the physically valid root
    (the small positive r with ``|c| + r < 1``).
    """
    if x >= 1.0:
        # Horocycle: |c| + r = 1.
        return float(1.0 - abs(c))
    abs_c2 = float(c.real * c.real + c.imag * c.imag)
    # x r^2 + (2x - 4) r + x (1 - |c|^2) = 0
    a = x
    b = 2.0 * x - 4.0
    cc = x * (1.0 - abs_c2)
    disc = b * b - 4.0 * a * cc
    disc = max(0.0, disc)
    # Choose the root with the minus sign (smaller positive r).
    return float((-b - np.sqrt(disc)) / (2.0 * a))


def load_circlepack(path: str) -> EuclideanPositionHEG:
    """Load a CirclePack `.p` file into an :class:`EuclideanPositionHEG`.

    Populates ``v['pos']`` and ``v['radius']`` on every vertex when the file
    provides RADII (and CENTERS, where applicable). For hyperbolic packings
    the file's RADII are interpreted as x-radii; the returned graph stores
    the corresponding euclidean (Poincaré-disk) center and radius, matching
    :func:`pleat.circle_packing.pack_hyperbolic`'s output convention. If a
    hyperbolic file omits CENTERS, the layout is computed via
    :func:`pleat.circle_packing._layout_hyperbolic`.

    The returned graph's ``geometry`` is set to :class:`EuclideanGeometry` or
    :class:`PoincareDiskModel` based on the file's GEOMETRY line; unknown or
    missing values default to :class:`EuclideanGeometry`.
    """
    data = parse_p_file(path)
    G, idx_to_v = _build_heg_from_data(data)

    is_hyperbolic = data.geometry == "hyperbolic"
    G.geometry = PoincareDiskModel if is_hyperbolic else EuclideanGeometry

    if is_hyperbolic:
        if data.radii is None:
            return G  # nothing to populate
        if data.centers is not None:
            for i in range(data.nodecount):
                c = complex(float(data.centers[i, 0]), float(data.centers[i, 1]))
                x = float(data.radii[i])
                idx_to_v[i]["pos"] = c
                idx_to_v[i]["radius"] = _r_eucl_from_x_and_center(x, c)
        else:
            # No CENTERS — lay out from x-radii.
            from .circle_packing import _choose_alpha, _choose_beta, _layout_hyperbolic

            x_radii = {idx_to_v[i]: float(data.radii[i]) for i in range(data.nodecount)}
            alpha = idx_to_v[data.alpha - 1] if data.alpha is not None else _choose_alpha(G)
            beta = idx_to_v[data.beta - 1] if data.beta is not None else _choose_beta(alpha)
            centers, eucl_radii = _layout_hyperbolic(G, x_radii, alpha, beta)
            for v in G.vertices:
                v["pos"] = centers[v]
                v["radius"] = eucl_radii[v]
    else:
        if data.centers is not None:
            for i in range(data.nodecount):
                idx_to_v[i]["pos"] = np.array([float(data.centers[i, 0]), float(data.centers[i, 1])])
        if data.radii is not None:
            for i in range(data.nodecount):
                idx_to_v[i]["radius"] = float(data.radii[i])

    return G


def _graph_to_circlepack_data(G: EuclideanPositionHEG) -> CirclePackData:
    """Build a :class:`CirclePackData` from an EHEG, assigning 1-based labels.

    Vertices are ordered with boundary first (preserving CirclePack's
    convention). For each boundary vertex the FLOWERS list starts at the
    incoming-boundary neighbor (the outgoing half-edge whose reverse has no
    face), so the outer face lies on the right of ``(v -> first_neighbor)``.
    Interior flowers are closed (first neighbor repeated at end).
    """
    vs = sorted(G.vertices, key=lambda v: not v.on_border())
    v_lookup = {v: i for i, v in enumerate(vs)}

    flowers: dict[int, list[int]] = {}
    for i, v in enumerate(vs):
        if v.on_border():
            outgoing = list(v.outgoing_iter())
            start = next(j for j, h in enumerate(outgoing) if h.rev.on_border())
            outgoing = outgoing[start:] + outgoing[:start]
            neighbors = [v_lookup[h.dest] for h in outgoing]
        else:
            neighbors = [v_lookup[w] for w in v.vertex_iter()]
            neighbors = neighbors + neighbors[:1]
        flowers[i] = neighbors

    if G.geometry is PoincareDiskModel:
        geometry: str | None = "hyperbolic"
    elif G.geometry is EuclideanGeometry:
        geometry = "euclidean"
    else:
        geometry = None

    alpha_idx = beta_idx = gamma_idx = None
    interior = [v for v in vs if not v.on_border()]
    if interior:
        from .circle_packing import _choose_alpha, _choose_beta

        alpha_v = _choose_alpha(G)
        beta_v = _choose_beta(alpha_v)
        alpha_idx = v_lookup[alpha_v] + 1
        beta_idx = v_lookup[beta_v] + 1
        for w in vs:
            if w is not alpha_v and w is not beta_v:
                gamma_idx = v_lookup[w] + 1
                break

    radii: np.ndarray | None = None
    if all("radius" in v.attributes for v in vs):
        if geometry == "hyperbolic":
            from .circle_packing import _x_radius_from_euclidean

            radii = np.array([_x_radius_from_euclidean(complex(v["pos"]), float(v["radius"])) for v in vs])
        else:
            radii = np.array([float(v["radius"]) for v in vs])

    centers: np.ndarray | None = None
    if all("pos" in v.attributes for v in vs):
        if geometry == "hyperbolic":
            centers = np.array([[complex(v["pos"]).real, complex(v["pos"]).imag] for v in vs])
        else:
            centers = np.array([np.asarray(v["pos"], dtype=float)[:2] for v in vs])

    return CirclePackData(
        nodecount=len(vs),
        geometry=geometry,
        alpha=alpha_idx,
        beta=beta_idx,
        gamma=gamma_idx,
        flowers=flowers,
        radii=radii,
        centers=centers,
    )


def save_circlepack(path: str, G: EuclideanPositionHEG, *, overwrite: bool = False) -> None:
    """Save an :class:`EuclideanPositionHEG` to a CirclePack `.p` file.

    Emits FLOWERS for the triangulation, plus RADII / CENTERS if every vertex
    has ``radius`` / ``pos`` attributes, plus GEOMETRY based on ``G.geometry``.
    Hyperbolic packings (``G.geometry is PoincareDiskModel``) are emitted with
    x-radii in the RADII section, matching CirclePack's convention.
    """
    data = _graph_to_circlepack_data(G)
    write_p_file(path, data, overwrite=overwrite)
