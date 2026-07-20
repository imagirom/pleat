"""File I/O for the ``.heg`` half-edge graph format (YAML-based)."""

from __future__ import annotations

import os

import numpy as np
import yaml

import pleat

from ..half import Face, HalfEdge, HalfEdgeGraph, Vertex


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
                    if isinstance(value, (str, bytes, bool)):
                        pass  # e.g. a hex colour_key like "#cc2222" -- keep as-is
                    elif np.iscomplexobj(value):
                        c = complex(value)  # type: ignore[arg-type]
                        value = {"complex": [c.real, c.imag]}
                    else:
                        value = float(value)  # type: ignore[arg-type]
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


def dict_to_graph(graph_dict: dict) -> pleat.half.EuclideanPositionHEG:
    """Inverse of :func:`graph_to_dict`: reconstruct a graph from its serialised dict.

    The returned graph is always an :class:`EuclideanPositionHEG` regardless of
    the source graph's class (TODO: persist the class).
    """

    def unwrap_attributes(obj_dict):
        result = {}
        for key, value in obj_dict.pop("attributes", {}).items():
            if isinstance(value, dict) and set(value) == {"complex"}:
                value = np.complex128(complex(*value["complex"]))
            elif isinstance(value, list):
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
    G = pleat.half.EuclideanPositionHEG()
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
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File exists: {filename}. Set overwrite=True to overwrite.")
    if extra_attributes_to_save is not None:
        if isinstance(extra_attributes_to_save, str):
            extra_attributes_to_save = (extra_attributes_to_save,)
        attributes_to_save = tuple(extra_attributes_to_save) + tuple(attributes_to_save)
    graph_dict = graph_to_dict(graph, attributes_to_save=attributes_to_save)
    with open(filename, "w") as f:
        f.write(yaml.dump(graph_dict))


def load_graph(filename: str) -> pleat.half.EuclideanPositionHEG:
    """Load a graph previously saved by :func:`save_graph`."""
    with open(filename, "r") as f:
        lines = f.read()
    graph_dict = yaml.load(lines, Loader=yaml.SafeLoader)
    return dict_to_graph(graph_dict)
