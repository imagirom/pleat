import os
import yaml
import numpy as np

import eucare as ec
from .half import Face, Vertex, HalfEdge


def graph_to_dict(G, attributes_to_save=('pos', 'length', 'in_angle', 'color_key')):
    # convert Graph to dict
    vertex_labels = {v: f'v{i}' for i, v in enumerate(G.vertices)}
    halfedge_labels = {h: f'h{i}' for i, h in enumerate(G.halfedges)}
    face_labels = {f: f'f{i}' for i, f in enumerate(G.faces)}

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
                result['attributes'] = attrs
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
            face=labels[h.face]
        )

    @add_attributes
    def represent_face(f):
        return dict(any_side=labels[f.any_side])

    vertex_dict = {label: represent_vertex(v) for v, label in vertex_labels.items()}
    halfedge_dict = {label: represent_halfedge(h) for h, label in halfedge_labels.items()}
    face_dict = {label: represent_face(f) for f, label in face_labels.items()}

    graph_dict = dict(vertices=vertex_dict, halfedges=halfedge_dict, faces=face_dict)
    return graph_dict


def dict_to_graph(graph_dict):
    def unwrap_attributes(obj_dict):
        result = {}
        for key, value in obj_dict.pop('attributes', {}).items():
            if isinstance(value, list):
                try:
                    value = np.array(value, dtype=np.float64)
                except Exception:
                    pass
            result[key] = value
        return result

    lookup = {None: None}
    # create the halfedges
    for label in graph_dict['halfedges']:
        lookup[label] = HalfEdge()

    vs = set()
    for label, v_dict in graph_dict['vertices'].items():
        attrs = unwrap_attributes(v_dict)
        v_dict['any_outgoing'] = lookup[v_dict['any_outgoing']]
        v = Vertex(**v_dict)
        v.attributes = attrs
        lookup[label] = v
        vs.add(v)

    fs = set()
    for label, f_dict in graph_dict['faces'].items():
        attrs = unwrap_attributes(f_dict)
        f_dict['any_side'] = lookup[f_dict['any_side']]
        f = Face(**f_dict)
        f.attributes = attrs
        lookup[label] = f
        fs.add(f)

    hs = set()
    for label, h_dict in graph_dict['halfedges'].items():
        attrs = unwrap_attributes(h_dict)
        h = lookup[label]
        for key in ['orig', 'dest', 'rev', 'nex', 'pre', 'face']:
            setattr(h, key, lookup[h_dict.pop(key, None)])
        h.attributes = attrs
        hs.add(h)

    # TODO make it so the class can be specified
    G = ec.half.EuclideanPositionHEG()
    G.vertices = vs
    G.faces = fs
    G.halfedges = hs
    return G


def save_graph(filename, graph, overwrite=False,
               extra_attributes_to_save=None, attributes_to_save=('pos', 'length', 'in_angle', 'color_key')):
    if not filename.endswith('.heg'):
        filename += '.heg'
    if not overwrite:
        assert not os.path.exists(filename), f'File exists: {filename}. Set overwrite=True to overwrite.'
    if extra_attributes_to_save is not None:
        if isinstance(extra_attributes_to_save, str):
            extra_attributes_to_save = (extra_attributes_to_save,)
        attributes_to_save = tuple(extra_attributes_to_save) + tuple(attributes_to_save)
    graph_dict = graph_to_dict(graph, attributes_to_save=attributes_to_save)
    with open(filename, 'w') as f:
        f.write(yaml.dump(graph_dict))


def load_graph(filename):
    with open(filename, 'r') as f:
        lines = f.read()
    graph_dict = yaml.load(lines, Loader=yaml.SafeLoader)
    return dict_to_graph(graph_dict)
