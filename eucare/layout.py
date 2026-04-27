import numpy as np
from copy import copy


def angle_to_height(G, angle):
    border_positions = np.array([v['pos'] for v in G.border_vertex_iter()])
    rot_border_positions = border_positions @ np.array([[np.cos(angle)], [-np.sin(angle)]])
    return np.max(rot_border_positions) - np.min(rot_border_positions)


def optimal_rotation(G, angle_offset=0, steps=10000):
    border_positions = np.array([v['pos'] for v in G.border_vertex_iter()])

    def _angle_to_height(angle):
        rot_border_positions = border_positions @ np.array([[np.cos(angle)], [-np.sin(angle)]])
        return np.max(rot_border_positions) - np.min(rot_border_positions)

    angles = np.linspace(0, np.pi, steps)
    heights = [_angle_to_height(a) for a in angles]
    angle = angles[np.argmin(heights)] + angle_offset
    return angle


def rotate_graph(G, angle):
    ps = G.get_position_view(return_vertices=False)
    ps[:] = ps @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])


def optimize_rotation(G, angle_offset=0):
    angle = optimal_rotation(G, angle_offset)
    rotate_graph(G, angle)
    return angle


def min_edge_length(G, include_border=True):
    edges = copy(G.halfedges)
    min_length = np.inf
    while edges:
        e = edges.pop()
        edges.remove(e.rev)
        if not include_border and (e.on_border() or e.rev.on_border()):
            continue
        min_length = min(((e.orig['pos'] - e.dest['pos']) ** 2).sum(), min_length)
    return np.sqrt(min_length)
