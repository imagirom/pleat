import cairo
from .base import angle_to_axis, unit_vector
from .half import rotate_by
import numpy as np


def inset_corner(a, b, c, dist):
    # insets angle abc
    w = b - a
    v = b - c
    alpha = ((angle_to_axis(v) - angle_to_axis(w)) % (2 * np.pi)) / 2
    diag_dist = dist / np.sin(alpha)
    return b + diag_dist * unit_vector(angle_to_axis(v) - alpha)


def inset_poly(pts, dist):
    return [inset_corner(a, b, c, dist) for a, b, c in rotate_by(pts, (0, 1, 2))]


class CairoRenderer:
    def __init__(self, width=1000, height=1000, line_width=0.2, scale=1, face_inset=0.15):
        self.width = width
        self.height = height
        self.scale = scale
        self.line_width = line_width
        self.face_inset = face_inset

        #self.surface = cairo.ImageSurface(
        #    cairo.FORMAT_RGB24, self.width, self.height)
        self.surface = cairo.SVGSurface(
            'output.svg', self.width, self.height
        )
        dc = cairo.Context(self.surface)
        dc.set_line_cap(cairo.LINE_CAP_ROUND)
        dc.set_line_join(cairo.LINE_JOIN_ROUND)
        dc.set_line_width(self.line_width)
        dc.set_font_size(18.0 / self.scale)
        dc.translate(self.width / 2, self.height / 2)
        dc.scale(self.scale, self.scale)
        dc.set_source_rgb(1, 1, 1)
        dc.paint()
        self.dc = dc

    def render_face(self, face):
        dc = self.dc
        points = [v['pos'] for v in face.vertex_iter()]
        inset = self.face_inset
        if inset != 0:
            points = inset_poly(points, inset)
        dc.move_to(*points[-1])
        for point in points:
            dc.line_to(*point)
        color = np.random.uniform(0, 1, 3)
        dc.set_source_rgb(*color)
        dc.fill_preserve()
        dc.set_source_rgb(*(color * 0.5))
        dc.stroke()
        return self.surface

    def render_edge(self, edge):
        dc = self.dc
        dc.move_to(*edge.orig['pos'])
        dc.line_to(*edge.dest['pos'])
        dc.set_source_rgb(0.0, 0.0, 0.0)
        if edge.attributes.get('delete', False):
            dc.set_dash([self.line_width*2, self.line_width*3])
        dc.stroke()
        dc.set_dash([])
        return self.surface

    def render_vertex(self, vertex):
        dc = self.dc
        dc.arc(*vertex['pos'], 2*self.line_width, 0, 2*np.pi)
        if vertex.attributes.get('join', False):
            dc.set_source_rgb(0.0, 1.0, 0.0)
        elif vertex.attributes.get('delete', False):
            dc.set_source_rgb(1.0, 1.0, 1.0)
        else:
            dc.set_source_rgb(0.0, 0.0, 0.0)
        dc.fill_preserve()
        dc.set_source_rgb(0.0, 0.0, 0.0)
        dc.stroke()

    def render_graph(self, graph, render_vertices=True, render_faces=True, render_edges=True):
        if render_faces:
            for f in graph.faces:
                self.render_face(f)

        if render_edges:
            rendered_edges = set()
            for h in graph.halfedges:
                if h.rev in rendered_edges:
                    continue
                rendered_edges.add(h)
                self.render_edge(h)

        if render_vertices:
            for v in graph.vertices:
                self.render_vertex(v)

        return self.surface
