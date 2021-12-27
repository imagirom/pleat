import cairo
from .base import angle_to_axis, unit_vector
from .half import rotate_by
import numpy as np
from collections import Iterable
try:
    import svgwrite
except ImportError:
    svgwrite = None


def inset_corner(a, b, c, dist, eps=1e-10):
    # insets angle abc
    w = b - a
    v = b - c
    if np.linalg.norm(v - w) < eps:  # 180 degree "corner"
        return b
    alpha = ((angle_to_axis(v) - angle_to_axis(w)) % (2 * np.pi)) / 2
    diag_dist = dist / np.sin(alpha)
    return b + diag_dist * unit_vector(angle_to_axis(v) - alpha)


def inset_poly(pts, dist):
    return [inset_corner(a, b, c, dist) for a, b, c in rotate_by(pts, (0, 1, 2))]


_seed_offset = np.random.randint(2**16)


def random_color(seed=None):
    if seed is not None:
        np.random.seed(hash(seed) + _seed_offset)
    return np.random.uniform(0, 1, 3)


def is_color(obj):
    return isinstance(obj, Iterable) and len(obj) in (3, 4) and all([isinstance(c, (int, float)) for c in obj])


class CairoRenderer:
    def __init__(self, width=None, height=None, line_width='auto', vertex_radius=None,
                 scale='auto', face_inset=None, path='output.svg',
                 position_key='pos'):
        if width is None and height is None:
            width, height = 512, 512
        self.width = width if width is not None else height
        self.height = height if height is not None else width
        self.scale = scale
        self.line_width = line_width
        self.vertex_radius = vertex_radius
        self.face_inset = face_inset
        self.position_key = position_key

        #self.surface = cairo.ImageSurface(
        #    cairo.FORMAT_RGB24, self.width, self.height)
        self.surface = cairo.SVGSurface(
            path, self.width, self.height
        )
        dc = cairo.Context(self.surface)
        dc.set_line_cap(cairo.LINE_CAP_ROUND)
        dc.set_line_join(cairo.LINE_JOIN_ROUND)
        self.line_width = line_width
        dc.translate(self.width / 2, self.height / 2)
        dc.set_source_rgb(1, 1, 1)
        dc.paint()
        self.dc = dc

    def render_face(self, face, color_key='color_key'):
        dc = self.dc
        points = [v[self.position_key] for v in face.vertex_iter()]
        inset = self.face_inset
        if inset != 0:
            points = inset_poly(points, inset)
        dc.move_to(*points[-1])
        for point in points:
            dc.line_to(*point)
        if color_key in face.attributes:
            color = face.attributes.get(color_key, None)
            if not is_color(color):
                color = random_color(color)
            # stroke_color = color #* 0.5
        else:
            color = np.array((0.0, 0.0, 1.0, 0.15))
            # color = np.array((0.0, 0.0, 0.0, 0.2))
            # stroke_color = np.array((0.0, 0.0, 0.0, 0.0))

        self.set_source_color(color)
        dc.fill_preserve()
        dc.set_line_width(0)
        dc.stroke()
        return self.surface

    def set_source_color(self, color):
        if not is_color(color):
            color = random_color(color)
        if len(color) == 3:
            self.dc.set_source_rgb(*color)
        else:
            self.dc.set_source_rgba(*color)

    def render_edge(self, edge, color_key='color_key', last_pos=None, tol=1e-6):
        dc = self.dc

        if 'line_width' in edge.orig.attributes and 'line_width' in edge.dest.attributes:
            lw_orig = edge.orig['line_width']
            lw_dest = edge.dest['line_width']
            direction = edge.dest['pos'] - edge.orig['pos']
            direction /= np.linalg.norm(direction)
            dc.set_line_width(0)
            points = [
                inset_corner(edge.pre.orig['pos'], edge.orig['pos'], edge.dest['pos'], lw_orig),
                edge.orig['pos'] - direction * lw_orig / 2,
                inset_corner(edge.dest['pos'], edge.orig['pos'], edge.rev.nex.dest['pos'], lw_orig),
                inset_corner(edge.rev.pre.orig['pos'], edge.dest['pos'], edge.orig['pos'], lw_dest),
                edge.dest['pos'] + direction * lw_dest / 2,
                inset_corner(edge.orig['pos'], edge.dest['pos'], edge.nex.dest['pos'], lw_dest),
            ]
            dc.move_to(*points[-1])
            for point in points:
                dc.line_to(*point)
            self.set_source_color(edge.attributes.get(color_key, edge.attributes.get(color_key, (0.5, 0.5, 1.0))))
            dc.fill_preserve()
        else:

            dc.set_line_width(edge.attributes.get('line_width', self.line_width))
            if True or last_pos is None or np.linalg.norm(last_pos - edge.orig[self.position_key]) > tol:
                dc.move_to(*edge.orig[self.position_key])
            else:
                pass
            dc.line_to(*edge.dest[self.position_key])
            #dc.set_source_rgb(0.0, 0.0, 0.0)
            # set color. TODO: this interacts weirdly with delayed dc.stroke..
            if color_key in edge.attributes:
                self.set_source_color(edge.attributes.get(color_key, None))
            else:
                self.set_source_color((0.5, 0.5, 1.0))

            if edge.attributes.get('delete', False):
                dc.set_dash([self.line_width*2, self.line_width*3])
                dc.stroke()
                dc.set_dash([])
            else:
                pass
                #dc.stroke()
        return self.surface if last_pos is None else (self.surface, edge.dest[self.position_key])

    def render_vertex(self, vertex):
        dc = self.dc
        dc.set_line_width(0)
        dc.arc(*vertex[self.position_key], self.vertex_radius, 0, 2*np.pi)
        if vertex.attributes.get('join', False):
            dc.set_source_rgb(0.0, 1.0, 0.0)
        elif vertex.attributes.get('delete', False):
            dc.set_source_rgb(1.0, 1.0, 1.0)
        else:
            dc.set_source_rgb(0.0, 0.0, 0.0)
        dc.fill_preserve()
        dc.set_source_rgb(0.0, 0.0, 0.0)
        dc.stroke()
        dc.set_line_width(self.line_width)

    def autoscale(self, graph):
        positions = np.array([v[self.position_key] for v in graph.vertices])
        max_abs_pos = np.max(np.abs(positions), axis=0)
        scale = np.min(np.array([self.width, self.height]) / max_abs_pos) / 2.2
        self.dc.scale(scale, scale)
        return self

    def autocenterscale(self, graph):
        positions = np.array([v[self.position_key] for v in graph.vertices])
        offset = (np.max(positions, axis=0) + np.min(positions, axis=0)) / 2
        max_abs_pos = np.max(np.abs(positions - offset[None]), axis=0)
        relative_margin = 0.05
        scale = np.min(np.array([self.width, self.height]) / max_abs_pos) / 2 / (1 + 2 * relative_margin)
        self.dc.scale(scale, scale)
        self.dc.translate(-offset[0] * (1 + 0 * relative_margin), -offset[1] * (1))
        return self

    @staticmethod
    def auto_line_width(graph):
        lengths = np.array([np.linalg.norm(h.dest['pos'] - h.orig['pos']) for h in graph.halfedges])
        return min(np.min(lengths) / 2, np.mean(lengths) / 10)

    def render_graph(self, graph, render_vertices=True, render_faces=True, render_edges=True, for_cutting=False):
        global _seed_offset
        _seed_offset = np.random.randint(2**16)
        if self.scale is 'auto':
            self.autocenterscale(graph)
        else:
            self.dc.scale(self.scale, self.scale)
        #self.dc.set_font_size(18.0 / self.scale)

        if self.line_width == 'auto':
            self.line_width = self.auto_line_width(graph)
        self.vertex_radius = self.vertex_radius if self.vertex_radius is not None else self.line_width
        self.face_inset = self.face_inset if self.face_inset is not None else self.line_width

        if render_faces:
            for f in graph.faces:
                self.render_face(f)

        if render_edges:
            self.dc.set_source_rgb(0.0, 0.0, 0.0)
            if not for_cutting:
                rendered_edges = set()
                for h in graph.halfedges:
                    if h.rev in rendered_edges:
                        continue
                    rendered_edges.add(h)
                    self.render_edge(h)
                    self.dc.stroke()
            else:
                raise NotImplementedError
                # order the edges such that the plotting takes less time
                edges = list(graph.halfedges)
                edge_to_index = {e: i for i, e in enumerate(edges)}
                n_edges = len(edges)
                rendered = np.zeros(n_edges, dtype=np.int32)
                origs = np.stack([e.orig[self.position_key] for e in edges])
                current = np.argmin(origs[:, 0])
                id_range = np.arange(n_edges)
                last_pos = np.array([np.inf, np.inf])
                while True:
                    e = edges[current]
                    _, last_pos = self.render_edge(e, last_pos=last_pos)
                    rendered[current] = 1
                    rendered[edge_to_index[e.rev]] = 1
                    if all(rendered):
                        break
                    dists = np.linalg.norm(origs[rendered == 0] - e.dest[self.position_key][None], axis=1)
                    current = id_range[rendered == 0][np.argmin(dists)]
                    if np.min(dists) > 1e-6:  # TODO: hardcoding this is bad, use relative deviation..
                        self.dc.stroke()
                self.dc.stroke()

        if render_vertices:
            for v in graph.vertices:
                self.render_vertex(v)

        return self.surface


class SvgwriteRenderer:
    """This is to be used with a cutting plotter"""
    def __init__(self, position_key='pos'):
        assert svgwrite is not None, f'SvgwriteRenderer requires the svgwrite package. You can install it via pip.'
        self.position_key = position_key
        
    def render_graph(self, filename, graph, render_vertices=False, render_faces=False, render_edges=True,
                     for_cutting=True, height=30, unit=svgwrite.cm):

        # get bbox of points that will be rendered
        ps, vs = graph.get_position_view()
        bbox = np.array([[f(ps[:, i]) for f in [np.min, np.max]] for i in [0, 1]])
        aspect_ratio = (bbox[0, 1] - bbox[0, 0]) / (bbox[1, 1] - bbox[1, 0])
        scale = height / (bbox[1, 1] - bbox[1, 0])
        width = height * aspect_ratio
        dwg = svgwrite.Drawing(filename, size=(width*unit, height*unit), viewBox=f'0 0 {width} {height}')

        if render_faces:
            raise NotImplementedError

        if render_edges:
            if not for_cutting:
                raise NotImplementedError
            else:
                # order the edges such that the plotting takes less time
                edges = list(graph.halfedges)
                edge_to_index = {e: i for i, e in enumerate(edges)}
                n_edges = len(edges)
                rendered = np.zeros(n_edges, dtype=np.int32)
                origs = np.stack([e.orig[self.position_key] for e in edges])
                current = int(np.argmin(origs[:, 0]))
                id_range = np.arange(n_edges)
                polylines = []
                current_polyline = [edges[current].orig[self.position_key]]
                while True:
                    e = edges[current]
                    last_pos = e.dest[self.position_key]
                    current_polyline.append(last_pos)
                    rendered[current] = 1
                    rendered[edge_to_index[e.rev]] = 1
                    if all(rendered):
                        break
                    dists = np.linalg.norm(origs[rendered == 0] - e.dest[self.position_key][None], axis=1)
                    current = id_range[rendered == 0][np.argmin(dists)]
                    if np.min(dists) > 1e-6:  # TODO: hardcoding this is bad, use relative deviation..
                        polylines.append(current_polyline)
                        current_polyline = [edges[current].orig[self.position_key]]
                polylines.append(current_polyline)

                for pts in polylines:
                    pts = np.array(pts)
                    pts -= bbox[:, 0][None]
                    pts *= scale
                    pth = dwg.path(fill_opacity=0, stroke_width='0.05', stroke='black')
                    pth.push('M', *pts)
                    dwg.add(pth)

        if render_vertices:
            raise NotImplementedError

        dwg.save()
        return dwg