import cairo
from .base import angle_to_axis, unit_vector
from .half import rotate_by
import numpy as np
try:
    import svgwrite
except ImportError:
    svgwrite = None


def inset_corner(a, b, c, dist):
    # insets angle abc
    w = b - a
    v = b - c
    alpha = ((angle_to_axis(v) - angle_to_axis(w)) % (2 * np.pi)) / 2
    diag_dist = dist / np.sin(alpha)
    return b + diag_dist * unit_vector(angle_to_axis(v) - alpha)


def inset_poly(pts, dist):
    return [inset_corner(a, b, c, dist) for a, b, c in rotate_by(pts, (0, 1, 2))]


def random_color(seed=None):
    if seed is not None:
        np.random.seed(hash(seed))
    return np.random.uniform(0, 1, 3)


class CairoRenderer:
    def __init__(self, width=1000, height=1000, line_width=0.03, scale='auto', face_inset=0.06):
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
        dc.translate(self.width / 2, self.height / 2)
        dc.set_source_rgb(1, 1, 1)
        dc.paint()
        self.dc = dc

    def render_face(self, face, color_key='color_key'):
        dc = self.dc
        points = [v['pos'] for v in face.vertex_iter()]
        inset = self.face_inset
        if inset != 0:
            points = inset_poly(points, inset)
        dc.move_to(*points[-1])
        for point in points:
            dc.line_to(*point)
        if color_key in face.attributes:
            color = random_color(face.attributes.get(color_key, None))
            stroke_color = color * 0.5
        else:
            color = np.array((0.0, 0.0, 0.0, 0.1))
            stroke_color = np.array((0.0, 0.0, 0.0, 0.0))

        if len(color) == 3:
            dc.set_source_rgb(*color)
        else:
            dc.set_source_rgba(*color)

        dc.fill_preserve()

        if len(color) == 3:
            dc.set_source_rgb(*stroke_color)
        else:
            dc.set_source_rgba(*stroke_color)

        dc.stroke()
        return self.surface

    def render_edge(self, edge, last_pos=None, tol=1e-6):
        dc = self.dc
        if True or last_pos is None or np.linalg.norm(last_pos - edge.orig['pos']) > tol:
            dc.move_to(*edge.orig['pos'])
        else:
            pass
        dc.line_to(*edge.dest['pos'])
        #dc.set_source_rgb(0.0, 0.0, 0.0)
        if edge.attributes.get('delete', False):
            dc.set_dash([self.line_width*2, self.line_width*3])
            dc.stroke()
            dc.set_dash([])
        else:
            pass
            #dc.stroke()
        return self.surface if last_pos is None else (self.surface, edge.dest['pos'])

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

    def autoscale(self, graph):
        positions = np.array([v['pos'] for v in graph.vertices])
        max_abs_pos = np.max(np.abs(positions), axis=0)
        scale = np.min(np.array([self.width, self.height]) / max_abs_pos) / 2.2
        self.dc.scale(scale, scale)
        return self

    def render_graph(self, graph, render_vertices=True, render_faces=True, render_edges=True, for_cutting=False):
        if self.scale is 'auto':
            self.autoscale(graph)
        else:
            self.dc.scale(self.scale, self.scale)
        #self.dc.set_font_size(18.0 / self.scale)

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
            else:
                # order the edges such that the plotting takes less time
                edges = list(graph.halfedges)
                edge_to_index = {e: i for i, e in enumerate(edges)}
                n_edges = len(edges)
                rendered = np.zeros(n_edges, dtype=np.int32)
                origs = np.stack([e.orig['pos'] for e in edges])
                current = np.argmin(origs[:, 0])
                id_range = np.arange(n_edges)
                last_pos = np.array([np.inf, np.inf])
                while True:
                    e = edges[current]
                    _, last_pos = self.render_edge(e, last_pos)
                    rendered[current] = 1
                    rendered[edge_to_index[e.rev]] = 1
                    if all(rendered):
                        break
                    dists = np.linalg.norm(origs[rendered == 0] - e.dest['pos'][None], axis=1)
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
    def __init__(self):
        assert svgwrite is not None, f'SvgwriteRenderer requires the svgwrite package. You can install it via pip.'
        
    def render_graph(self, filename, graph, render_vertices=False, render_faces=False, render_edges=True,
                     for_cutting=True, height=30, unit=svgwrite.cm):

        # get bbox of points that will be rendered
        ps, vs = graph.get_position_view()
        bbox = np.array([[f(ps[:, i]) for f in [np.min, np.max]] for i in [0, 1]])
        aspect_ratio = (bbox[0, 1] - bbox[0, 0]) / (bbox[1, 1] - bbox[1, 0])
        print(bbox)
        scale = height / (bbox[1, 1] - bbox[1, 0])
        print(scale)
        width = height * aspect_ratio
        print(width)
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
                origs = np.stack([e.orig['pos'] for e in edges])
                current = int(np.argmin(origs[:, 0]))
                id_range = np.arange(n_edges)
                polylines = []
                current_polyline = [edges[current].orig['pos']]
                while True:
                    e = edges[current]
                    last_pos = e.dest['pos']
                    current_polyline.append(last_pos)
                    rendered[current] = 1
                    rendered[edge_to_index[e.rev]] = 1
                    if all(rendered):
                        break
                    dists = np.linalg.norm(origs[rendered == 0] - e.dest['pos'][None], axis=1)
                    current = id_range[rendered == 0][np.argmin(dists)]
                    if np.min(dists) > 1e-6:  # TODO: hardcoding this is bad, use relative deviation..
                        polylines.append(current_polyline)
                        current_polyline = [edges[current].orig['pos']]
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