from collections import defaultdict
import numpy as np
import numba
from numba import jit, njit

import eucare as ec
from eucare.overlap import line_segment_intersections
from eucare.redering import inset_poly


"""Methods to check if point lies in polygon from https://stackoverflow.com/a/48760556"""


@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D


@numba.jit(nopython=True)
def polygon_line_segment_intersections(poly, line_segment, eps=1e-12):
    intersections = []
    p1 = poly[-1]
    for i in range(len(poly)):
        p2 = poly[i]
        intersections.extend(line_segment_intersections(line_segment, np.stack((p1, p2)), eps))
        p1 = p2
    return intersections


class CuttingRegion:
    def inside(self, points):
        raise NotImplementedError

    def intersections(self, line_segments):
        raise NotImplementedError

    def corners(self):
        return []


class Halfplane(CuttingRegion):
    def __init__(self, p, v):
        """Halfplane defined by (x - p) * v > 0"""
        self.p = p
        self.v = v / np.linalg.norm(v)

    def signed_distance(self, points):
        return -((points - self.p) * self.v).sum(-1)

    def intersections(self, line_segments):
        """
        Compute intersections of the half-plane boundary and the line_segments.
        They are assumed to exist.

        The line segments have shape (n_segments, (start, end), (x, y))
        """
        mixing_coefficients = np.abs(((line_segments - self.p) * self.v).sum(-1, keepdims=True))
        mixing_coefficients /= mixing_coefficients.sum(1, keepdims=True)
        mixing_coefficients = 1 - mixing_coefficients
        return (line_segments * mixing_coefficients).sum(1)


class Polygon(CuttingRegion):
    def __init__(self, pts, eps=1e-6):
        self.pts = np.array(pts)
        self.eps = eps
        self.inset_pts = inset_poly(pts, self.eps/2)
        self.outset_pts = inset_poly(pts, -self.eps/2)
        assert len(self.pts.shape) == 2 and self.pts.shape[1] == 2, f'{self.pts.shape}'

    def signed_distance(self, points):
        return 1 - parallelpointinpolygon(points, self.inset_pts) - parallelpointinpolygon(points, self.outset_pts)

    def intersections(self, line_segments):
        result = []
        for ls in line_segments:
            intersections = polygon_line_segment_intersections(self.pts, ls, self.eps)
            result.append(intersections[0])
        return np.stack(result)

    def corners(self):
        return self.pts


class Circle(Polygon):
    def __init__(self, p, r, res=64, **kwargs):
        t = np.linspace(0, 2*np.pi, res+1)[:-1]
        super().__init__(p + r * np.stack([np.sin(t), np.cos(t)], axis=-1), **kwargs)


def cut_graph(G, region, delete_outside=True, eps=1e-6):
    """
    Cuts a speciefied region out of the graph.
    The following configuratino is currently problematic, the corner (on the left) will not be placed

    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMNKKXWMMMMMMMMMMMMMMMMMMMMMMMMMMWXOxOWMMMMMM
    MMMMMMMMMMMMMMMMMMMMMO'.,OMMMMMMMMMMMMMMMMMMMMMMNKOdl;''c0MMMMMM
    MMMMMMMMMMMMMMMMMMMMMx. .OMMMMMMMMMMMMMMMMMWNKkdc;'';cdOKWMMMMMM
    MMMMMMMMMMMMMMMMMMMMMx. .OMMMMMMMMMMMMMWNKkoc;'';ldOKWWMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMx. .OMMMMMMMMMWN0koc,'';ldOXWWMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMx. .OMMMMWXK0xo:,',:lxOXWWMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMx. .OWX0xl:;,',:lx0XWMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMx. .ll:,',:lox0XWMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMWXOc...',cok0NWWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMWWKOdl;''..'oKNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMWNKkdc;'';cdc. 'OMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMWNKkdc;'';cdOKNWd. '0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMWKkoc,'';ldOXWMMMMMd  '0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMNOo:,'',cxKNMMMMMMMd  '0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMWN0xl;'';ldOXWWMMd  '0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMWXOdc;',:ox0Nd  '0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMWNKko:,',c;. ,0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMWX0xl;''.,o0NWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMWKl...',:okKNWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMd  'oo:'';codOXWMMMMMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMd  ,0WXOdl;,'';lx0NWMMMMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMd  ,KMMMMWK0ko:,',:okKNMMMMMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMo  ,KMMMMMMMMWN0xl;'';ldOXWMMMMMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMo  ,KMMMMMMMMMMMMWXOdc;'':lx0NWMMMMMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMo  ,KMMMMMMMMMMMMMMMWNKko:,',cokKWWMMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMk,'lXMMMMMMMMMMMMMMMMMMMWN0xl;'';ld0WMMMMMM
    MMMMMMMMMMMMMMMMMMMMMWWWWMMMMMMMMMMMMMMMMMMMMMMMWWXOdc,;xNMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWNKKNMMMMMMM
    MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
    :param G: EuclideanPositionHEG
    The graph to be cut.
    :param region: CuttingRegion
    The region to be cut out.
    :param delete_outside: bool, optional
    Whether the part of the graph outside the region should be deleted. Defaults to true.
    :param eps: float, optional
    Numerical epsilon: Distance below wich points are considered to be coincident.
    :return:
    """
    assert isinstance(region, CuttingRegion), f'{region}'
    pts, vs = G.get_position_view()
    dists = region.signed_distance(pts)
    for v, d in zip(vs, dists):
        v['signed_distance'] = d if abs(d) > eps else 0

    # find edges that need to be split and split them
    hs = []
    fs = []
    for h in G.halfedges:
        if h.orig['signed_distance'] < 0 and h.dest['signed_distance'] > 0:
            hs.append(h)
        if h.orig['signed_distance'] <= 0 and h.dest['signed_distance'] >= 0:
            fs.extend([h.face, h.rev.face])
    fs = list(set(f for f in fs if f is not None))  # delete duplicates

    corners = region.corners()
    corner_dict = defaultdict(list)
    if len(corners):
        corners_in_faces = []
        for f in fs:
            corners_in_faces.append(parallelpointinpolygon(corners, np.stack([v['pos'] for v in f.vertex_iter()])))
        corners_in_faces = np.stack(corners_in_faces, axis=-1)
        for i, (corner, ind) in enumerate(zip(corners, corners_in_faces)):
            if np.sum(ind) != 1:
                continue
            f = fs[int(np.argwhere(ind))]
            if np.min([np.linalg.norm(v['pos'] - corner) for v in f.vertex_iter()]) < eps:
                continue
            corner_dict[f].append(i)

    line_segments = np.array([[h.orig['pos'], h.dest['pos']] for h in hs])
    if len(fs) == 0:
        return

    if len(line_segments) > 0:
        intersections = region.intersections(line_segments)
        for h, pos in zip(hs, intersections):
            G.subdivide_edge(h, pos=pos, signed_distance=0)

    # add extra edges
    for f in fs:
        v_inside = set()
        v_outside = set()
        v_border = set()
        for v in f.vertex_iter():
            d = v['signed_distance']
            if d > 0:
                v_outside.add(v)
            elif d < 0:
                v_inside.add(v)
            else:
                v_border.add(v)
        if v_inside and v_outside:
            assert len(v_border) == 2, f'{v_inside}, {v_outside}, {v_border}'
            v1, v2 = list(v_border)

            # make it so that the nex path from v1 to v2 is inside
            v1_out = next(h for h in f.halfedge_iter() if h.orig is v1)
            if v1_out.dest['signed_distance'] > 0:
                v1, v2 = v2, v1
                v1_out = next(h for h in f.halfedge_iter() if h.orig is v1)

            G.subdivide_face(f, v1, v2)

            if f in corner_dict:
                corners_in_face = corner_dict[f]

                if 0 in corners_in_face:
                    # sort corners
                    for i, (c1, c2) in enumerate(zip(corners_in_face[:-1], corners_in_face[1:])):
                        if c2 - c1 > 1:
                            break
                    i = i + 1
                    corners_in_face = corners_in_face[i:] + corners_in_face[:i]

                # add the corners by subdividing the new edge
                new_edge = v1_out.pre.rev
                for i in corners_in_face:
                    new_edge['new_border'] = True
                    G.subdivide_edge(new_edge, pos=corners[i])
                    new_edge = new_edge.nex
                new_edge['new_border'] = True

            else:
                v1_out.pre.rev['new_border'] = True

        elif v_outside and len(v_border) > 1:
            assert len(v_border) == 2, f'{v_inside}, {v_border}'
            v1, v2 = list(v_border)
            v1_out = next(h for h in f.halfedge_iter() if h.orig is v1)
            assert v1_out.dest is v2 or v1_out.pre.orig is v2, f'{v1}, {v2}'
            h = v1_out if v1_out.dest is v2 else v1_out.pre
            h['new_border'] = True

    if delete_outside:
        # floodfill the outside region and delete it
        to_delete = {h.face for h in G.halfedges if 'new_border' in h.attributes}
        border = to_delete.copy()
        while border:
            f = border.pop()
            for h in f.halfedge_iter():
                if 'new_border' in h.attributes:
                    continue
                f2 = h.rev.face
                if f2 and f2 not in to_delete:
                    to_delete.add(f2)
                    border.add(f2)
        G.delete_subset(to_delete)

    for v in G.vertices:
        if 'signed_distance' in v.attributes:
            del v['signed_distance']

    for h in G.halfedges:
        if 'new_border' in h.attributes:
            del h['new_border']
