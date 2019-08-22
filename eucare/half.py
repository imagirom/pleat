from itertools import chain
import matplotlib.pyplot as plt
import networkx as nx
from copy import copy
from math import pi
import numpy as np
from collections import deque
from .base import unit_vector, angle_to_axis, edge_lengths_and_in_angles

# TODO: the big ones
# - add rendering for proto tiles
# - conway: add border styles
# - add non euclidean geometries
# - add functions to expand tilings, e.g. all border edges / vertices,
#   or until a certain region (rectangle or sphere or more general..) is filled with the tiling.
#   (partially done, confusion on what is best..)
# - add function to compute angles for InAngleHEG (to apply after conway)

# TODO: optional stuff
# - add functionality to select parts of the tiling (all borders with pentagons adjacent to them)
# - a class without explicit faces

# TODO: cleanup stuff
# - separate InAngleHEG and EuclideanPositionHEG?
# - make up mind about e or h for halfedges


class AttributeObject:
    def __init__(self):
        super(AttributeObject, self).__init__()
        self.attributes = dict()

    def has_attributes(self):
        return bool(self.attributes)

    def __getitem__(self, attr):
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        self.attributes[attr] = value

    def __iter__(self):
        return iter(self.attributes)

    def __delitem__(self, key):
        del self.attributes[key]

    def __contains__(self, item):
        return item in self.attributes

    def keys(self):
        return self.attributes.keys()

    def values(self):
        return self.attributes.values()

    def items(self):
        return self.attributes.items()


# TODO: remove this or solve in a better way
class IdObject(AttributeObject):
    current_ids = dict()

    def __init__(self):
        super(IdObject, self).__init__()
        cls = type(self)
        IdObject.current_ids[cls] = IdObject.current_ids.get(cls, 0) + 1
        self['id'] = IdObject.current_ids[cls]

    def __repr__(self):
        cls = type(self)
        clspre = cls.printname if hasattr(cls, 'printname') else cls.__name__
        return f'{clspre}{self["id"]}'

    @classmethod
    def reset_ids(cls):
        if cls is IdObject:
            #print('resetting all ids')
            IdObject.current_ids = dict()
        else:
            IdObject.current_ids[cls] = 0


def check_cyclic_iterator_consitency(iterator):
    items = set()
    for item in iterator:
        assert item not in items, f'{iterator}, {item}, {items}'
        items.add(item)


class HalfEdge(IdObject):
    printname = 'HE'

    def __init__(self, rev=None, nex=None, pre=None, orig=None, dest=None, face=None):
        super(HalfEdge, self).__init__()
        # HalfEdge references
        self.rev = rev
        self.nex = nex
        self.pre = pre

        # Vertex references
        self.orig = orig
        self.dest = dest

        # Face reference
        self.face = face

    def __repr__(self):
        return f'{self}({self.orig},{self.dest})'

    def __str__(self):
        return f'H{self["id"]}'

    def on_border(self):
        return self.face is None

    def check_consistency(self):
        assert self.rev.rev is self, f'{self}, {self.rev}, {self.rev.rev}'


class Vertex(IdObject):
    printname = 'V'

    def __init__(self, any_outgoing=None):
        super(Vertex, self).__init__()
        self.any_outgoing = any_outgoing

    def outgoing_iter(self):
        initial = self.any_outgoing
        current = initial
        while True:
            yield current
            current = current.pre.rev
            if current is initial:
                break

    def reverse_outgoing_iter(self):
        initial = self.any_outgoing
        current = initial
        while True:
            yield current
            current = current.rev.nex
            if current is initial:
                break

    def order(self):
        return len(list(self.outgoing_iter()))

    def incoming_iter(self):
        for h in self.outgoing_iter():
            yield h.rev

    def face_iter(self):
        for h in self.outgoing_iter():
            yield h.face

    def vertex_iter(self):
        for h in self.outgoing_iter():
            yield h.dest

    def on_border(self):
        return any(h.on_border() for h in self.outgoing_iter())

    def get_outgoing_border(self):
        # search for borders
        border_edges = [h for h in self.outgoing_iter() if h.face is None]
        assert len(border_edges) > 0, f'Vertex {self} does not lie on a border. {self.order()}'
        assert len(border_edges) < 2, f'Vertex {self} has multiple adjacent border edges. Please specify one.'
        return border_edges[0]

    def combine_with(self, other):
        # This method might be used in subclasses to e.g. average positions when vertices are combined.
        # For now, just use one of them.
        return self

    def check_consistency(self):
        check_cyclic_iterator_consitency(self.outgoing_iter())
        check_cyclic_iterator_consitency(self.reverse_outgoing_iter())
        for e in self.outgoing_iter():
            assert e.orig is self, f'{self}, {e.orig}, {e}'
        for e in self.incoming_iter():
            assert e.dest is self, f'{self}, {e.dest}, {e}'
        assert self.order() > 0, f'{self}, {self.order()}' # TODO: > 1

    def angle_sum(self):
        return sum(h['in_angle']
                   for h in self.incoming_iter()
                   if h.face is not None)


class Face(IdObject):
    def __init__(self, any_side=None):
        super(Face, self).__init__()
        self.any_side = any_side

    def halfedge_iter(self):
        initial = self.any_side
        current = initial
        while True:
            yield current
            current = current.nex
            if current is initial:
                break

    def reverse_halfedge_iter(self): # TODO: this should be the reverse of halfedge_iter, no? (not returning the rev)
        for h in self.halfedge_iter():
            yield h.rev

    def order(self):
        return len(list(self.halfedge_iter()))

    def vertex_iter(self):
        for h in self.halfedge_iter():
            yield h.orig

    def face_iter(self):
        for h in self.halfedge_iter():
            yield h.rev.face

    def midpoint(self):
        return np.mean([v['pos'] for v in self.vertex_iter()], axis=0)

    def recompute_lengths_and_angles(self):
        points = np.stack([v['pos'] for v in self.vertex_iter()])
        lengths, angles = edge_lengths_and_in_angles(points)
        for e, length, angle in zip(self.halfedge_iter(), lengths, angles):
            e['length'] = length
            e['in_angle'] = angle

    def check_consistency(self):
        check_cyclic_iterator_consitency(self.halfedge_iter())
        check_cyclic_iterator_consitency(self.reverse_halfedge_iter())
        for e in self.halfedge_iter():
            assert e.face is self, f'{self}, {e.face}, {e}'
        assert self.order() > 1, f'{self}, {self.order()}'


class HalfEdgeGraph:
    def __init__(self, other=None):
        if other is not None:
            self.halfedges = copy(other.halfedges)
            self.vertices = copy(other.vertices)
            self.faces = copy(other.faces)
            self._any_border = other._any_border
        else:
            self.halfedges = set()
            self.vertices = set()
            self.faces = set()
            self._any_border = None

    def add_graph(self, other):
        if not isinstance(other, HalfEdgeGraph):
            raise TypeError(f'other must be a HalfEdgeGraph. Got {type(other)}')
        self.add_halfedges(other.halfedges)
        self.add_vertices(other.vertices)
        self.add_faces(other.faces)

    @property
    def order(self):
        return len(self.vertices)

    @property
    def size(self):
        return len(self.halfedges) // 2

    def add_vertex(self, v):
        self.vertices.add(v)

    def add_vertices(self, vs):
        self.vertices.update(vs)

    def add_face(self, f):
        self.faces.add(f)

    def add_faces(self, fs):
        self.faces.update(fs)

    def add_halfedge(self, h):
        self.halfedges.add(h)

    def add_halfedges(self, hs):
        self.halfedges.update(hs)

    def delete_face(self, f):
        # removes the face from the mesh, together with all of its edges that lie on the border
        self.faces.remove(f)
        # mark edges that will be deleted
        any_deletions = False
        adjacent_halfedges = list(f.halfedge_iter())
        for h in adjacent_halfedges:
            h.face = None
            if h.rev.on_border():
                h['to_delete'] = True
                any_deletions = True
        if not any_deletions:
            # nothing left to do
            return
        # update pre and nex where necessary, remove the edges
        for h in adjacent_halfedges:
            if 'to_delete' in h.attributes:
                if 'to_delete' not in h.pre.attributes:
                    h.pre.nex = h.rev.nex
                    h.rev.nex.pre = h.pre
                    h.orig.any_outgoing = h.pre.rev
                if 'to_delete' not in h.nex.attributes:
                    # TODO?: while 'to_delete' h.nex.pre ...
                    # (this is not necessary, if borders of the tilings are sufficiently far apart..)
                    h.nex.pre = h.rev.pre
                    h.rev.pre.nex = h.nex
                    h.dest.any_outgoing = h.nex
                else:
                    # remove vertex surrounded by border
                    self.vertices.remove(h.dest)
                self.halfedges.difference_update({h, h.rev})

    def delete_edge(self, h):
        if h.on_border():
            raise NotImplementedError
        if h.face is h.rev.face:
            assert h.orig.order() == 1 or h.dest.order() == 1, \
                f"removal of {h} might lead to {h.face} not being simply connected."
        else:
            self.faces.remove(h.face)
        self.halfedges.difference_update({h, h.rev})
        f = h.rev.face
        f.any_side = h.rev.nex

        # assign new, bigger face to edges adjacent to deleted face
        for k in h.face.halfedge_iter():
            k.face = f

        # update nex and pre and any_outgoing where necessary
        for k in [h, h.rev]:
            k.pre.nex = k.rev.nex
            k.nex.pre = k.rev.pre
            k.dest.any_outgoing = k.nex

        # TODO  check if there are any dangling edges next to h after removal. remove them.
        # for k in [h, h.rev]:
        #     if k.orig.order() == 1 and k.orig.any_outgoing in self.halfedges:
        #         self.delete_edge(k.orig.any_outgoing)

    def join_vertex(self, v):
        assert v.order() == 2, f'Can only join vertices of order 2, ({v}, {v.order()})'
        v_out = v.any_outgoing
        v_out.rev.nex.rev = v_out
        v_out.rev = v_out.rev.nex
        v_out.check_consistency()
        v_out.rev.check_consistency()

        for h in [v_out, v_out.rev]:
            self.halfedges.remove(h.pre)
            h.orig = h.pre.orig
            h.orig.any_outgoing = h
            h.pre = h.pre.pre
            h.pre.nex = h
            if not h.on_border():
                h.face.any_side = h
        self.vertices.remove(v)

    def subdivide_edge(self, h, v=None):
        v = Vertex if v is None else v
        raise NotImplementedError

    def to_networkx_undirected(self):
        result = nx.Graph()
        result.add_edges_from([(h.orig, h.dest) for h in self.halfedges])
        return result

    def get_any_border(self):
        if self._any_border is not None and self._any_border.on_border() and self._any_border in self.halfedges:
            return self._any_border
        for h in self.halfedges:
            if h.on_border():
                self._any_border = h
                return h
        raise LookupError('No border found.')

    def border_edge_iter(self):
        initial = self.get_any_border()
        current = initial
        while True:
            yield current
            current = current.nex
            if current is initial:
                break

    def border_edges(self):
        return list(self.border_edge_iter())

    def border_vertex_iter(self):
        for h in self.border_edge_iter():
            yield h.orig

    def border_vertices(self):
        return list(self.border_vertex_iter())

    def glue_v2v(self, v1=None, v2=None, v1_out=None, v2_out=None):
        # check if both vertices are suited for gluing
        if v1 is None:
            assert v1_out is not None, f'v1 or v1_out must be specified.'
            v1 = v1_out.orig
        if v2 is None:
            assert v2_out is not None, f'v2 or v2_out must be specified.'
            v2 = v2_out.orig

        # search for border edges if not specified
        v1_out = v1.get_outgoing_border() if v1_out is None else v1_out
        v2_out = v2.get_outgoing_border() if v2_out is None else v2_out
        assert (v1_out.orig is v1) and (v2_out.orig is v2), f'({repr(v1_out)}, {v1}), ({repr(v2_out)}, {v2})'
        # get incoming border edges
        v1_in, v2_in = v1_out.pre, v2_out.pre
        #assert v1_out.on_border() and v2_out.on_border() and v1_in.on_border() and v2_in.on_border()

        # assign new vertex, remove old
        v = v1.combine_with(v2)
        for h in chain(v1.outgoing_iter(), v2.outgoing_iter()):
            h.orig = v
            h.rev.dest = v
        self.vertices.difference_update({v1, v2})
        self.vertices.add(v1)

        # do the shuffle
        v1_out.pre = v2_in
        v2_out.pre = v1_in
        v1_in.nex = v2_out
        v2_in.nex = v1_out

    def glue_e2e(self, e1, e2):
        if all(a.orig is not b.dest for a, b in [(e1, e2), (e2, e1)]):
            for e in (e1, e2):
                if not e.on_border():
                    raise ValueError(f'Cannot glue: Edge {e} not on border.')
        else:
            # handle face stuff now
            if e1.face is e2.face is None:
                pass
            elif e1.nex is e2 and e2.nex is e1:
                self.faces.remove(e1.face)
            elif e1.nex is e2:
                e1.face.any_side = e2.nex
            elif e2.nex is e1:
                e1.face.any_side = e1.nex
            else:
                raise ValueError(f'Cannot glue: Edges {[e1, e2]} of face {e1.face} are not adjacent.')

        # glue vertices
        for v1_out, v2_out in ((e1, e2.nex), (e1.nex, e2)):
            if v1_out.orig != v2_out.orig:
                self.glue_v2v(v1_out=v1_out, v2_out=v2_out)

        # eliminate double edge
        e1.rev.rev = e2.rev
        e2.rev.rev = e1.rev

        e1.orig.any_outgoing = e2.rev#e1.rev.nex
        e1.dest.any_outgoing = e1.rev#e1.rev.pre.rev
        self.halfedges.difference_update({e1, e2})

    def glue_graph_e2e(self, graph, e1, e2):
        self.add_graph(graph)
        self.glue_e2e(e1, e2)

    def close_vertex(self, v):
        # get edges to be glued
        e1 = v.get_outgoing_border()
        e2 = e1.pre
        self.glue_e2e(e1, e2)

    def execute_edge_instruction(self, h, instruction=None, key=None):
        if instruction is None:
            key = 'instruction' if key is None else key
            instruction = h[key]
        else:
            assert key is None, f'Please specify not more than one of [key, instruction].'
        instruction(self, h)

    def execute_all_edge_instructions(self, instruction=None, key=None):
        for h in self.border_edges():
            self.execute_edge_instruction(h, instruction, key)

    def show_spring_layout(self, figsize=(15, 15), emph_func=None):

        G = nx.Graph()
        G.add_edges_from([(h.orig, h.dest) for h in self.halfedges])

        if emph_func is None:
            emph_func = lambda h: h.attributes.get('delete', False)
        G.add_edges_from([(h.orig, h.dest) for h in self.halfedges if emph_func(h)],
                              color='r')

        pos = nx.spring_layout(G.to_undirected())
        plt.figure(figsize=figsize)


        colors = [G[u][v].get('color', 'b') for u,v in G.edges]
        nx.draw_networkx_nodes(G, pos, node_size=500)
        nx.draw_networkx_edges(G, pos, edge_color=colors, arrows=True)

        nx.draw_networkx_labels(G, pos)
        plt.show()

    def check_consistency(self):
        # check local consistency
        for e in self.halfedges:
            e.check_consistency()
        for f in self.faces:
            f.check_consistency()
        for v in self.vertices:
            v.check_consistency()

        # global checks
        referenced_halfedges = set()
        referenced_halfedges.update({h.nex for h in self.halfedges})
        referenced_halfedges.update({h.pre for h in self.halfedges})
        referenced_halfedges.update({h.rev for h in self.halfedges})
        referenced_halfedges.update({v.any_outgoing for v in self.vertices})
        referenced_halfedges.update({f.any_side for f in self.faces})

        referenced_vertices = set()
        referenced_vertices.update({h.orig for h in self.halfedges})
        referenced_vertices.update({h.dest for h in self.halfedges})

        referenced_faces = {h.face for h in self.halfedges}
        referenced_faces.discard(None)

        if (
                referenced_vertices != self.vertices or
                referenced_faces != self.faces or
                referenced_halfedges != self.halfedges
        ):
            print('='*50)
            print('consistency error')

            print(f'halfedges: {referenced_halfedges.difference(self.halfedges)}, '
                  f'{self.halfedges.difference(referenced_halfedges)}')
            f'vertices: {referenced_vertices.difference(self.vertices)}, {self.vertices.difference(referenced_vertices)}'
            f'faces: {referenced_faces.difference(self.faces)}, {self.faces.difference(referenced_faces)}'

            reference_dict = {obj: set() for obj in
                              referenced_halfedges.union(referenced_vertices).union(referenced_faces).union([None])}
            for h in self.halfedges:
                for attr in ['nex', 'pre', 'rev', 'orig', 'dest', 'face']:
                    reference_dict[getattr(h, attr)].add((h, attr))
            for v in self.vertices:
                for attr in ['any_outgoing']:
                    reference_dict[getattr(v, attr)].add((v, attr))
            for f in self.faces:
                for attr in ['any_side']:
                    reference_dict[getattr(f, attr)].add((f, attr))
            for obj in (referenced_vertices.difference(self.vertices)
                        .union(referenced_faces.difference(self.faces))
                        .union(referenced_halfedges.difference(self.halfedges))):
                print(f'{obj} referenced by {reference_dict[obj]}.')
            assert False


# ------------------------------------------------ faces with in-angles ------------------------------------------------

# class AngledHalfEdge(HalfEdge):
#     def __init__(self, in_angle=None, *super_args, **super_kwargs):
#         super(AngledHalfEdge, self).__init__(*super_args, **super_kwargs)
#         self['in_angle'] = in_angle


class InAngleHEG(HalfEdgeGraph):
    # class for HalfEdgeGraphs with in-angles.
    # the angle between e and e.nex is stored in e['in_angle'].
    # whenever e.face is not None, it should have the 'in_angle' attribute.

    def __init__(self, angle_sum=None, eps=None, other=None):
        super(InAngleHEG, self).__init__(other=other)
        self.tau = 2*pi
        self.eps = 1e-6
        if other is not None:
            self.tau = other.tau if hasattr(other, 'tau') else self.tau
            self.eps = other.eps if hasattr(other, 'eps') else self.eps
        # tau = 2*pi, the sum of angles
        self.tau = angle_sum if angle_sum is not None else self.tau
        # tolerance for deciding weather angles are equal
        self.eps = eps if eps is not None else self.eps

    def is_tau(self, angle):
        return abs(angle - self.tau) < self.eps

    def delete_edge(self, h):
        super(InAngleHEG, self).delete_edge(h)
        for k in [h, h.rev]:
            k.pre['in_angle'] += k.rev['in_angle']

    def close_vertex(self, v, reverse=False):
        # reverse specifies which vertex will be kept.
        e = v.get_outgoing_border()
        edges = (e, e.pre)
        if reverse:
            edges = edges[::-1]
        super(InAngleHEG, self).glue_e2e(*edges)
        # return the new vertex
        return e.rev.orig

    def autoclose_vertex(self, v, reverse=False, recursive=True):
        anglesum = v.angle_sum()
        if self.is_tau(anglesum):
            v_next = self.close_vertex(v, reverse=reverse)
            if recursive:
                self.autoclose_vertex(v_next, reverse=reverse, recursive=True)
        elif anglesum > self.tau:
           print(f'Vertex {v} has anglesum of {anglesum} > {self.tau}')
           # assert False, f'Vertex {v} has anglesum of {anglesum} > {self.tau}'

    def glue_e2e(self, e1, e2, auto_close=True, auto_close_recursive=True):
        # glue the edges
        super(InAngleHEG, self).glue_e2e(e1, e2)
        if not auto_close:
            # if the vertices should not be closed, that is it
            return
        if e1.rev.orig.on_border():  # should always be the case
            # TODO: think about reverse properly...
            self.autoclose_vertex(e1.rev.orig, reverse=False, recursive=auto_close_recursive)
        else:
            assert False
        if e1.rev.dest.on_border():  # this might not be the case, if everything is closed by the previous operation
            self.autoclose_vertex(e1.rev.dest, reverse=True, recursive=auto_close_recursive)


class EuclideanPositionHEG(InAngleHEG):
    def __init__(self, **super_kwargs):
        super(EuclideanPositionHEG, self).__init__(**super_kwargs)

    def positions_coincide(self, p1, p2):
        return np.linalg.norm(p1 -p2) < self.eps

    def lengths_equal(self, l1, l2):
        return abs(l1 - l2) <= self.eps

    def glue_graph_e2e(self, graph, e1, e2):
        # clear positions of new graph, they have to be recalculated (from angles)
        if isinstance(graph, EuclideanPositionHEG):
            for v in graph.vertices:
                del v['pos']
        assert self.lengths_equal(e1.rev['length'], e2.rev['length'])
        super(EuclideanPositionHEG, self).glue_graph_e2e(graph, e1, e2)

        # compute positions for new vertices, face by face
        e = e1 if e1 in graph.halfedges else e2
        if e.rev.on_border():
            assert False, f'{e}'
        faces_to_process = {(e.rev.face, e.rev.nex)}
        processed_faces = set()
        while faces_to_process:
            f, initial = faces_to_process.pop()
            processed_faces.add(f)
            e = initial
            while True:
                assert 'pos' in e.orig.attributes, f'{e}'
                assert 'pos' in e.pre.orig.attributes, f'{e},{e.pre}'
                if 'pos' in e.dest.attributes:
                    # nothing to do here, both positions already determined
                    pass
                else:
                    length = e['length']
                    next_angle = angle_to_axis(e.orig['pos'] - e.pre.orig['pos']) + np.pi - e.pre['in_angle']
                    dir = unit_vector(next_angle)
                    next_pos = e.orig['pos'] + length * dir
                    e.dest['pos'] = next_pos
                opposite_face = e.rev.face
                if opposite_face in graph.faces and opposite_face not in processed_faces:
                    faces_to_process.add((opposite_face, e.rev.nex))
                e = e.nex
                if e is initial:
                    break

    def recompute_lengths_and_angles(self):
        for f in self.faces:
            f.recompute_lengths_and_angles()

    def show(self, render_faces=True, render_edges=True, render_vertices=True, **kwargs):
        from .redering import CairoRenderer
        import matplotlib.image as mpimg
        renderer = CairoRenderer(**kwargs)
        surface = renderer.render_graph(self, render_faces=render_faces, render_edges=render_edges, render_vertices=render_vertices)
        filename = 'output.png'
        surface.write_to_png(filename)
        # surface.finish()
        img = mpimg.imread(filename)
        plt.figure(figsize=(13, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


# ------------------------------------------------ cyclic graph example ------------------------------------------------

def rotate_by(list_like, offset):
    if isinstance(offset, int):
        l = list(list_like)
        return l[offset:] + l[:offset]
    else:
        return zip(*(rotate_by(list_like, off) for off in offset))


def any_element(s):
    return next(iter(s))


class CyclicHalfedgeGraph(HalfEdgeGraph):
    def __init__(self, vs, inner_hs=None, outer_hs=None, f=None):
        super(CyclicHalfedgeGraph, self).__init__()

        # init face if necessary
        f = Face() if f is None else f
        assert isinstance(f, Face), f"{type(f)}"

        # init the halfedges, first only with the vertices
        inner_hs = [HalfEdge() for _ in vs] if inner_hs is None else inner_hs
        outer_hs = [HalfEdge() for _ in vs] if outer_hs is None else outer_hs

        # give face reference to a halfedge
        f.any_side = inner_hs[0]

        # orig and dest for inner and outer halfedges
        for h, (orig, dest) in zip(inner_hs, rotate_by(vs, (0, 1))):
            h.orig = orig
            h.dest = dest
        for h, (dest, orig) in zip(outer_hs, rotate_by(vs, (0, 1))):
            h.orig = orig
            h.dest = dest

        # nex and pre and face for inner halfedges
        for h, pre, nex in rotate_by(inner_hs, (0, -1, 1)):
            h.nex = nex
            h.pre = pre
            h.face = f
            h.orig.any_outgoing = h

        # nex and pre for outer halfedges
        for h, pre, nex in rotate_by(outer_hs, (0, 1, -1)):  # note different offsets
            h.nex = nex
            h.pre = pre

        # rev for all halfedges
        for h_inner, h_outer in zip(inner_hs, outer_hs):
            h_inner.rev = h_outer
            h_outer.rev = h_inner

        # finally, add everything to self
        self.add_vertices(vs)
        self.add_face(f)
        self.face = f
        self.add_halfedges(inner_hs)
        self.add_halfedges(outer_hs)
        

class RegularNGon(CyclicHalfedgeGraph, InAngleHEG):
    def __init__(self, n, *super_args, **super_kwargs):
        super(RegularNGon, self).__init__(vs=[Vertex() for _ in range(n)], *super_args, **super_kwargs)
        f = any_element(self.faces)
        for e in f.halfedge_iter():
            e['in_angle'] = (n-2) / n * pi 
