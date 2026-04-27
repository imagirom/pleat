"""Module for loading SVG files of crease patterns as half-edge graphs."""
import logging
import operator
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import svgpathtools as spt

import eucare as ec
from eucare.overlap import CREASE_ASSIGNMENT, MOUNTAIN, VALLEY

logger = logging.getLogger(__name__)


def load_svg(filepath: str) -> ec.half.EuclideanPositionHEG:

    def get_stroke(attrs):
        # hits = re.search('stroke:(#.{6})', attrs['style'])
        hits = re.search('stroke:(.*?);', attrs['style'])
        if hits is not None:
            hits = hits.groups()
        if not hits:
            return None
        elif len(hits) == 1:
            return hits[0]
        else:
            raise ValueError(f'Found multiple strokes: {list(hits)}')

    paths, attributes = spt.svg2paths(filepath, convert_rectangles_to_paths=False)

    counts_by_stroke = defaultdict(int)
    counts_by_style = defaultdict(int)

    # step 1:

    points = []
    edges = []

    for path, attrs in zip(paths, attributes):
        for line in path:
            assert isinstance(line, spt.path.Line), f'{type(line)}'
            # line = path[0]
            start = np.array([line.start.real, line.start.imag], dtype=np.float32)
            end = np.array([line.end.real, line.end.imag], dtype=np.float32)
            # print(start, end)
            crease_type = None
            # this works for cps exported from oripa
            if 'style' in attrs:
                if 'red' in attrs['style']:
                    crease_type = MOUNTAIN
                elif 'blue' in attrs['style']:
                    crease_type = VALLEY
                elif 'gray' in attrs['style']:
                    continue
            edge_attrs = dict() if crease_type is None else {CREASE_ASSIGNMENT: crease_type}

            # this is e.g. for robert langs cps
            if len(path) == 1:
                if 'style' in attrs:
                    stroke = get_stroke(attrs)
                    counts_by_stroke[stroke] += 1
                    counts_by_style[attrs['style']] += 1
                    edge_attrs['svg_stroke'] = stroke
                elif 'stroke' in attrs:
                    counts_by_stroke[attrs['stroke']] += 1
                    edge_attrs['svg_stroke'] = attrs['stroke']

            edges.append((len(points), len(points) + 1, edge_attrs))
            points.extend([start, end])
    points = np.stack(points)
    points -= np.mean(points, axis=0)
    points /= 2 * np.max(np.abs(points))
    points += [[0.5, 0.5]]

    clustering = ec.overlap.group_closeby(points, 1e-5)
    first_occurences = np.argmax(clustering[None] == np.arange(np.max(clustering) + 1)[:, None], axis=1)
    merged_points = points[first_occurences]

    G = nx.Graph()
    G.add_edges_from([(tuple(merged_points[clustering[i]]), tuple(merged_points[clustering[j]]), attrs)
                      for i, j, attrs in edges])

    G = ec.conversions.EHEG_from_nx(G)

    if len(counts_by_stroke) not in (2, 3):
        logger.warning('encountered %d different stroke colors, not assigning valleys and mountains', len(counts_by_stroke))
        pass
    else:
        if len(counts_by_stroke) == 3:  # assume one is the border stroke
            on_border_counts = {key: 0 for key in list(counts_by_stroke.keys()) + [None]}
            for e in G.border_edges():
                on_border_counts[e.attributes.get('svg_stroke', None)] += 1
                on_border_counts[e.rev.attributes.get('svg_stroke', None)] += 1
            del on_border_counts[None]
            border_stroke = max(on_border_counts.items(), key=operator.itemgetter(1))[0]
            del on_border_counts[border_stroke]
            crease_strokes = sorted(on_border_counts)
        else:  # 2 strokes
            crease_strokes = sorted(counts_by_stroke)
        for e in G.halfedges:
            stroke = e.attributes.get('svg_stroke', None)
            if stroke in crease_strokes:
                e[CREASE_ASSIGNMENT] = MOUNTAIN if stroke == crease_strokes[0] else VALLEY

    from eucare.overlap import color_creases

    color_creases(G)

    return G
