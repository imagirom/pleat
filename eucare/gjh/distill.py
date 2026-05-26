"""Distill a finite tiled half-edge graph back into a minimal :data:`TilesetSpec`.

The algorithm iteratively refines two equivalence classifiers:

1. Faces are first classified by *congruency* (same shape).
2. Half-edges are then classified by their local context: own class, reverse
   class, neighbors, and the classes of their two incident faces.
3. Faces are re-classified by the cyclic sequence of their edges' classes.

Steps 2-3 repeat until the number of distinct classes stops growing. The
fixpoint yields, for each tile in the spec, an exemplar face plus a labelling
of its outgoing edges.
"""

from __future__ import annotations

from itertools import count

import numpy as np

from ..classifiers import CyclicClassifier, RepresentationClassifier, congruency_classifier
from ..half import EuclideanPositionHEG, Face, HalfEdge
from ..overlap import fast_group_closeby
from ..tileset_spec import TilesetSpec


class _EdgeClassifier(RepresentationClassifier):
    def __init__(self, face_dict: dict[Face, int], edge_dict: dict[HalfEdge, int]) -> None:
        super().__init__()
        self.face_dict = face_dict
        self.edge_dict = edge_dict

    def _compare_representations(self, query_rep, saved_rep) -> bool:
        return bool(np.all(query_rep == saved_rep))

    def _represent_item(self, h: HalfEdge) -> np.ndarray:
        return np.array(
            [
                self.edge_dict.get(h, -1),
                self.edge_dict.get(h.rev, -1),
                self.edge_dict.get(h.nex, -1),
                self.edge_dict.get(h.pre, -1),
                self.face_dict.get(h.face, -1),
                self.face_dict.get(h.rev.face, -1),
            ],
            dtype=np.float32,
        )

    def _get_index(self, item):
        query_rep = self._represent_query_item(item)
        if np.any(query_rep < 0):
            return -1
        if not self.represented_first and self.current_count == 1:
            self.count_to_repr[0] = self._represent_item(self.count_to_repr[0])
            self.represented_first = True
        for index, rep in self.count_to_repr.items():
            if self._compare_representations(query_rep, rep):
                return index
        if self.current_count == 0:
            self.count_to_repr[0] = item
        else:
            self.count_to_repr[self.current_count] = self._represent_item(item)
        self.current_count += 1
        return self.current_count - 1


class _FaceClassifier(CyclicClassifier):
    def __init__(self, edge_dict: dict[HalfEdge, int]) -> None:
        super().__init__(tolerance=0)
        self.edge_dict = edge_dict

    def _represent_query_item(self, f: Face) -> np.ndarray:
        return np.array([self.edge_dict[h] for h in f.halfedge_iter()])

    def _get_index(self, item):
        query_rep = self._represent_query_item(item)
        if np.any(query_rep < 0):
            return -1
        if not self.represented_first and self.current_count == 1:
            self.count_to_repr[0] = self._represent_item(self.count_to_repr[0])
            self.represented_first = True
        for index, rep in self.count_to_repr.items():
            if self._compare_representations(query_rep, rep):
                return index
        if self.current_count == 0:
            self.count_to_repr[0] = item
        else:
            self.count_to_repr[self.current_count] = self._represent_item(item)
        self.current_count += 1
        return self.current_count - 1


def spec_from_graph(G: EuclideanPositionHEG) -> TilesetSpec:
    """Analyse a finite tiled Euclidean graph and produce a :data:`TilesetSpec`."""
    cc = congruency_classifier()
    fs = list(G.faces)
    face_dicts: list[dict[Face, int]] = [{f: cc.classify(f) for f in fs}]

    hs = list(G.halfedges)
    lengths = np.array([np.linalg.norm(h.orig["pos"] - h.dest["pos"]) for h in hs])
    length_groups = fast_group_closeby(lengths[:, None], eps=1e-6)
    edge_dicts: list[dict[HalfEdge, int]] = [{h: int(lg) for h, lg in zip(hs, length_groups)}]

    before = 0
    for _ in count():
        hc = _EdgeClassifier(face_dicts[-1], edge_dicts[-1])
        edge_dicts.append({h: hc.classify(h) for h in hs})

        fc = _FaceClassifier(edge_dicts[-1])
        face_dicts.append({f: fc.classify(f) for f in fs})

        after = len(set(edge_dicts[-1].values())) + len(set(face_dicts[-1].values()))
        if before >= after:
            break
        before = after

    # Pick one exemplar face per class (skipping classes that touched the open border).
    exemplar_fs: list[Face] = []
    for key in set(face_dicts[-2].values()):
        if key < 0:
            continue
        exemplar_fs.append(next(f for f in fs if face_dicts[-2][f] == key and face_dicts[-1][f] >= 0))
    exemplar_fs.sort(key=lambda f: -f.order())

    tile_names = {f: chr(97 + i) for i, f in enumerate(exemplar_fs)}
    edge_names: dict[int, tuple[str, int]] = {}
    for f in exemplar_fs:
        for i, h in enumerate(f.halfedge_iter()):
            edge_class = edge_dicts[-2][h]
            if edge_class not in edge_names:
                edge_names[edge_class] = (tile_names[f], i)

    return {tile_names[f]: [edge_names[edge_dicts[-2][h.rev]] for h in f.halfedge_iter()] for f in exemplar_fs}
