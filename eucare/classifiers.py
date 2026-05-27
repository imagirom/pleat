"""Classifiers for grouping graph elements by equivalence relations.

A :class:`Classifier` maps each item to a hashable index; items with the
same index are deemed equivalent.  Concrete subclasses implement different
equivalences: by length, by multiset, up to cyclic rotation (with optional
reflection), and combinations via :class:`NestedClassifier`.

The headline use case is :func:`congruency_classifier`, which groups faces
of a tiling by polygon congruence (matching edge-length and interior-angle
sequences up to cyclic rotation).  Used by :mod:`eucare.colorization`.
"""

from __future__ import annotations

import numpy as np


class Classifier:
    """Classify items by a hashable index, optionally tracking items and indices per class."""

    def __init__(self, save_items: bool = False, save_indices: bool = False) -> None:
        super(Classifier, self).__init__()

        self.used_indices = set()

        # option to keep track of a dict mapping classes to items
        self.save_items = save_items
        if self.save_items:
            self.saved_items = dict()
        else:
            self.saved_items = None

        # option to keep track of a dict mapping classes to items
        self.save_indices = save_indices
        if self.save_indices:
            self.saved_indices = dict()
        else:
            self.saved_indices = None

    def _get_index(self, item):
        # the returned 'index' can be any hashable
        raise NotImplementedError

    def classify(self, item):
        """Return the equivalence class index for ``item`` and update saved items/indices."""
        index = self._get_index(item)
        if self.save_items:
            self.saved_items[index] = self.saved_items.get(index, set()).union({item})
        if self.save_indices:
            self.saved_indices[item] = index
        return index


class CountingClassifier(Classifier):
    """Wrap a classifier to remap its indices to consecutive natural numbers."""

    def __init__(self, other, *super_args, **super_kwargs):
        super(CountingClassifier, self).__init__(*super_args, **super_kwargs)
        self.non_counting_classifier = other
        self.current_count = 0
        self.index_to_count = dict()

    def _get_index(self, item):
        index = self.non_counting_classifier.classify(item)
        if index not in self.index_to_count:
            self.index_to_count[index] = self.current_count
            self.current_count += 1
        return self.index_to_count[index]


class RepresentationClassifier(Classifier):
    """Classify items by computing a representation and comparing it against known classes."""

    def __init__(self, *super_args, **super_kwargs):
        super(RepresentationClassifier, self).__init__(*super_args, **super_kwargs)
        self.current_count = 0
        self.count_to_repr = dict()
        self.represented_first = False

    def _compare_representations(self, query_rep, saved_rep):
        return query_rep == saved_rep

    def _represent_item(self, item):
        return item

    def _represent_query_item(self, item):
        return self._represent_item(item)

    def _get_index(self, item):
        query_rep = self._represent_query_item(item)
        if (
            not self.represented_first and self.current_count == 1
        ):  # compute representation that was skipped for performance (see below)
            self.count_to_repr[0] = self._represent_item(self.count_to_repr[0])
            self.represented_first = True
        for index, rep in self.count_to_repr.items():
            if self._compare_representations(query_rep, rep):
                return index
        if self.current_count == 0:  # for better performance, calculate representation only at first comparison
            self.count_to_repr[0] = item
        else:
            self.count_to_repr[self.current_count] = self._represent_item(item)
        self.current_count += 1
        return self.current_count - 1


class NestedClassifier(Classifier):
    """Chain multiple classifiers from coarse to fine, producing a tuple index."""

    def __init__(self, coarse_to_fine, *super_args, **super_kwargs):
        # coarse_to_fine should be a list of classifier classes
        super(NestedClassifier, self).__init__(*super_args, **super_kwargs)
        self.coarse_to_fine = coarse_to_fine
        self.nested_classfier_dict = dict()
        self.base_classifier = self.coarse_to_fine[0]()

    def _get_index(self, item):
        current_dict = self.nested_classfier_dict
        current_index = self.base_classifier.classify(item)
        result = (current_index,)
        for cls in self.coarse_to_fine[1:]:
            if current_index not in current_dict:
                current_dict[current_index] = dict(classifier=cls(), index_mapping=dict())
            current_dict = current_dict[current_index]
            classifier = current_dict["classifier"]
            current_index = classifier.classify(item)
            result += (current_index,)
            current_dict = current_dict["index_mapping"]
        return result


def lambda_classifier(func):
    """Create a Classifier class that uses the given function as its index."""

    class LambdaClassifier(Classifier):
        """Classifier whose index is computed by the wrapped function."""

        def _get_index(self, item):
            return func(item)

    return LambdaClassifier


class LenClassifier(Classifier):
    """Classify items by their length."""

    def _get_index(self, item):
        return len(item)


tol = 1e-4


class SumClassifier(RepresentationClassifier):
    """Classify items by the sum of their elements (with tolerance)."""

    def _compare_representations(self, query_rep, saved_rep):
        return np.all(np.abs(query_rep - saved_rep) < tol)

    def _represent_item(self, item):
        return np.sum(np.array(item))


class UnorderedClassifier(RepresentationClassifier):
    """Classify items by their sorted elements, ignoring order."""

    def _compare_representations(self, query_rep, saved_rep):
        return np.all(query_rep == saved_rep)

    def _represent_item(self, item):
        return np.sort(np.array(item))


class CyclicClassifier(RepresentationClassifier):
    """Classify items up to cyclic permutation (and optionally reflection)."""

    def __init__(self, tolerance=tol, allow_flip=False, *super_args, **super_kwargs):
        super(CyclicClassifier, self).__init__(*super_args, **super_kwargs)
        self.tolerance = tolerance
        self.allow_flip = allow_flip

    def _compare_representations(self, query_rep, saved_rep):
        if query_rep.shape != saved_rep.shape[1:]:
            return False
        if self.tolerance == 0:
            return np.max(np.min((saved_rep == query_rep).reshape(len(saved_rep), -1), axis=1))
        else:
            return (
                np.min(np.sum(np.abs(saved_rep - query_rep[None]).reshape(len(saved_rep), -1), axis=1), axis=0)
                <= self.tolerance
            )

    def _represent_item(self, item):
        pts = self._represent_query_item(item)
        if not self.allow_flip:
            return np.stack([np.roll(pts, i, axis=0) for i in np.arange(len(pts))])
        else:
            return np.concatenate(
                [
                    np.stack([np.roll(pts, i, axis=0) for i in np.arange(len(pts))]),
                    np.stack([np.roll(pts[::-1], i, axis=0) for i in np.arange(len(pts))]),
                ],
                axis=0,
            )

    def _represent_query_item(self, item):
        return np.array(item)


class PreMapClassifier(Classifier):
    """Apply a function to each item before passing it to another classifier."""

    def __init__(self, other, func, *super_args, **super_kwargs):
        super(PreMapClassifier, self).__init__(*super_args, **super_kwargs)
        self.func = func
        self.other = other

    def _get_index(self, item):
        return self.other.classify(self.func(item))


def _face_to_array(f) -> np.ndarray:
    """Represent a face as an (n, 2) array of (length, in_angle) pairs along its boundary."""
    data = []
    for e in f.halfedge_iter():
        data.append((e["length"], e["in_angle"]))
        # data.append(np.array(e.orig['pos'], dtype=np.float32))
    data = np.stack(data)
    # data -= np.mean(data, axis=0, keepdims=True)
    # print(data)
    return data


def congruency_classifier(allow_flip=False):
    """Return a classifier that groups faces by polygon congruence (edge lengths and angles)."""
    return CountingClassifier(
        PreMapClassifier(
            NestedClassifier([LenClassifier, SumClassifier, lambda: CyclicClassifier(allow_flip=allow_flip)]),
            _face_to_array,
        )
    )


class AdjacencyClassifier(CyclicClassifier):
    """Classify faces by the cyclic sequence of a given attribute on their neighbors."""

    def __init__(self, key, *super_args, **super_kwargs):
        super(AdjacencyClassifier, self).__init__(tolerance=0, *super_args, **super_kwargs)
        self.key = key

    def _represent_query_item(self, item):
        return np.array([(f[self.key] if f is not None else None, item[self.key]) for f in item.face_iter()])


class EdgeLengthClassifier(RepresentationClassifier):
    """Classify half-edges by their ``length`` attribute (with tolerance ``tol``)."""

    def _compare_representations(self, query_rep, saved_rep):
        return abs(query_rep - saved_rep) < tol

    def _represent_item(self, item):
        return float(item["length"])


class EdgeOrientationClassifier(RepresentationClassifier):
    """Classify half-edges by orientation mod π, so an edge and its reverse share a class."""

    def _compare_representations(self, query_rep, saved_rep):
        d = abs(query_rep - saved_rep)
        # Circular distance on [0, π): both ends of the interval are the same angle.
        return min(d, np.pi - d) < tol

    def _represent_item(self, item):
        v = item.dest["pos"] - item.orig["pos"]
        return float(np.arctan2(v[1], v[0]) % np.pi)


class VertexOrderClassifier(Classifier):
    """Classify vertices by their degree (number of incident edges)."""

    def _get_index(self, item):
        return item.order()
