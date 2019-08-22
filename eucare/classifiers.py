import numpy as np


class Classifier:
    """
    Class to classify stuff, e.g. faces by their number of sides,
    or by some equivalence relation such as congruency of polygons
    """

    def __init__(self, save_items=False, save_indices=False):
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
        index = self._get_index(item)
        if self.save_items:
            self.saved_items[index] = self.saved_items.get(index, set()).union({item})
        if self.save_indices:
            self.saved_indices[item] = index
        return index


class CountingClassifier(Classifier):
    """transforms a classifier returning any kind of indices to one returning counting natural numbers"""

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
    def __init__(self, *super_args, **super_kwargs):
        super(RepresentationClassifier, self).__init__(*super_args, **super_kwargs)
        self.current_count = 0
        self.count_to_repr = dict()

    def _compare_representations(self, query_rep, saved_rep):
        return query_rep == saved_rep

    def _represent_item(self, item):
        return item

    def _represent_query_item(self, item):
        return self._represent_item(item)

    def _get_index(self, item):
        query_rep = self._represent_query_item(item)
        for index, rep in self.count_to_repr.items():
            if self._compare_representations(query_rep, rep):
                return index
        self.count_to_repr[self.current_count] = self._represent_item(item)
        self.current_count += 1
        return self.current_count - 1


class NestedClassifier(Classifier):
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
            classifier = current_dict['classifier']
            current_index = classifier.classify(item)
            result += (current_index,)
            current_dict = current_dict['index_mapping']
        return result


def lambda_classifier(func):
    class LambdaClassifier(Classifier):
        def _get_index(self, item):
            return func(item)

    return LambdaClassifier


class LenClassifier(Classifier):
    def _get_index(self, item):
        return len(item)


tol = 1e-4


class SumClassifier(RepresentationClassifier):
    def _compare_representations(self, query_rep, saved_rep):
        return np.all(np.abs(query_rep - saved_rep) < tol)

    def _represent_item(self, item):
        return np.sum(np.array(item))


class UnorderedClassifier(RepresentationClassifier):
    def _compare_representations(self, query_rep, saved_rep):
        return np.all(query_rep == saved_rep)

    def _represent_item(self, item):
        return np.sort(np.array(item))


class CyclicClassifier(RepresentationClassifier):
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
            return np.min(np.sum(
                np.abs(saved_rep - query_rep[None]).reshape(len(saved_rep), -1),
                axis=1), axis=0) <= self.tolerance

    def _represent_item(self, item):
        pts = self._represent_query_item(item)
        if not self.allow_flip:
            return np.stack([np.roll(pts, i, axis=0) for i in np.arange(len(pts))])
        else:
            return np.concatenate([
                np.stack([np.roll(pts, i, axis=0) for i in np.arange(len(pts))]),
                np.stack([np.roll(pts[::-1], i, axis=0) for i in np.arange(len(pts))])
            ], axis=0)

    def _represent_query_item(self, item):
        return np.array(item)
