"""Tests for classifier equivalence behaviour."""
from __future__ import annotations

import numpy as np

from eucare.classifiers import (
    CountingClassifier,
    CyclicClassifier,
    LenClassifier,
    NestedClassifier,
    SumClassifier,
    UnorderedClassifier,
    lambda_classifier,
)


def test_len_classifier_groups_by_length():
    c = LenClassifier()
    assert c.classify([1, 2, 3]) == c.classify([4, 5, 6])
    assert c.classify([1, 2]) != c.classify([1, 2, 3])


def test_unordered_classifier_ignores_order():
    c = UnorderedClassifier()
    a = c.classify([1, 2, 3])
    b = c.classify([3, 1, 2])
    different = c.classify([1, 2, 4])
    assert a == b
    assert a != different


def test_sum_classifier_within_tolerance():
    c = SumClassifier()
    a = c.classify([1.0, 2.0, 3.0])
    b = c.classify([2.0, 4.0])  # same sum
    different = c.classify([10.0])
    assert a == b
    assert a != different


def test_cyclic_classifier_groups_rotations():
    c = CyclicClassifier(tolerance=1e-9)
    a = c.classify(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    b = c.classify(np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]))
    different = c.classify(np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    assert a == b
    assert a != different


def test_cyclic_classifier_with_flip_groups_reflections():
    c = CyclicClassifier(tolerance=1e-9, allow_flip=True)
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    a = c.classify(pts)
    b = c.classify(pts[::-1])
    assert a == b


def test_counting_classifier_yields_consecutive_ints():
    c = CountingClassifier(LenClassifier())
    i0 = c.classify([1])
    i1 = c.classify([1, 2])
    i0_again = c.classify([9])
    assert {i0, i1} == {0, 1}
    assert i0_again == i0


def test_lambda_classifier_uses_supplied_func():
    Cls = lambda_classifier(lambda x: x % 2)
    c = Cls()
    assert c.classify(2) == c.classify(4)
    assert c.classify(2) != c.classify(3)


def test_nested_classifier_returns_tuple_index():
    c = NestedClassifier([LenClassifier, UnorderedClassifier])
    a = c.classify([1, 2, 3])
    b = c.classify([3, 2, 1])
    diff_len = c.classify([1, 2])
    diff_set = c.classify([1, 2, 4])
    assert isinstance(a, tuple) and len(a) == 2
    assert a == b
    assert a != diff_len
    assert a != diff_set
