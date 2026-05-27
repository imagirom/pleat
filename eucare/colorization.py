"""Resolve per-element colours from ``color_key`` attributes and ``color_by`` classifiers.

A *colour key* is any hashable used by renderers to look up a colour. This module:

- assigns hashable colour keys onto graph elements (:func:`colorize`), and
- resolves those keys plus optional ``color_by`` classifiers into concrete
  RGB(A) colours using a matplotlib colormap (:func:`resolve_colors`).
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import matplotlib as mpl
import numpy as np

from .classifiers import (
    Classifier,
    EdgeLengthClassifier,
    EdgeOrientationClassifier,
    VertexOrderClassifier,
    congruency_classifier,
    lambda_classifier,
)

# ---- Default colormaps (edit here to change the project-wide defaults) ----
DEFAULT_FACE_CMAP: str = "tab10"
DEFAULT_EDGE_CMAP: str = "tab10"
DEFAULT_VERTEX_CMAP: str = "tab10"


# ---- Preset registries ----------------------------------------------------

FACE_PRESETS: dict[str, Callable[[], Classifier]] = {
    "congruency": congruency_classifier,
    "order": lambda: lambda_classifier(lambda f: f.order())(),
}

EDGE_PRESETS: dict[str, Callable[[], Classifier]] = {
    "length": EdgeLengthClassifier,
    "orientation": EdgeOrientationClassifier,
}

VERTEX_PRESETS: dict[str, Callable[[], Classifier]] = {
    "order": VertexOrderClassifier,
}


# ---- Colour literal detection --------------------------------------------


def is_color(obj: object) -> bool:
    """Return True if *obj* looks like an RGB(A) tuple/list/array of numbers or a ``#RRGGBB`` string."""
    if isinstance(obj, str):
        if obj.startswith("#") and len(obj) == 7:
            try:
                int(obj[1:], 16)
                return True
            except ValueError:
                return False
        return False
    if isinstance(obj, np.ndarray):
        return obj.ndim == 1 and obj.shape[0] in (3, 4) and np.issubdtype(obj.dtype, np.number)
    if not isinstance(obj, Iterable):
        return False
    seq = list(obj)
    return len(seq) in (3, 4) and all(isinstance(c, (int, float)) for c in seq)


def _to_rgba(color) -> np.ndarray:
    """Normalise a colour literal (tuple, list, ndarray, or '#RRGGBB') to a length-4 RGBA ndarray."""
    if isinstance(color, str):
        arr = np.array([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0])
        return arr
    arr = np.asarray(color, dtype=float)
    if arr.shape[0] == 3:
        return np.concatenate([arr, [1.0]])
    return arr


# ---- Mutating colour assignment (existing helper, unchanged behaviour) ---


def colorize(graph, classifier: Classifier, key: str = "color_key") -> None:
    """Assign ``face[key] = classifier.classify(face)`` for every face in *graph*."""
    for f in graph.faces:
        f[key] = classifier.classify(f)


def congruency_colorize(graph, **kwargs) -> None:
    """Colour faces by polygon congruence (same edge lengths and interior angles)."""
    colorize(graph, congruency_classifier(), **kwargs)


# ---- Palette construction ------------------------------------------------

# Qualitative cmaps store at most this many colours; anything larger (e.g. viridis's
# 256-entry ListedColormap) is treated as a continuous gradient and sampled across
# its full [0, 1] range rather than packed at the start.
_QUALITATIVE_MAX_SIZE: int = 20

# Cyclic colormaps where the 0 and 1 endpoints map to the same colour. Sampled at
# ``i/n`` (open interval) to avoid wasting a slot on the duplicate endpoint.
_CYCLIC_CMAPS: frozenset[str] = frozenset({"hsv", "twilight", "twilight_shifted"})


def _sample_continuous(cm, n: int) -> np.ndarray:
    """Sample a continuous Colormap at *n* points covering its useful range."""
    if n == 1:
        return np.array([cm(0.5)])
    if getattr(cm, "name", "") in _CYCLIC_CMAPS:
        return np.array([cm(i / n) for i in range(n)])
    return np.array([cm(i / (n - 1)) for i in range(n)])


def _make_palette(n: int, cmap: Any) -> np.ndarray:
    """Return an ``(n, 4)`` RGBA array drawn from *cmap*.

    *cmap* may be a matplotlib colormap name, a ``Colormap`` instance, or a list
    of colour literals.

    Behaviour by cmap kind:

    - **List of colours:** cycled in order (``palette[i] = colors[i % len]``).
    - **Qualitative** colormap (≤ :data:`_QUALITATIVE_MAX_SIZE` discrete entries,
      e.g. ``tab10``, ``Set2``): the discrete entries are used directly; if *n*
      exceeds the cmap's size, the palette falls back to evenly-spaced samples
      of ``hsv`` so every class still gets a distinct hue.
    - **Continuous** colormap (``viridis``, ``plasma``, ``hsv``, ...): sampled
      across the full ``[0, 1]`` range so categorical inputs span the whole
      gradient. Cyclic cmaps use the half-open interval to avoid a duplicate
      endpoint.
    """
    if isinstance(cmap, (list, tuple)) or (isinstance(cmap, np.ndarray) and cmap.ndim == 2):
        colors = [_to_rgba(c) for c in cmap]
        return np.array([colors[i % len(colors)] for i in range(n)])
    cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    discrete = getattr(cm, "colors", None)
    if discrete is not None and len(discrete) <= _QUALITATIVE_MAX_SIZE:
        if n <= len(discrete):
            return np.array([_to_rgba(discrete[i]) for i in range(n)])
        cm = mpl.colormaps["hsv"]  # qualitative overflow → distinct hues
    return _sample_continuous(cm, n)


# ---- color_by resolution -------------------------------------------------


def _make_classifier(color_by, presets):
    """Convert a *color_by* spec (str / Classifier / callable) into a (element → hashable) function."""
    if color_by is None:
        return None
    if isinstance(color_by, str):
        if color_by not in presets:
            raise ValueError(f"unknown preset {color_by!r}; available: {sorted(presets)}")
        return presets[color_by]().classify
    if isinstance(color_by, Classifier):
        return color_by.classify
    if callable(color_by):
        return color_by
    raise TypeError(f"color_by must be a preset name, Classifier, or callable; got {type(color_by).__name__}")


def resolve_colors(
    elements,
    color_by,
    cmap,
    presets: dict[str, Callable[[], Classifier]],
    key: str = "color_key",
) -> dict[Any, np.ndarray]:
    """Compute a ``{element → RGBA}`` map for the given *elements*.

    Resolution rules (highest priority first):

    1. If ``element[key]`` is a colour literal (RGB(A) tuple / hex / array), use it as-is.
    2. If ``element[key]`` exists but is a non-colour hashable, treat it as a class index.
    3. Otherwise, if *color_by* is set, run its classifier on the element.
    4. Otherwise, leave the element unassigned (the renderer applies its own fallback).

    Distinct class indices are then sorted by frequency (descending; iteration
    order breaks ties) and assigned palette slots in that order.

    If *color_by* is set but every element already has a ``key`` attribute, a
    :class:`UserWarning` is emitted (the classifier had nothing to do).
    """
    elements = list(elements)
    classify = _make_classifier(color_by, presets)

    literal_colors: dict[Any, np.ndarray] = {}
    class_index: dict[Any, Any] = {}  # element → hashable class id
    class_counts: Counter = Counter()
    class_first_seen: dict[Any, int] = {}
    untouched_by_color_by = 0

    for el in elements:
        existing = el.attributes.get(key)
        if existing is not None and is_color(existing):
            literal_colors[el] = _to_rgba(existing)
            untouched_by_color_by += 1
            continue
        if existing is not None:
            idx = existing
            untouched_by_color_by += 1
        elif classify is not None:
            idx = classify(el)
        else:
            continue
        try:
            hash(idx)
        except TypeError as e:
            raise TypeError(f"colour class index {idx!r} is not hashable") from e
        class_index[el] = idx
        if idx not in class_first_seen:
            class_first_seen[idx] = len(class_first_seen)
        class_counts[idx] += 1

    if classify is not None and elements and untouched_by_color_by == len(elements):
        warnings.warn(
            "color_by was specified, but every element already has a color_key; classifier had no effect.",
            UserWarning,
            stacklevel=2,
        )

    if not class_counts:
        return literal_colors

    sorted_classes = sorted(class_counts, key=lambda i: (-class_counts[i], class_first_seen[i]))
    palette = _make_palette(len(sorted_classes), cmap)
    class_to_color = {idx: palette[rank] for rank, idx in enumerate(sorted_classes)}

    out = dict(literal_colors)
    for el, idx in class_index.items():
        out[el] = class_to_color[idx]
    return out
