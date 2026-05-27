"""Declarative tileset format: a dict mapping tile names to per-edge gluing references.

A :data:`TilesetSpec` is a ``dict[str, list[tuple[str, int]]]``. Each key is a
tile name (one or more lowercase letters by convention). Each list entry
describes one edge of that tile in cyclic order: the tuple ``(neighbor, j)``
means "this edge is glued to edge ``j`` of tile ``neighbor``".

Example::

    spec = {
        "a": [("b", 1), ("b", 1), ("b", 1)],         # triangle, 3 edges
        "b": [("a", 0)] * 6,                          # hexagon, 6 edges
    }

The spec currently assumes regular Euclidean polygons with unit edge length.
A future extension could add per-edge ``length`` / ``angle`` fields to cover
non-regular or non-Euclidean tilings, but full flexibility will always remain
the domain of hand-built :class:`~eucare.prototiles.ProtoTile` constructions.

On disk the format uses dotted strings (``"b.1"``); on read the older compact
form (``"b1"``) is also accepted for backwards compatibility with the original
notebook output.
"""

from __future__ import annotations

import re

import yaml

from .prototiles import RegularEuclideanTile

#: A declarative tileset description. See module docstring for the format.
TilesetSpec = dict[str, list[tuple[str, int]]]

_EDGE_REF_RE = re.compile(r"^([a-zA-Z]+)\.?(\d+)$")


def parse_edge_ref(s: str) -> tuple[str, int]:
    """Parse a YAML edge reference like ``"b.1"`` or ``"b1"`` into ``("b", 1)``."""
    match = _EDGE_REF_RE.match(s)
    if not match:
        raise ValueError(f"Cannot parse edge reference {s!r}; expected '<name>.<index>' or '<name><index>'")
    return match.group(1), int(match.group(2))


def _format_edge_ref(ref: tuple[str, int]) -> str:
    """Render an edge reference as the canonical dotted YAML string."""
    return f"{ref[0]}.{ref[1]}"


def spec_from_yaml(text: str) -> TilesetSpec:
    """Parse YAML text into a :data:`TilesetSpec`. Accepts both dotted and compact forms."""
    raw = yaml.safe_load(text)
    return {name: [parse_edge_ref(s) for s in edges] for name, edges in raw.items()}


def spec_to_yaml(spec: TilesetSpec, header: str | None = None) -> str:
    """Render a :data:`TilesetSpec` as YAML text in canonical dotted form.

    Args:
        spec: The spec to serialise.
        header: Optional comment line (e.g. ``"GJH: 6-3-3/m_/r(h3)"``) prepended to the file.
    """
    plain = {name: [_format_edge_ref(ref) for ref in edges] for name, edges in spec.items()}
    body = yaml.safe_dump(plain, default_flow_style=None, sort_keys=False)
    if header is not None:
        return f"# {header}\n{body}"
    return body


def tileset_from_spec(spec: TilesetSpec) -> list[RegularEuclideanTile]:
    """Build a list of :class:`RegularEuclideanTile` with mutual edge instructions from a spec.

    The returned list is sorted by decreasing polygon order (hexagons before
    triangles, etc.) to match the convention used by the rest of the package
    when picking a base tile for :func:`~eucare.example_graphs.from_tiles`.
    """
    # First pass: create the tiles with structured edge labels (name, index).
    tile_dict: dict[str, RegularEuclideanTile] = {}
    for name, edges in spec.items():
        n = len(edges)
        tile_dict[name] = RegularEuclideanTile(
            n=n,
            edge_labels=[(name, i) for i in range(n)],
            face_label=name,
        )

    # Second pass: wire up mutual gluing instructions.
    for name, edges in spec.items():
        tile = tile_dict[name]
        for own_label, (other_name, other_idx) in zip(tile.edge_labels, edges):
            other_tile = tile_dict[other_name]
            tile.edge_instructions[own_label] = other_tile.attach_instruction((other_name, other_idx))

    return sorted(tile_dict.values(), key=lambda t: -t.order)
