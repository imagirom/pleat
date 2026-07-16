"""Eager-loaded library of cached :data:`TilesetSpec`s indexed by GJH code.

At import time we scan ``pleat/gjh/_data/`` for ``*.yml`` files, parse each
into a :data:`TilesetSpec`, and read the GJH code from the leading
``# GJH: <code>`` comment. :data:`GJH_CODES` preserves the numeric filename
prefix as the canonical ordering (regular → 1-uniform → 2-uniform →
3-uniform), matching the source paper.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Final

from ..tileset_spec import TilesetSpec, spec_from_yaml

_DATA_PKG = "pleat.gjh._data"


def _read_code_and_spec(yaml_text: str) -> tuple[str, TilesetSpec]:
    first_line = yaml_text.splitlines()[0]
    if not first_line.startswith("# GJH:"):
        raise ValueError(f"Cached YAML missing '# GJH: ...' header: {first_line!r}")
    code = first_line[len("# GJH:") :].strip()
    return code, spec_from_yaml(yaml_text)


def _load_all() -> tuple[list[str], dict[str, TilesetSpec]]:
    codes: list[str] = []
    specs: dict[str, TilesetSpec] = {}
    data_root = files(_DATA_PKG)
    yml_files = sorted(
        (p for p in data_root.iterdir() if p.name.endswith(".yml")),
        key=lambda p: p.name,
    )
    for resource in yml_files:
        text = resource.read_text()
        code, spec = _read_code_and_spec(text)
        codes.append(code)
        specs[code] = spec
    return codes, specs


_codes, _specs = _load_all()
GJH_CODES: Final[list[str]] = _codes
CACHED_SPECS: Final[dict[str, TilesetSpec]] = _specs


def cached_spec(code: str) -> TilesetSpec:
    """Return the cached :data:`TilesetSpec` for ``code``. Raises :class:`KeyError` if unknown."""
    code = code.replace(" ", "")
    try:
        return CACHED_SPECS[code]
    except KeyError as e:
        raise KeyError(f"GJH code {code!r} is not in the cached library") from e
