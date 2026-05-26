"""Convert the legacy `b1` GJH spec form to the canonical `b.1` form.

One-shot script used to seed `eucare/gjh/_data/` from
`outputs/archimedian/*.yml`. Kept committed for reproducibility.

Usage::

    python scripts/convert_gjh_yaml_format.py \\
        outputs/archimedian eucare/gjh/_data
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Match a single edge ref of the legacy form: letters followed by digits, no dot.
_LEGACY_RE = re.compile(r"\b([a-zA-Z]+)(\d+)\b")


def convert_text(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        out_lines.append(_LEGACY_RE.sub(r"\1.\2", line))
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: convert_gjh_yaml_format.py SRC_DIR DST_DIR", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.yml"))
    if not files:
        print(f"No .yml files in {src}", file=sys.stderr)
        sys.exit(1)

    for path in files:
        converted = convert_text(path.read_text())
        (dst / path.name).write_text(converted)
        print(f"  {path.name} -> {dst / path.name}")
    print(f"Converted {len(files)} files.")


if __name__ == "__main__":
    main()
