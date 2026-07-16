"""Test for the SVG crease-pattern loader."""

from __future__ import annotations

from pleat.svg import load_svg

# Minimal SVG with three colored line strokes:
#   * red   = mountain
#   * blue  = valley
#   * gray  = ignored
# Plus a fourth red line that closes a square so the graph is connected.
_SVG = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <path d="M 0,0 L 100,0" style="stroke:red;fill:none" />
  <path d="M 100,0 L 100,100" style="stroke:red;fill:none" />
  <path d="M 100,100 L 0,100" style="stroke:red;fill:none" />
  <path d="M 0,100 L 0,0" style="stroke:red;fill:none" />
  <path d="M 0,0 L 100,100" style="stroke:blue;fill:none" />
  <path d="M 100,0 L 0,100" style="stroke:gray;fill:none" />
</svg>
"""


def test_load_svg_basic(tmp_path):
    p = tmp_path / "cp.svg"
    p.write_text(_SVG)
    G = load_svg(str(p))
    G.check_consistency()
    # 4 corners; the gray (ignored) edge contributes no vertex pair.
    assert G.order == 4
    # 4 border edges + 1 diagonal = 5 undirected edges -> 10 halfedges.
    assert len(G.halfedges) == 10
