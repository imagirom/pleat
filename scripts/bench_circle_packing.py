"""Benchmark pack_euclidean and pack_hyperbolic on representative cases.

Includes hyperbolic Platonic tilings of varying schlafli signatures and ring
counts, which tend to produce very uneven radii in the maximal (boundary
horocycle) regime and stress the Collins-Stephenson iteration.

Run with:

    python scripts/bench_circle_packing.py

Optionally pass a label as argv[1]; it is included in the printed table for
easy before/after comparison.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from eucare.circle_packing import pack_euclidean, pack_hyperbolic
from eucare.conway import kis_graph
from eucare.example_graphs import from_tiles
from eucare.example_tilesets import curved_platonic, platonic


@dataclass
class Case:
    name: str
    build: Callable[[], object]
    pack: Callable[[object], object]
    n_runs: int = 3


def _build_hex(rings: int):
    return from_tiles(platonic(3), rings=rings)


def _build_hyperbolic_kis(n: int, k: int, rings: int):
    G = from_tiles(curved_platonic(n, k), rings=rings)
    # Triangulate by inserting one vertex per face.
    return kis_graph()(G)


def cases() -> list[Case]:
    out: list[Case] = []
    # Euclidean hex triangulation (small / medium / large).
    for rings, runs in [(3, 5), (5, 3), (7, 2)]:
        out.append(
            Case(
                f"euclidean hex r={rings}",
                lambda r=rings: _build_hex(r),
                lambda G: pack_euclidean(G, boundary_radii=1.0),
                runs,
            )
        )
    # Hyperbolic platonic tilings, kised to triangulate, maximal packing.
    for (n, k), rings, runs in [
        ((7, 3), 2, 5),
        ((7, 3), 3, 3),
        ((5, 4), 2, 5),
        ((5, 4), 3, 2),
        ((8, 3), 2, 3),
        ((6, 4), 2, 3),
    ]:
        out.append(
            Case(
                f"hyperbolic {{{n},{k}}} kis r={rings} maximal",
                lambda n=n, k=k, r=rings: _build_hyperbolic_kis(n, k, r),
                lambda G: pack_hyperbolic(G, boundary_x_radii=1.0),
                runs,
            )
        )
    # One non-maximal hyperbolic to compare regime.
    out.append(
        Case(
            "hyperbolic {7,3} kis r=2 x=0.5",
            lambda: _build_hyperbolic_kis(7, 3, 2),
            lambda G: pack_hyperbolic(G, boundary_x_radii=0.5),
            5,
        )
    )
    return out


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else ""

    rows: list[tuple[str, int, float, float]] = []
    for c in cases():
        # Build once; reuse the same graph (copy_graph=True default makes the pack copy).
        G = c.build()
        nverts = len(list(G.vertices))
        times: list[float] = []
        # warm-up
        c.pack(G)
        for _ in range(c.n_runs):
            t0 = time.perf_counter()
            c.pack(G)
            times.append(time.perf_counter() - t0)
        rows.append((c.name, nverts, float(np.median(times)), float(np.min(times))))

    width = max(len(r[0]) for r in rows)
    header = f"{'case'.ljust(width)}  {'V':>5}  {'median (s)':>11}  {'best (s)':>9}"
    print(header)
    print("-" * len(header))
    for name, nv, med, best in rows:
        print(f"{name.ljust(width)}  {nv:>5}  {med:>11.4f}  {best:>9.4f}")
    if label:
        print(f"\nlabel: {label}")


if __name__ == "__main__":
    main()
