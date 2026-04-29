"""Interactive 3D preview of intersecting-cylinders models.

The :func:`to_3d_mesh` function builds a triangle mesh of the folded surface;
:func:`show_3d` returns a :mod:`plotly` figure suitable for interactive display
in Jupyter notebooks. ``plotly`` is an optional dependency installed via the
``intersecting_cylinders`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .. import base
from .profiles import Profile

if TYPE_CHECKING:
    from ..half import EuclideanPositionHEG


def to_3d_mesh(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float = 1.0,
    n_along_edge: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a triangle mesh of the folded intersecting-cylinders surface.

    For every face the surface is decomposed into one curved patch per edge.
    The patch sits at ``z = 0`` along the edge and rises to height
    ``alt_length = |c - hc|`` (the inradius for a face with a true incenter)
    where ``hc`` is the foot of the perpendicular from the incenter ``c`` onto
    the edge. With ``r = 1`` each patch is a curved triangle peaking at
    ``(hc, alt_length)``; with ``0 < r < 1`` the patch is a curved
    quadrilateral whose top edge lies on the lifted shrunken inner face.

    Args:
        G: Input tiling whose faces have well-defined (pseudo-)incenters.
            A working copy is made; the input is not modified.
        profile: Cross-section curve.
        r: Triangle scaling matching :func:`make_intersecting_cylinders`.
        n_along_edge: Number of subdivisions along each edge of every patch.

    Returns:
        ``(vertices, triangles)`` with shapes ``(N, 3)`` and ``(M, 3)`` ready
        to feed into :class:`plotly.graph_objects.Mesh3d` or :mod:`meshio`.
    """
    if not 0.0 < r <= 1.0:
        raise ValueError("r must lie in (0, 1]")

    G = G.copy()
    for f in G.faces:
        f["midpoint"] = f.pseudo_incenter()

    sf = profile.shrink_factor
    l_norm = profile.l / sf  # in [0, 1]
    t_norm = profile.t / sf  # in [0, 1]
    n_profile = len(t_norm)
    n_cols = n_along_edge + 1

    vertices: list[NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []

    for f in G.faces:
        c = f.midpoint()
        for h in f.halfedge_iter():
            p1 = h.orig["pos"]
            p2 = h.dest["pos"]
            hc = base.project_to_line(np.stack([p1, p2]), c)
            alt_length = float(np.linalg.norm(c - hc))
            if alt_length == 0.0:
                continue

            base_idx = len(vertices)
            for j in range(n_cols):
                u = j / (n_cols - 1)
                edge_pt = p1 + u * (p2 - p1)
                dest_xy = hc if r == 1.0 else (1.0 - r) * edge_pt + r * c
                offset = dest_xy - edge_pt
                for k in range(n_profile):
                    xy = edge_pt + offset * l_norm[k]
                    z = alt_length * t_norm[k]
                    vertices.append(np.array([xy[0], xy[1], z]))

            for j in range(n_cols - 1):
                for k in range(n_profile - 1):
                    a = base_idx + j * n_profile + k
                    b = a + 1
                    cc = a + n_profile
                    d = cc + 1
                    triangles.append((a, b, d))
                    triangles.append((a, d, cc))

    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)


def show_3d(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float = 1.0,
    n_along_edge: int = 8,
    color: str = "lightblue",
    opacity: float = 1.0,
    height: int = 600,
) -> "Any":
    """Return an interactive plotly 3D figure of the folded model.

    Requires :mod:`plotly` (``pip install plotly`` or
    ``pip install -e ".[intersecting_cylinders]"``).

    Args:
        G: Input tiling.
        profile: Cross-section curve.
        r: Triangle scaling matching :func:`make_intersecting_cylinders`.
        n_along_edge: Mesh resolution along each edge.
        color: Surface colour.
        opacity: Surface opacity in ``[0, 1]``.
        height: Figure height in pixels.

    Returns:
        Plotly ``Figure`` ready for ``fig.show()`` or rendering in a notebook.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("show_3d requires plotly; install with `pip install plotly`.") from exc

    vertices, triangles = to_3d_mesh(G, profile, r=r, n_along_edge=n_along_edge)

    mesh = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        color=color,
        opacity=opacity,
        flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2, roughness=0.5),
        lightposition=dict(x=100, y=100, z=300),
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
    )
    return fig
