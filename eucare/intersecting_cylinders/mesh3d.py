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

    The surface above each face is decomposed into one curved strip per edge,
    plus, for ``r < 1``, a flat lifted copy of the shrunken inner face.

    Geometry. Each strip is parametrised by ``(s, w) in [0, 1]^2``:

    * ``s`` runs along the edge from ``p1`` to ``p2``;
    * ``w`` runs from the edge (at perpendicular distance 0) to the
      corresponding inner-face point.

    The shrinking is governed by the same ``expand_t`` used by
    :func:`make_intersecting_cylinders` (so the mesh's inner edges line up
    exactly with the inner shrunken face produced by the crease pattern):

        ``expand_t = (1 - r) * sf / (1 - (1 - r) * (1 - sf))``

    where ``sf = profile.shrink_factor``. For ``r = 1`` the strip extends all
    the way to the incenter (``perp_3D in [0, R]``); for ``r < 1`` it ends at
    the inner edge (``perp_3D in [0, expand_t * R]``).

    The 3D height at perpendicular fraction ``p in [0, 1]`` is
    ``height(p) * R`` where ``R = |c - hc|`` is the (pseudo-)inradius.

    Args:
        G: Input tiling whose faces have well-defined (pseudo-)incenters.
            A working copy is made; the input is not modified.
        profile: Cross-section curve.
        r: Triangle scaling matching :func:`make_intersecting_cylinders`.
        n_along_edge: Number of subdivisions along each edge of every strip.

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
    perp_axis = profile.t / sf  # perpendicular fraction in [0, 1]
    height_axis = profile.y / sf  # height (in same units as perp), in [0, ymax]

    # Apex perp fraction: 1 for r=1 (strip reaches incenter), expand_t for r<1.
    if r == 1.0:
        apex_perp = 1.0
    else:
        apex_perp = (1.0 - r) * sf / (1.0 - (1.0 - r) * (1.0 - sf))
    inner_scale = 1.0 - apex_perp  # scale factor of inner face (about c)

    # Sub-sample the profile within [0, apex_perp], inserting an exact endpoint.
    interior = perp_axis[(perp_axis > 0) & (perp_axis < apex_perp - 1e-12)]
    sub_perp = np.concatenate([[0.0], interior, [apex_perp]])
    sub_height = np.interp(sub_perp, perp_axis, height_axis)

    n_perp = len(sub_perp)
    n_along = n_along_edge + 1

    vertices: list[NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []

    for f in G.faces:
        c = f.midpoint()
        if not np.all(np.isfinite(c)):
            continue

        # Compute per-face inradius from the first non-degenerate edge.
        face_R: float | None = None
        for h0 in f.halfedge_iter():
            hc0 = base.project_to_line(np.stack([h0.orig["pos"], h0.dest["pos"]]), c)
            R0 = float(np.linalg.norm(c - hc0))
            if R0 > 0:
                face_R = R0
                break
        if face_R is None:
            continue

        # Curved strips: one per halfedge of the face.
        for h in f.halfedge_iter():
            p1 = h.orig["pos"]
            p2 = h.dest["pos"]
            hc = base.project_to_line(np.stack([p1, p2]), c)
            R = float(np.linalg.norm(c - hc))
            if R == 0.0:
                continue

            p1_inner = inner_scale * p1 + apex_perp * c
            p2_inner = inner_scale * p2 + apex_perp * c

            base_idx = len(vertices)
            for j in range(n_along):
                s = j / (n_along - 1)
                edge_pt = p1 + s * (p2 - p1)
                inner_pt = p1_inner + s * (p2_inner - p1_inner)
                for k in range(n_perp):
                    w = sub_perp[k] / apex_perp  # 0 at edge, 1 at inner_pt
                    xy = (1.0 - w) * edge_pt + w * inner_pt
                    z = sub_height[k] * R
                    vertices.append(np.array([xy[0], xy[1], z]))

            for j in range(n_along - 1):
                for k in range(n_perp - 1):
                    a = base_idx + j * n_perp + k
                    b = a + 1
                    cc = a + n_perp
                    d = cc + 1
                    triangles.append((a, b, d))
                    triangles.append((a, d, cc))

        # Flat lifted inner face for r<1 (fan triangulation from c).
        if r < 1.0:
            z_lift = sub_height[-1] * face_R
            c_idx = len(vertices)
            vertices.append(np.array([c[0], c[1], z_lift]))
            inner_corner_idx: list[int] = []
            for v in f.vertex_iter():
                p_inner = inner_scale * v["pos"] + apex_perp * c
                inner_corner_idx.append(len(vertices))
                vertices.append(np.array([p_inner[0], p_inner[1], z_lift]))
            n_corners = len(inner_corner_idx)
            for i in range(n_corners):
                triangles.append(
                    (
                        c_idx,
                        inner_corner_idx[i],
                        inner_corner_idx[(i + 1) % n_corners],
                    )
                )

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
