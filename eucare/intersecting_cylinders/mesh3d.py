"""Interactive 3D preview of intersecting-cylinders models.

The :func:`to_3d_mesh` function builds a triangle mesh of the folded surface;
:func:`show_3d` returns a :mod:`plotly` figure suitable for interactive display
in Jupyter notebooks. ``plotly`` is an optional dependency installed via the
``intersecting_cylinders`` extra.

Geometry
--------

The 3D model is built on top of the **ortho** Conway operator applied to the
input tiling, with edge-midpoint vertices repositioned to the points where the
two dual circle packings touch (the same construction used in
``show_dual_circle_packings`` in the docs notebook). Each ortho quad has cyclic
corners ``(v, t1, c, t2)``:

* ``v`` is an original tiling vertex (a *spike apex* in 3D),
* ``c`` is the incenter of an adjacent face (a *flat base* point in 3D),
* ``t1, t2`` are the points where the original edges incident to ``v`` are
  tangent to both the face incircle (around ``c``) and the *vertex circle* of
  ``v``.

The vertex-circle radius ``r_v = |v - t|`` controls the depth of the spike at
``v``. The model is split into half-triangles ``(c, v, t)`` and each is lifted
into a curved patch. We use one barycentric coordinate, ``bary_v``, which
varies from ``0`` along the flat-base edge ``c-t`` (the boundary shared with the
adjacent face patch — surfaces meet smoothly here) to ``1`` at the spike apex
``v`` (where neighbouring patches converge into a sharp point). The patch
height is

.. code-block:: text

    z(bary_v) = - r_v * scale * spike_depth(bary_v) * (1 - apex_inset),

where ``scale = profile.y[-1] / shrink_factor`` is the profile's peak height
and ``spike_depth(0) = 0`` (flat at the c-t base), ``spike_depth(1) = 1``
(full spike depth at ``v``). ``spike_depth`` is the profile's cross-section
read from the *apex* end inwards: it has *zero slope at the base* (so adjacent
patches meet flat across ``c-t``) and *maximum slope at the apex* (so the
spike at ``v`` is pointy). See :func:`_spike_depth_from_profile`.

For ``r < 1`` the original vertex is replaced by a flat polygon dual to it,
``apex_inset = (1 - r) * sf / (1 - (1 - r)(1 - sf)) > 0`` is the perpendicular
fraction of the half-triangle covered by that flat polygon, and the curved
patch only fills the inner sub-triangle ``{c_inner, v, t_inner}`` (with
``c_inner = v + (1 - apex_inset)(c - v)`` and ``t_inner`` analogous). The
complementary trapezoid ``{c, c_inner, t_inner, t}`` is flat at ``z = 0`` and
tiles the lifted shrunken inner face.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .. import base, conway, half
from .profiles import Profile

if TYPE_CHECKING:
    from ..half import EuclideanPositionHEG


def _spike_depth_from_profile(
    profile: Profile,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Re-parameterise the 2D profile as a spike-depth function for 3D lifting.

    The :class:`Profile` stores its cross-section in the convention used by the
    2D crease-pattern pipeline: ``profile.t`` runs from ``0`` at the vertex
    side to ``sf = shrink_factor`` at the face/incenter side, and ``profile.y``
    is the corresponding cylinder height (zero at the vertex side, peak at the
    face side). For the canonical circular profile this means the slope of
    ``y`` is steepest at ``t=0`` and zero at ``t=sf``.

    When lifting into 3D we want the *opposite* orientation along the
    half-triangle ``(c, v, t)``: the patch should be flat (zero slope) along
    the ``c-t`` base (so neighbouring patches meet smoothly there) and steep
    at the vertex apex ``v`` (so vertices look pointy). We therefore flip the
    profile so that:

    * ``spike_bary[0] = 0`` corresponds to the c-t base, ``spike_bary[-1] = 1``
      corresponds to the apex ``v``;
    * ``spike_depth[0] = 0`` (no depth at base), ``spike_depth[-1] = 1`` (full
      depth at apex), with zero slope at ``bary=0`` and maximum slope at
      ``bary=1``.

    Returns:
        ``(spike_bary, spike_depth, scale)`` where ``scale = profile.y[-1] /
        profile.shrink_factor`` is the unscaled peak height (preserved for the
        existing convention ``h_v = r_v * scale * curved_extent``).
    """
    sf = profile.shrink_factor
    profile_x = profile.t / sf  # 0..1 in pipeline convention (0 = vertex side)
    profile_y = profile.y / sf  # 0..scale in pipeline convention
    scale = float(profile_y[-1])
    if scale <= 0.0:
        return profile_x.copy(), np.zeros_like(profile_x), 0.0
    spike_bary = 1.0 - profile_x[::-1]
    spike_depth = 1.0 - profile_y[::-1] / scale
    return spike_bary, spike_depth, scale


def _build_ortho_with_tangent_points(
    G: "EuclideanPositionHEG",
) -> "EuclideanPositionHEG":
    """Apply the ortho Conway operator and move edge-midpoint vertices to the
    actual edge tangent points (intersection of the original edge with the line
    between the two adjacent face incenters).
    """
    G = G.copy()
    for f in G.faces:
        f["midpoint"] = f.pseudo_incenter()

    G_ortho = conway.ortho_graph()(G, delete_on_border=False, copy_graph=True)

    for v in list(G_ortho.vertices):
        if "pre_conway" not in v.attributes:
            continue
        if not isinstance(v["pre_conway"], half.Face):
            continue
        # ``v`` is the ortho-vertex sitting at a face incenter; iterate the
        # adjacent edge-midpoint vertices and move them onto the original edge.
        for h_o in v.outgoing_iter():
            v_t = h_o.dest
            v_orig_1 = h_o.nex.dest
            v_orig_2 = h_o.rev.pre.orig
            v_t["pos"] = base.project_to_line(
                np.stack([v_orig_1["pos"], v_orig_2["pos"]]),
                v["pos"],
            )

    return G_ortho


def _vertex_circle_radii(G_ortho: "EuclideanPositionHEG") -> dict:
    """Return ``{original_vertex: r_v}`` where ``r_v = |v - t|`` is the radius
    of the blue circle centered at the original vertex (its distance to any of
    its incident edge-tangent points after the position fix).
    """
    radii: dict = {}
    for v_o in G_ortho.vertices:
        if "pre_conway" not in v_o.attributes:
            continue
        pre = v_o["pre_conway"]
        if not isinstance(pre, half.Vertex):
            continue
        if pre in radii:
            continue
        for h_o in v_o.outgoing_iter():
            d = h_o.dest
            d_pre = d["pre_conway"] if "pre_conway" in d.attributes else None
            # Edge-tangent corner: not an original Vertex, not an original Face.
            if not isinstance(d_pre, half.Vertex) and not isinstance(d_pre, half.Face):
                radii[pre] = float(np.linalg.norm(v_o["pos"] - d["pos"]))
                break
    return radii


def _classify_ortho_quad(face) -> tuple | None:
    """Return ``(v_corner, c_corner, t1_corner, t2_corner)`` for a 4-corner ortho
    quad, or ``None`` if it does not have the expected (V, E, F, E) structure.

    The ``v`` corner has ``pre_conway`` referencing an original
    :class:`half.Vertex`, the ``c`` corner references an original
    :class:`half.Face`, and the two ``t`` corners are the edge-midpoint
    vertices introduced by the ortho operator (typically without a
    ``pre_conway`` attribute, or referencing a :class:`half.HalfEdge`).
    """
    corners = list(face.vertex_iter())
    if len(corners) != 4:
        return None

    types: list[str] = []
    for cn in corners:
        pre = cn["pre_conway"] if "pre_conway" in cn.attributes else None
        if isinstance(pre, half.Vertex):
            types.append("V")
        elif isinstance(pre, half.Face):
            types.append("F")
        else:
            # Either no pre_conway or a HalfEdge: treat as edge-tangent corner.
            types.append("E")

    if types.count("V") != 1 or types.count("F") != 1 or types.count("E") != 2:
        return None

    v_idx = types.index("V")
    c_idx = types.index("F")
    e_indices = [i for i, t in enumerate(types) if t == "E"]
    return corners[v_idx], corners[c_idx], corners[e_indices[0]], corners[e_indices[1]]


def to_3d_mesh(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float = 1.0,
    n_along_edge: int = 8,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a triangle mesh of the folded intersecting-cylinders surface.

    See the module docstring for the geometric construction. The mesh is built
    by splitting every ortho quad along its ``v-c`` diagonal into two
    half-triangles ``{c, v, t}`` and lifting each with a profile-shaped height
    function. For ``r < 1`` the shrunken inner faces are filled with flat
    trapezoids at ``z = 0``.

    Args:
        G: Input tiling whose faces have well-defined (pseudo-)incenters.
            A working copy is taken; the input is not modified.
        profile: Cross-section curve.
        r: Triangle scaling matching :func:`make_intersecting_cylinders`.
        n_along_edge: Number of barycentric subdivisions along each side of
            every half-triangle (mesh resolution).

    Returns:
        ``(vertices, triangles)`` with shapes ``(N, 3)`` and ``(M, 3)``.
    """
    if not 0.0 < r <= 1.0:
        raise ValueError("r must lie in (0, 1]")

    G_ortho = _build_ortho_with_tangent_points(G)
    r_v_per_vertex = _vertex_circle_radii(G_ortho)

    spike_bary, spike_depth, scale = _spike_depth_from_profile(profile)

    if r == 1.0:
        apex_inset = 0.0
    else:
        sf = profile.shrink_factor
        apex_inset = (1.0 - r) * sf / (1.0 - (1.0 - r) * (1.0 - sf))
    curved_extent = 1.0 - apex_inset  # fraction of v-c filled by curved patch

    n = max(2, int(n_along_edge))
    vertices: list[NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []

    def _add_curved_half_triangle(
        c_pt: NDArray[np.float64],
        v_pt: NDArray[np.float64],
        t_pt: NDArray[np.float64],
        h_v: float,
    ) -> None:
        """Mesh a triangle with corners c (z=0), v (z=-h_v), t (z=0).

        Height varies with the barycentric coordinate ``bary_v`` (0 along the
        flat ``c-t`` base, 1 at the spike apex ``v``):

            z(bary_v) = -h_v * spike_depth(bary_v)

        where ``spike_depth`` has zero slope at ``bary_v=0`` (so neighbouring
        patches meet smoothly across the ``c-t`` base) and maximum slope at
        ``bary_v=1`` (so vertices look pointy). The two adjacent half-triangles
        in an ortho-quad have opposite 2D orientations around ``(c, v, t)``;
        we detect that here and swap to keep all triangle normals pointing up.
        """
        # Ensure CCW 2D orientation of (c, v, t) so every triangle's normal
        # points upward (+z), matching the single-valued height field.
        cross = (v_pt[0] - c_pt[0]) * (t_pt[1] - c_pt[1]) - (v_pt[1] - c_pt[1]) * (t_pt[0] - c_pt[0])
        if cross < 0.0:
            t_pt, c_pt = c_pt, t_pt

        coords: dict[tuple[int, int], int] = {}
        for i in range(n + 1):
            for j in range(n + 1 - i):
                bary_v = i / n
                bary_t = j / n
                bary_c = 1.0 - bary_v - bary_t
                pos = bary_c * c_pt + bary_v * v_pt + bary_t * t_pt
                if scale > 0:
                    z = -h_v * float(np.interp(bary_v, spike_bary, spike_depth))
                else:
                    z = 0.0
                coords[(i, j)] = len(vertices)
                vertices.append(np.array([pos[0], pos[1], z]))
        for i in range(n):
            for j in range(n - i):
                a = coords[(i, j)]
                b = coords[(i + 1, j)]
                cc = coords[(i, j + 1)]
                triangles.append((a, b, cc))
                if j < n - i - 1:
                    d = coords[(i + 1, j + 1)]
                    triangles.append((b, d, cc))

    for face in G_ortho.faces:
        classified = _classify_ortho_quad(face)
        if classified is None:
            continue
        v_corner, c_corner, t1_corner, t2_corner = classified

        v_orig = v_corner["pre_conway"]
        r_v = r_v_per_vertex.get(v_orig, 0.0)
        if r_v == 0.0:
            continue

        c_pos = np.asarray(c_corner["pos"], dtype=float)
        v_pos = np.asarray(v_corner["pos"], dtype=float)
        # Spike depth scaled by curved_extent so the cylinder proportions are
        # preserved as the curved patch shrinks toward v for r < 1.
        h_v = r_v * scale * curved_extent

        # Inner corner along v-c (r=1 => c_inner = c).
        c_inner = v_pos + curved_extent * (c_pos - v_pos)

        for t_corner in (t1_corner, t2_corner):
            t_pos = np.asarray(t_corner["pos"], dtype=float)
            t_inner = v_pos + curved_extent * (t_pos - v_pos)

            _add_curved_half_triangle(c_inner, v_pos, t_inner, h_v)

            # For r<1, fill the complementary flat trapezoid at z=0.
            if apex_inset > 0.0:
                quad = [c_pos, c_inner, t_inner, t_pos]
                cross = (quad[1][0] - quad[0][0]) * (quad[2][1] - quad[0][1]) - (quad[1][1] - quad[0][1]) * (
                    quad[2][0] - quad[0][0]
                )
                if cross < 0.0:
                    quad = quad[::-1]
                idx0 = len(vertices)
                for p in quad:
                    vertices.append(np.array([p[0], p[1], 0.0]))
                triangles.append((idx0, idx0 + 1, idx0 + 2))
                triangles.append((idx0, idx0 + 2, idx0 + 3))

    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)


def _fold_curves(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float,
) -> list[NDArray[np.float64]]:
    """Polylines for sharp folds in the 3D model.

    Returns the curved-crease polylines that go from each face incenter (or its
    shrunken-face corner for ``r<1``) down to the corresponding vertex spike,
    plus, for ``r<1``, the closed polygon of every shrunken inner face.
    """
    G_ortho = _build_ortho_with_tangent_points(G)
    r_v_per_vertex = _vertex_circle_radii(G_ortho)

    spike_bary, spike_depth, scale = _spike_depth_from_profile(profile)

    if r == 1.0:
        apex_inset = 0.0
    else:
        sf = profile.shrink_factor
        apex_inset = (1.0 - r) * sf / (1.0 - (1.0 - r) * (1.0 - sf))
    curved_extent = 1.0 - apex_inset

    curves: list[NDArray[np.float64]] = []

    # One polyline per ortho-quad: the c_inner -> v curve along the v-c diagonal.
    for face in G_ortho.faces:
        classified = _classify_ortho_quad(face)
        if classified is None:
            continue
        v_corner, c_corner, _, _ = classified
        v_orig = v_corner["pre_conway"]
        r_v = r_v_per_vertex.get(v_orig, 0.0)
        if r_v == 0.0:
            continue

        c_pos = np.asarray(c_corner["pos"], dtype=float)
        v_pos = np.asarray(v_corner["pos"], dtype=float)
        h_v = r_v * scale * curved_extent
        c_inner = v_pos + curved_extent * (c_pos - v_pos)

        n_samples = len(spike_bary)
        curve = np.empty((n_samples, 3))
        for k in range(n_samples):
            bary_v = float(spike_bary[k])
            xy = (1.0 - bary_v) * c_inner + bary_v * v_pos
            z = -h_v * float(spike_depth[k]) if scale > 0 else 0.0
            curve[k] = [xy[0], xy[1], z]
        curves.append(curve)

    # Inner shrunken-face polygons (closed) at z=0 for r<1 - one per original face.
    if apex_inset > 0.0:
        G_copy = G.copy()
        for f in G_copy.faces:
            f["midpoint"] = f.pseudo_incenter()
        for f in G_copy.faces:
            c = f.midpoint()
            if not np.all(np.isfinite(c)):
                continue
            poly = []
            for vert in f.vertex_iter():
                p_inner = curved_extent * np.asarray(vert["pos"], dtype=float) + apex_inset * c
                poly.append([p_inner[0], p_inner[1], 0.0])
            poly.append(poly[0])
            curves.append(np.asarray(poly, dtype=float))

    return curves


def show_3d(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float = 1.0,
    n_along_edge: int = 8,
    color: str = "lightblue",
    opacity: float = 1.0,
    height: int = 600,
    edge_color: str = "black",
    edge_width: float = 2.0,
    show_edges: bool = True,
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
        edge_color: Colour of the sharp-fold polylines.
        edge_width: Line width of the sharp-fold polylines.
        show_edges: When ``True`` (default), draw the sharp-fold lines where
            adjacent curved strips meet (and the inner face boundary for
            ``r < 1``).

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
        lighting=dict(
            ambient=0.4,
            diffuse=0.7,
            specular=0.6,
            roughness=0.2,
            fresnel=0.2,
        ),
        lightposition=dict(x=100, y=100, z=300),
    )

    data: list[Any] = [mesh]

    if show_edges:
        # Concatenate all polylines into a single Scatter3d trace, separating
        # individual curves with NaNs (plotly's standard polyline-break trick).
        curves = _fold_curves(G, profile, r=r)
        if curves:
            xs: list[float] = []
            ys: list[float] = []
            zs: list[float] = []
            for curve in curves:
                xs.extend(curve[:, 0].tolist())
                ys.extend(curve[:, 1].tolist())
                zs.extend(curve[:, 2].tolist())
                xs.append(np.nan)
                ys.append(np.nan)
                zs.append(np.nan)
            data.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=edge_color, width=edge_width),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig = go.Figure(data=data)
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
