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
``v``. The model is split into half-triangles ``(c, v, t)`` and each half is
lifted using the profile's cross-section.

For ``r = 1`` the half-triangle becomes one curved patch with a sharp apex at
``v`` (depth ``-r_v * scale``). For ``r < 1`` the cylinder cross-section stays
*self-similar* but is rescaled: the curved patch occupies only the outer
fraction ``curved_extent = 1 - apex_inset`` of the ``v-c`` direction, and the
spike depth shrinks proportionally to ``-r_v * scale * curved_extent``. The
missing apex is replaced by a flat cap at the original vertex:

* a **curved trapezoid** ``{c, c_near_v, t_near_v, t}`` filling the outer
  portion of the half-triangle, with ``c, t`` at ``z = 0`` and
  ``c_near_v, t_near_v`` at the flat-tip depth ``-r_v * scale * curved_extent``,
  and
* a **flat tip triangle** ``{v, c_near_v, t_near_v}`` at the same depth.
  Combined across all half-triangles incident to ``v``, these triangles form
  the closed flat polygon that caps the (proportionally smaller) cylinder at
  the vertex.

Here ``c_near_v = v + apex_inset * (c - v)`` and ``t_near_v = v + apex_inset *
(t - v)`` -- both close to ``v``. ``apex_inset = (1 - r) * sf / (1 - (1 - r) *
(1 - sf))`` (zero for ``r = 1``).

The curved trapezoid is lifted by interpreting the profile as a spike-depth
function read from the *apex end inwards*: zero depth and zero slope at the
``c-t`` base (so neighbouring half-triangle patches meet smoothly across
``c-t``) and maximum slope toward the flat tip (so vertices look pointy).
See :func:`_spike_depth_from_profile`.

Sampling
--------

Along the curved direction, the lift uses the profile's own
RDP-simplified sample points (which already concentrate samples near the
steep apex). When the profile has more than ``max_profile_samples`` points
they are uniformly subsampled in index space, always keeping the first and
last. Across the edge, the mesh uses ``n_across_edge`` uniform subdivisions
(the trapezoid is linear in that direction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .. import base, conway, half
from .profiles import Profile

if TYPE_CHECKING:
    from ..half import EuclideanPositionHEG, Face


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
        profile.shrink_factor`` is the height of a full (un-truncated) spike
        at a vertex of radius ``r_v = 1``.
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


def _curved_patch_samples(
    spike_bary: NDArray[np.float64],
    spike_depth: NDArray[np.float64],
    max_samples: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pick the sample positions along the curved direction of one patch.

    Uses the profile's own (RDP-simplified) sample points -- they already
    concentrate where the curve is steep, which is exactly where the 3D
    mesh needs more resolution. ``u_samples`` cover the full curve from
    base (``u=0``) to apex (``u=1``); the curved patch is then mapped onto
    its (possibly shrunken) perpendicular extent ``c -> c_near_v`` by linear
    interpolation. The endpoints ``0`` and ``1`` are guaranteed to be
    present.

    If the resulting sample list has more than ``max_samples`` entries it is
    uniformly subsampled in index space, always preserving the first and last
    sample.

    Args:
        spike_bary: Profile barycentric positions (``0`` = c-t base, ``1`` =
            spike apex).
        spike_depth: Matching depth fractions (``0`` at base, ``1`` at apex).
        max_samples: Cap on the number of returned samples (``>= 2``).

    Returns:
        ``(u_samples, depth_samples)`` where ``u_samples`` are in ``[0, 1]``
        (``0`` at the c-side base, ``1`` at the flat tip / spike apex) and
        ``depth_samples`` are the corresponding spike-depth fractions in
        ``[0, 1]``.
    """
    max_samples = max(2, int(max_samples))
    bary = spike_bary.astype(float, copy=True)
    depth = spike_depth.astype(float, copy=True)
    if bary.size == 0 or bary[0] > 1e-12:
        bary = np.concatenate([[0.0], bary])
        depth = np.concatenate([[0.0], depth])
    if bary[-1] < 1.0 - 1e-12:
        bary = np.concatenate([bary, [1.0]])
        depth = np.concatenate([depth, [1.0]])

    if bary.size > max_samples:
        idx = np.unique(np.linspace(0, bary.size - 1, max_samples).round().astype(int))
        bary = bary[idx]
        depth = depth[idx]

    return bary, depth


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


def _vertex_circle_radii(G_ortho: "EuclideanPositionHEG") -> dict[half.Vertex, float]:
    """Return ``{original_vertex: r_v}`` where ``r_v = |v - t|`` is the radius
    of the blue circle centered at the original vertex (its distance to any of
    its incident edge-tangent points after the position fix).
    """
    radii: dict[half.Vertex, float] = {}
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


def _classify_ortho_quad(
    face: "Face",
) -> tuple[half.Vertex, half.Vertex, half.Vertex, half.Vertex] | None:
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
    n_across_edge: int = 8,
    max_profile_samples: int = 30,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Build a triangle mesh of the folded intersecting-cylinders surface.

    See the module docstring for the geometric construction. Each ortho quad
    ``(v, t1, c, t2)`` is split along its ``v-c`` diagonal into two
    half-triangles ``(c, v, t)``; each half-triangle is decomposed into a
    curved trapezoid ``{c, c_near_v, t_near_v, t}`` filling the outer part
    plus (for ``r < 1``) a flat tip triangle ``{v, c_near_v, t_near_v}`` at
    the vertex. The curved trapezoid is lifted using the profile's
    cross-section.

    Args:
        G: Input tiling whose faces have well-defined (pseudo-)incenters.
            A working copy is taken; the input is not modified.
        profile: Cross-section curve.
        r: Triangle scaling matching :func:`make_intersecting_cylinders`.
        n_across_edge: Number of uniform subdivisions across each half-edge
            (the ``c-t`` direction of the curved trapezoid). The trapezoid is
            linear in this direction.
        max_profile_samples: Cap on the number of sample points used along
            the curved (spike-depth) direction. The profile's own
            RDP-simplified samples are used (so the resolution is highest
            where the curve is steepest); when there are more than this many,
            they are uniformly subsampled in index space, always keeping the
            first and last.

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

    u_samples, depth_samples = _curved_patch_samples(spike_bary, spike_depth, max_profile_samples)
    n_u = u_samples.size  # rows along the spike-depth direction
    n_w = max(2, int(n_across_edge)) + 1  # columns across the c-t direction

    vertices: list[NDArray[np.float64]] = []
    triangles: list[tuple[int, int, int]] = []

    def _add_curved_trapezoid(
        c_pt: NDArray[np.float64],
        c_near_v: NDArray[np.float64],
        t_near_v: NDArray[np.float64],
        t_pt: NDArray[np.float64],
        h_v: float,
    ) -> None:
        """Mesh the curved trapezoid ``{c, c_near_v, t_near_v, t}``.

        ``u`` runs ``0 -> 1`` from the ``c-t`` base (z=0) to the
        ``c_near_v-t_near_v`` top (z=-h_v, the flat-tip depth); ``w`` runs
        ``0 -> 1`` from the c-side to the t-side. The full profile shape is
        rescaled onto the patch's perpendicular extent so the cylinder keeps
        its proportions (the curved patch always lifts to its full natural
        depth ``h_v`` at the apex side; only its perpendicular extent shrinks
        as ``apex_inset`` grows).

        The two adjacent half-triangles in an ortho-quad have opposite 2D
        orientations around ``(c, v, t)``; we detect this here (using the
        cross product of (c -> t) and (c -> c_near_v)) and swap the c/t pair
        to keep every triangle's normal pointing up.
        """
        cross = (c_near_v[0] - c_pt[0]) * (t_pt[1] - c_pt[1]) - (c_near_v[1] - c_pt[1]) * (t_pt[0] - c_pt[0])
        if cross < 0.0:
            c_pt, t_pt = t_pt, c_pt
            c_near_v, t_near_v = t_near_v, c_near_v

        base_offset = len(vertices)
        for iu in range(n_u):
            u = float(u_samples[iu])
            z = -h_v * float(depth_samples[iu]) if scale > 0 else 0.0
            base_u = (1.0 - u) * c_pt + u * c_near_v
            top_u = (1.0 - u) * t_pt + u * t_near_v
            for iw in range(n_w):
                w = iw / (n_w - 1)
                pos = (1.0 - w) * base_u + w * top_u
                vertices.append(np.array([pos[0], pos[1], z]))

        for iu in range(n_u - 1):
            for iw in range(n_w - 1):
                a = base_offset + iu * n_w + iw
                b = base_offset + (iu + 1) * n_w + iw
                c_i = base_offset + iu * n_w + (iw + 1)
                d = base_offset + (iu + 1) * n_w + (iw + 1)
                triangles.append((a, b, c_i))
                triangles.append((b, d, c_i))

    def _add_flat_tip_triangle(
        v_pt: NDArray[np.float64],
        c_near_v: NDArray[np.float64],
        t_near_v: NDArray[np.float64],
        z: float,
    ) -> None:
        """Add the flat tip triangle ``{v, c_near_v, t_near_v}`` at ``z``.

        Detect CCW orientation in 2D and swap so the normal points up.
        """
        cross = (c_near_v[0] - v_pt[0]) * (t_near_v[1] - v_pt[1]) - (c_near_v[1] - v_pt[1]) * (t_near_v[0] - v_pt[0])
        a, b, c_i = v_pt, c_near_v, t_near_v
        if cross < 0.0:
            b, c_i = c_i, b
        idx0 = len(vertices)
        for p in (a, b, c_i):
            vertices.append(np.array([p[0], p[1], z]))
        triangles.append((idx0, idx0 + 1, idx0 + 2))

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
        # The curved patch always reaches its full natural depth at the apex
        # side; the patch's perpendicular extent shrinks with curved_extent
        # (the c-to-c_near_v segment), and the depth scales proportionally
        # so the cylinder cross-section stays self-similar.
        h_v = r_v * scale * curved_extent
        z_tip = -h_v if scale > 0 else 0.0

        c_near_v = v_pos + apex_inset * (c_pos - v_pos)

        for t_corner in (t1_corner, t2_corner):
            t_pos = np.asarray(t_corner["pos"], dtype=float)
            t_near_v = v_pos + apex_inset * (t_pos - v_pos)

            _add_curved_trapezoid(c_pos, c_near_v, t_near_v, t_pos, h_v)

            if apex_inset > 0.0:
                _add_flat_tip_triangle(v_pos, c_near_v, t_near_v, z_tip)

    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)


def _fold_curves(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float,
    max_profile_samples: int = 30,
) -> list[NDArray[np.float64]]:
    """Polylines for sharp folds in the 3D model.

    Returns:

    * one curved polyline per half-edge of every ortho-quad, running along
      the ``v-c`` diagonal from the face incenter ``c`` (``z = 0``) down to
      the corner of the flat tip ``c_near_v`` (``z = z_tip``);
    * for ``r < 1``, the boundary segments of every flat tip (the
      ``c_near_v - t_near_v`` lines at ``z = z_tip``), which together outline
      the flat caps at each original vertex.
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

    u_samples, depth_samples = _curved_patch_samples(spike_bary, spike_depth, max_profile_samples)

    curves: list[NDArray[np.float64]] = []

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
        h_v = r_v * scale * curved_extent
        z_tip = -h_v if scale > 0 else 0.0

        c_near_v = v_pos + apex_inset * (c_pos - v_pos)

        # Ridge polyline: c -> c_near_v along the v-c diagonal.
        n_s = u_samples.size
        curve = np.empty((n_s, 3))
        for k in range(n_s):
            u = float(u_samples[k])
            xy = (1.0 - u) * c_pos + u * c_near_v
            z = -h_v * float(depth_samples[k]) if scale > 0 else 0.0
            curve[k] = [xy[0], xy[1], z]
        curves.append(curve)

        # Flat-tip boundary segments (one per half-triangle).
        if apex_inset > 0.0:
            for t_corner in (t1_corner, t2_corner):
                t_pos = np.asarray(t_corner["pos"], dtype=float)
                t_near_v = v_pos + apex_inset * (t_pos - v_pos)
                seg = np.array(
                    [
                        [c_near_v[0], c_near_v[1], z_tip],
                        [t_near_v[0], t_near_v[1], z_tip],
                    ]
                )
                curves.append(seg)

    return curves


def show_3d(
    G: "EuclideanPositionHEG",
    profile: Profile,
    r: float = 1.0,
    n_across_edge: int = 8,
    max_profile_samples: int = 30,
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
        n_across_edge: Mesh resolution across each edge (linear direction).
        max_profile_samples: Cap on the number of sample points along the
            curved (spike-depth) direction; see :func:`to_3d_mesh`.
        color: Surface colour.
        opacity: Surface opacity in ``[0, 1]``.
        height: Figure height in pixels.
        edge_color: Colour of the sharp-fold polylines.
        edge_width: Line width of the sharp-fold polylines.
        show_edges: When ``True`` (default), draw the sharp-fold lines where
            adjacent curved strips meet (and the flat-tip boundaries for
            ``r < 1``).

    Returns:
        Plotly ``Figure`` ready for ``fig.show()`` or rendering in a notebook.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("show_3d requires plotly; install with `pip install plotly`.") from exc

    vertices, triangles = to_3d_mesh(
        G,
        profile,
        r=r,
        n_across_edge=n_across_edge,
        max_profile_samples=max_profile_samples,
    )

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
            diffuse=0.6,
            specular=0.3,
            roughness=0.3,
        ),
        lightposition=dict(x=100, y=100, z=300),
    )

    data: list[Any] = [mesh]

    if show_edges:
        # Concatenate all polylines into a single Scatter3d trace, separating
        # individual curves with NaNs (plotly's standard polyline-break trick).
        curves = _fold_curves(G, profile, r=r, max_profile_samples=max_profile_samples)
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
