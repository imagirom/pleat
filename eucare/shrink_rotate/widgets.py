"""Interactive widget for exploring shrink-rotate tessellations.

Requires the ``[notebook]`` extra (matplotlib + ipywidgets + ipympl).
Use inside a Jupyter notebook with ``%matplotlib widget``.
"""
from __future__ import annotations

import numpy as np

from .. import base
from ..utils import random_directed_set


def _require_notebook_deps():
    try:
        import ipywidgets as widgets  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        from matplotlib.collections import LineCollection, PolyCollection  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ShrinkRotateExplorer requires the 'notebook' extra. "
            "Install with: pip install 'eucare[notebook]'"
        ) from exc


class ShrinkRotateExplorer:
    """Interactive shrink-rotate / reciprocal-figure explorer.

    Build state once from an SRG (output of
    :func:`eucare.shrink_rotate.shrink_rotate_pattern`), then offer sliders for
    ``alpha`` and ``factor`` (or, in *reparametrized* mode, ``beta`` and
    ``gamma``) plus a few display toggles.

    Vertex positions on the supplied SRG are mutated in place at each tick,
    so downstream operations (e.g. ``fold_complete``) see current geometry.

    Parameters
    ----------
    SRG :
        Shrink-rotate graph with ``base_pos`` on every vertex and
        ``rotation_center`` on each twist face.
    alpha0, factor0:
        Initial slider values (``alpha`` is in units of π).
    figsize :
        Matplotlib figure size.
    figure_id :
        Stable figure id; passed as ``num=`` to :func:`plt.figure` so
        re-displaying replaces the previous figure rather than stacking.

    Notes
    -----
    Call :meth:`display` to show the widget. Public attributes ``figure``,
    ``alpha_slider``, ``factor_slider`` etc. are exposed for further
    customisation.
    """

    def __init__(
        self,
        SRG,
        *,
        alpha0: float = 1 / 6,
        factor0: float = 0.58,
        figsize: tuple[float, float] = (7, 7),
        figure_id: str = 'shrink-rotate',
    ) -> None:
        _require_notebook_deps()
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection

        self.SRG = SRG
        self._state = self._build_state(SRG)

        # Initial draw.
        self._reshrinkrotate(alpha0 * np.pi, factor0)

        # Suppress matplotlib's auto-display of the figure: we display the
        # canvas ourselves in :meth:`display`, so re-running the cell shows
        # the plot reliably (and exactly once).
        with plt.ioff():
            fig = plt.figure(num=figure_id, figsize=figsize, clear=True)
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        ax = fig.add_subplot(1, 1, 1)
        lc = LineCollection(self._segments(), antialiased=True, color='k', linewidth=1)
        pc = PolyCollection(self._polys(), antialiased=True, color='k', alpha=0.1)
        ax.add_collection(pc)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='datalim')

        self.figure = fig
        self.ax = ax
        self._lines = lc
        self._polys_artist = pc

        self.alpha_slider = widgets.FloatSlider(
            alpha0, min=-1, max=1, step=0.02, description='alpha/π'
        )
        self.factor_slider = widgets.FloatSlider(
            factor0, min=0, max=6, step=0.05, description='factor'
        )
        self.folded_cb = widgets.Checkbox(value=False, description='folded')
        self.reparam_cb = widgets.Checkbox(value=False, description='reparametrized')
        self.scale_folded_cb = widgets.Checkbox(value=False, description='scale_folded')
        self.show_lines_cb = widgets.Checkbox(value=False, description='show_lines')
        self.show_polys_cb = widgets.Checkbox(value=True, description='show_polys')
        self.info_label = widgets.Label(value='')

        self._last_reparametrized = False
        self._in_update = False  # re-entry guard

        self._ui = widgets.VBox([
            widgets.HBox([self.alpha_slider, self.factor_slider]),
            widgets.HBox([self.folded_cb, self.reparam_cb, self.scale_folded_cb]),
            widgets.HBox([self.show_lines_cb, self.show_polys_cb]),
            self.info_label,
        ])
        self._out = widgets.interactive_output(
            self._update,
            dict(
                alpha=self.alpha_slider,
                factor=self.factor_slider,
                folded=self.folded_cb,
                reparametrized=self.reparam_cb,
                scale_folded=self.scale_folded_cb,
                show_lines=self.show_lines_cb,
                show_polys=self.show_polys_cb,
            ),
        )

    # ------------------------------------------------------------------ state

    @staticmethod
    def _build_state(SRG) -> dict:
        """Precompute index arrays so each tick is one batched matmul."""
        vertex_list = list(SRG.vertices)
        vidx = {id(v): i for i, v in enumerate(vertex_list)}
        base_positions = np.array([v['base_pos'] for v in vertex_list])

        rot_v_idx, rot_base, rot_centers = [], [], []
        for f in SRG.faces:
            if 'rotation_center' not in f.attributes:
                continue
            c = np.asarray(f['rotation_center'])
            for v in f.vertex_iter():
                rot_v_idx.append(vidx[id(v)])
                rot_base.append(v['base_pos'])
                rot_centers.append(c)

        edges = list(random_directed_set(SRG.halfedges))
        edge_idx = np.array(
            [[vidx[id(e.orig)], vidx[id(e.dest)]] for e in edges], dtype=np.intp
        )
        face_idx = [
            np.array([vidx[id(v)] for v in f.vertex_iter()], dtype=np.intp)
            for f in SRG.faces
        ]
        return dict(
            vertex_list=vertex_list,
            positions=base_positions.copy(),
            rot_v_idx=np.array(rot_v_idx, dtype=np.intp),
            rot_base=np.array(rot_base),
            rot_centers=np.array(rot_centers),
            edge_idx=edge_idx,
            face_idx=face_idx,
        )

    def _reshrinkrotate(self, alpha: float, factor: float, global_scale: float = 1.0) -> None:
        s = self._state
        R = base.rotation_matrix(alpha)
        rot = s['rot_centers'] + (s['rot_base'] - s['rot_centers']) @ R * factor
        if global_scale != 1:
            rot *= global_scale
        s['positions'][s['rot_v_idx']] = rot
        pos = s['positions']
        for i, v in enumerate(s['vertex_list']):
            v['pos'] = pos[i]

    def _segments(self):
        s = self._state
        return s['positions'][s['edge_idx']]

    def _polys(self):
        s = self._state
        pos = s['positions']
        return [pos[idx] for idx in s['face_idx']]

    # ----------------------------------------------------------------- update

    def _update(self, alpha, factor, folded, reparametrized, scale_folded, show_lines, show_polys):
        if self._in_update:
            return
        if factor <= 0:
            self.info_label.value = 'factor must be > 0'
            return

        alpha = alpha * np.pi
        if not self._last_reparametrized:
            denom = np.sqrt(factor ** 2 - 2 * factor * np.cos(alpha) + 1)
            gamma = factor / denom
            beta = np.arccos(np.sin(alpha) / denom)
        else:
            gamma = factor
            beta = alpha
            # TODO: sign
            denom = np.sqrt(gamma ** 2 + 2 * gamma * np.sin(beta) + 1)
            alpha = np.arccos((gamma + np.sin(beta)) / denom)
            factor = gamma / denom

        if reparametrized is not self._last_reparametrized:
            self._last_reparametrized = reparametrized
            self._in_update = True
            try:
                if reparametrized:
                    self.alpha_slider.description = 'beta/π'
                    self.factor_slider.description = 'gamma'
                    self.alpha_slider.value = beta / np.pi
                    self.factor_slider.value = gamma
                else:
                    self.alpha_slider.description = 'alpha/π'
                    self.factor_slider.description = 'factor'
                    self.alpha_slider.value = alpha / np.pi
                    self.factor_slider.value = factor
            finally:
                self._in_update = False

        if not folded:
            self._reshrinkrotate(alpha, factor)
        else:
            denom_f = np.sqrt(gamma ** 2 - 2 * gamma * np.sin(beta) + 1)
            factor_folded = gamma / denom_f
            alpha_folded = np.sign(alpha) * np.arccos((gamma - np.sin(beta)) / denom_f)
            self._reshrinkrotate(
                alpha_folded,
                factor_folded,
                global_scale=1 if not scale_folded else factor / factor_folded,
            )

        self._lines.set_segments(self._segments() if show_lines else [])
        self._polys_artist.set_paths(self._polys() if show_polys else [])
        self.info_label.value = f'gamma={gamma:.4f}  beta={np.degrees(beta):.2f}°'
        self.figure.canvas.draw_idle()

    # ----------------------------------------------------------------- public

    def display(self):
        """Display the widget UI in a Jupyter notebook.

        Re-displays the figure canvas explicitly, so re-executing a cell
        that calls :meth:`display` keeps showing the plot (with
        ``%matplotlib widget``, the canvas widget from the previous
        execution is otherwise orphaned).
        """
        from IPython.display import display as ipy_display

        ipy_display(self.figure.canvas, self._ui, self._out)
