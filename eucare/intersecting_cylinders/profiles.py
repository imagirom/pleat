"""Profile curves used by the intersecting-cylinders pattern.

A :class:`Profile` describes the cross-section of the curved triangles that sit
on the edges of the input tiling. It stores arc-length-parametrised samples of
a height function ``fn(x)`` for ``x in [0, 1]``, normalised so that the total
arc length equals 1.

The cross-section is interpreted differently by the 2D pipeline (which produces
the crease pattern in :mod:`eucare.intersecting_cylinders.pipeline`) and by the
3D mesh builder (:mod:`eucare.intersecting_cylinders.mesh3d`):

* In 2D, ``profile.t`` is the *perpendicular* coordinate of the crease and
  ``profile.l`` the *along-edge* arc-length coordinate. ``profile.t = 0``
  corresponds to the *vertex side* of the half-triangle and ``profile.t = sf``
  to the *face/incenter side*.
* In 3D, the same height function is re-used but read from the *apex end
  inwards* so that the spike forms at the vertex; see
  :func:`eucare.intersecting_cylinders.mesh3d._spike_depth_from_profile`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from rdp import rdp


@dataclass(frozen=True)
class Profile:
    """Arc-length-parametrised cross-section curve.

    The cross-section is the curve ``(x, fn(x))`` for ``x in [0, 1]``, where
    ``fn(0) = 0`` and ``fn`` is non-negative. Samples are simplified by the
    Ramer-Douglas-Peucker algorithm and stored after rescaling so that the
    **total arc length is 1** (and the perpendicular extent is therefore
    ``shrink_factor = 1 / unscaled_arc_length <= 1``).

    Attributes:
        t: Perpendicular coordinate, scaled to ``[0, shrink_factor]``. In the
            2D pipeline, ``t = 0`` is the *vertex-side* end of the curved
            crease and ``t = shrink_factor`` is the *face-side* end (toward the
            incenter).
        l: Arc length along the curve, scaled to ``[0, 1]``. Used by the 2D
            pipeline as the along-edge coordinate of the curved crease.
        y: Height ``fn(x)``, scaled by ``shrink_factor`` so it ranges in
            ``[0, fn_max * shrink_factor]``. Drives the 3D spike depth (after
            re-orientation by ``mesh3d._spike_depth_from_profile``).
        shrink_factor: ``1 / total_unscaled_arc_length``. Equals the
            perpendicular extent of the *folded* fundamental strip relative to
            the original tile edge length.
    """

    t: NDArray[np.float64]
    l: NDArray[np.float64]
    y: NDArray[np.float64]
    shrink_factor: float

    @classmethod
    def from_function(
        cls,
        fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        n_samples: int = 1000,
        rdp_tol: float = 1e-4,
    ) -> "Profile":
        """Build a :class:`Profile` from a height function ``fn`` on ``[0, 1]``.

        Args:
            fn: Height function with ``fn(0) == 0``. Evaluated on
                ``np.linspace(0, 1, n_samples)``.
            n_samples: Number of samples used before RDP simplification.
            rdp_tol: Ramer-Douglas-Peucker tolerance used to simplify the
                ``(t, l)`` polyline.

        Returns:
            A :class:`Profile` with the curve normalised so total arc length is 1.
        """
        t_dense = np.linspace(0.0, 1.0, n_samples)
        y_dense = np.asarray(fn(t_dense), dtype=float)

        dy = np.diff(y_dense)
        dt = np.diff(t_dense)
        l_dense = np.concatenate([[0.0], np.cumsum(np.sqrt(dy * dy + dt * dt))])

        # Simplify the (t, l) polyline; pad with a zero column to silence the
        # numpy 2.0 deprecation warning about 2D vectors. ``return_mask`` lets us
        # subset the parallel ``y`` array at the same surviving indices.
        mask = rdp(
            np.stack([t_dense, l_dense, np.zeros_like(t_dense)], axis=-1),
            rdp_tol,
            return_mask=True,
        )
        t = t_dense[mask]
        l = l_dense[mask]
        y = y_dense[mask]

        total_length = float(l[-1])
        if total_length <= 0.0:
            raise ValueError("profile must have positive total arc length")
        shrink_factor = 1.0 / total_length
        return cls(
            t=t * shrink_factor,
            l=l * shrink_factor,
            y=y * shrink_factor,
            shrink_factor=shrink_factor,
        )

    def plot(self, ax: Any = None) -> None:
        """Plot the simplified ``(t, y)`` polyline."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot(self.t, self.y)
        ax.set_aspect(1)


def circular_profile(scale: float = 1.3, n_samples: int = 1000, rdp_tol: float = 1e-4) -> Profile:
    """Quarter-ellipse cross-section ``y = scale * sqrt(1 - (1 - x)**2)``.

    With ``scale = 1`` this is the canonical quarter-circle and produces
    intersecting half-cylinders for the platonic 4 and 3 tilings. Larger
    values of ``scale`` make the bumps taller and steeper.

    Args:
        scale: Height multiplier on the quarter-circle.
        n_samples: Forwarded to :meth:`Profile.from_function`.
        rdp_tol: Forwarded to :meth:`Profile.from_function`.
    """
    return Profile.from_function(
        lambda x: scale * np.sqrt(np.clip(1.0 - (1.0 - x) ** 2, 0.0, 1.0)),
        n_samples=n_samples,
        rdp_tol=rdp_tol,
    )


def parabolic_profile(scale: float = 1.3, n_samples: int = 1000, rdp_tol: float = 1e-4) -> Profile:
    """Parabolic cross-section ``y = scale * (1 - (1 - x)**2)``.

    This is a smoother curve than the circular profile, with zero slope at the
    start and a more gradual approach to the maximum height. The resulting
    crease pattern is less spiky and may be easier to fold.

    Args:
        scale: Height multiplier on the parabola.
        n_samples: Forwarded to :meth:`Profile.from_function`.
        rdp_tol: Forwarded to :meth:`Profile.from_function`.
    """
    return Profile.from_function(
        lambda x: scale * (1.0 - (1.0 - x) ** 2),
        n_samples=n_samples,
        rdp_tol=rdp_tol,
    )
