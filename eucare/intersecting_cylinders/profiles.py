"""Profile curves used by the intersecting-cylinders pattern.

A :class:`Profile` describes the cross-section of the curved triangles that sit on
the edges of the input tiling. It stores arc-length-parametrised samples ``(t, l)``
(both normalised so total arc-length is 1) together with the corresponding
``shrink_factor`` (the inverse of the original total arc-length).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from rdp import rdp


@dataclass(frozen=True)
class Profile:
    """Arc-length-parametrised cross-section curve.

    The cross-section is a curve in the ``(perpendicular, height)`` plane defined
    by a height function ``fn(x)`` for ``x in [0, 1]``. Samples of this curve are
    simplified by Ramer-Douglas-Peucker and stored in normalised form so that
    the **total arc length is 1**.

    Attributes:
        t: ``perpendicular_axis * shrink_factor``; ranges in ``[0, shrink_factor]``.
            This is what the crease pattern uses as the perpendicular component
            of the curved fold.
        l: Normalised arc length; ranges in ``[0, 1]``.
        y: ``height_axis * shrink_factor``; ranges in ``[0, ymax * shrink_factor]``.
            Used for 3D mesh construction.
        shrink_factor: ``1 / total_unscaled_arc_length`` of the original curve.
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
        """Build a :class:`Profile` from a height function ``fn`` defined on ``[0, 1]``.

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

    def plot(self, ax=None) -> None:
        """Plot the simplified ``(l, t)`` polyline; convenience for notebooks."""
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot(self.l, self.t)
        ax.scatter(self.l, self.t, marker=".")
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
