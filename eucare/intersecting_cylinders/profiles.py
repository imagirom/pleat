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

    Attributes:
        t: Normalised position along the flat edge (``[0, shrink_factor]``).
        l: Normalised arc length, parallel to ``t`` (``[0, shrink_factor]`` end).
        shrink_factor: ``1 / total_unscaled_arc_length`` of the original curve.
            After normalisation ``t[-1] == l[-1] == shrink_factor``.
    """

    t: NDArray[np.float64]
    l: NDArray[np.float64]
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
        y = np.asarray(fn(t_dense), dtype=float)

        dy = np.diff(y)
        dt = np.diff(t_dense)
        l_dense = np.concatenate([[0.0], np.cumsum(np.sqrt(dy * dy + dt * dt))])

        # rdp deprecates 2D vectors in NumPy 2.0; pad with a zero column.
        tl = np.stack([t_dense, l_dense, np.zeros_like(t_dense)], axis=-1)
        tl = rdp(tl, rdp_tol)[:, :2]
        t, l = tl.T

        total_length = float(l[-1])
        if total_length <= 0.0:
            raise ValueError("profile must have positive total arc length")
        shrink_factor = 1.0 / total_length
        return cls(t=t * shrink_factor, l=l * shrink_factor, shrink_factor=shrink_factor)

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
