"""Marching-cubes mesh generation from signed-distance fields.

Thin wrapper around :mod:`sdf` for converting SDFs into triangle meshes that
can be exported to STL via :mod:`meshio`.  Used to produce 3D-printable
thickened versions of crease patterns.  Optional dependency: ``threed``.
"""

from __future__ import annotations

import logging
from numbers import Number
from typing import Any, Callable

import numpy as np
import sdf
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Notes on the marching-cubes pipeline:
#
# Inputs:
#   - callable sdf mapping positions of shape (n, 3) to signed distances of shape (n,)
#   - bounds ((x0, y0, z0), (x1, y1, z1)); can be auto-estimated via
#     sdf.mesh._estimate_bounds
#   - target resolution (dx, dy, dz)
#
# Algorithm:
#   1. Compute the final grid size nx, ny, nz.
#   2. Construct a series of abstract grid objects that compose the oct-tree.


class CartesianGrid:
    """Axis aligned cartesian grid in n-dimensional euclidean space"""

    def __init__(
        self,
        ndim: int,
        origin: NDArray[Any] | None = None,
        cell_size: NDArray[Any] | None = None,
    ) -> None:
        self.ndim = ndim
        self.origin = np.zeros(self.ndim) if origin is None else origin
        self.cell_size = np.ones(self.ndim, dtype=np.float32) if cell_size is None else cell_size
        assert (
            self.origin.shape == self.cell_size.shape == (self.ndim,)
        ), f"Inconsistent shapes: {self.origin.shape=}, {self.cell_size.shape=}, {self.ndim=}"
        self.cell_center_to_corner_distance = np.linalg.norm(self.cell_size) / 2

    def cell_centers(self, indices: np.ndarray) -> np.ndarray:
        """Get the centers of grid cells at specified indices

        Args:
            indices: Array of shape (n, dim), of indices along the different axes.

        Returns:
            center positions as array of shape (n, dim).
        """

        return (indices + 0.5) * self.cell_size[None]

    def _parse_subdivisions(self, subdivisions: int | tuple | np.ndarray) -> np.ndarray:
        """Coerce a subdivision spec into a 1-D ``int`` array of length ``ndim``."""
        if isinstance(subdivisions, Number):
            subdivisions = np.ones(self.ndim, dtype=int) * subdivisions
        else:
            subdivisions = np.asarray(subdivisions, dtype=int)
            assert subdivisions.shape == (self.ndim,), f"{subdivisions.shape=} != {(self.ndim,)=}"
        return subdivisions

    def get_1d_axis_grids(self) -> list["CartesianGrid"]:
        """Get a list of 1D grids, corresponding to the axes of this grid"""
        return [
            CartesianGrid(ndim=1, origin=o, cell_size=s) for o, s in zip(self.origin[:, None], self.cell_size[:, None])
        ]

    def subdivide(self, subdivisions: int | tuple | np.ndarray = 2) -> "CartesianGrid":
        """Subdivide the grid, resulting in a finer grid

        Args:
            subdivisions: Number of subdivisions per dimension. If integer, each dimension will be subdivided by the
                given number. Default is 2.

        Returns:
            The subdivided :class:`CartesianGrid`
        """
        subdivisions = self._parse_subdivisions(subdivisions)

        return CartesianGrid(
            ndim=self.ndim,
            origin=self.origin,
            cell_size=self.cell_size / subdivisions,
        )

    def coarsen(self, subdivisions: int | tuple | np.ndarray = 2) -> "CartesianGrid":
        """The inverse to :meth:`subdivide`."""
        subdivisions = self._parse_subdivisions(subdivisions)

        return CartesianGrid(
            ndim=self.ndim,
            origin=self.origin,
            cell_size=self.cell_size * subdivisions,
        )

    def coarsen_indices(self, indices: np.ndarray, subdivisions: int | tuple | np.ndarray) -> np.ndarray:
        """Get list of unique indices of cells in coarser grid, that the grid cells at given indices are part of."""
        subdivisions = self._parse_subdivisions(subdivisions)
        assert len(indices.shape) == 2 and indices.shape[1] == self.ndim, f"{indices.shape=}, {self.ndim=}"
        return np.unique(indices // subdivisions, axis=0)

    def coarsen_block_indices(
        self, indices_per_dim: list[np.ndarray], subdivisions: int | tuple | np.ndarray
    ) -> list[np.ndarray]:
        """Apply :meth:`coarsen_indices` independently per axis to a block of indices."""
        subdivisions = self._parse_subdivisions(subdivisions)
        assert len(indices_per_dim) == self.ndim, f"{len(indices_per_dim)=} != {self.ndim=}"
        axis_grids = self.get_1d_axis_grids()
        return [
            grid.coarsen_indices(indices[:, None], subdivisions=s)[:, 0]
            for grid, indices, s in zip(axis_grids, indices_per_dim, subdivisions)
        ]

    def subdivide_indices(self, indices: np.ndarray, subdivisions: int | tuple | np.ndarray) -> np.ndarray:
        """Get list of indices of cells in finer grid that encompass the grid cells at given indices."""
        subdivisions = self._parse_subdivisions(subdivisions)

        # Indices of the cells in the finer grid that make up the (0, 0, ..., 0) cell in the coarser grid.
        # Shape (np.prod(subdivisions), self.ndim)
        block_indices = np.stack(np.meshgrid(*[np.arange(s) for s in subdivisions], indexing="ij"), axis=-1).reshape(
            -1, self.ndim
        )

        fine_indices = ((indices * subdivisions)[:, None] + block_indices[None]).reshape(-1, self.ndim)
        return fine_indices

    # def


def oct_tree_marching_cubes(
    f: Callable[[NDArray[Any]], NDArray[Any]],
    step: NDArray[Any],
    bounds: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], CartesianGrid]:
    """Locate boundary cells of an SDF using an oct-tree refinement strategy.

    Builds successively coarser grids until the bounding box fits in one
    cell, then refines top-down, keeping only cells whose distance is within
    the cell radius. Returns the surviving cell indices and the finest grid.

    Args:
        f: Signed-distance function mapping ``(n, 3)`` positions to ``(n, 1)``
            distances.
        step: Grid spacing per axis (or scalar applied to all).
        bounds: Axis-aligned bounding box ``((x0, y0, z0), (x1, y1, z1))`` to
            sample. If ``None``, estimated via :func:`sdf.mesh._estimate_bounds`.

    Returns:
        ``(indices, grid)`` where ``indices`` is the array of surviving cell
        indices in the finest grid and ``grid`` is that :class:`CartesianGrid`.
    """
    ndim = 3

    if bounds is None:
        bounds = sdf.mesh._estimate_bounds(f)
    bounds = np.asarray(bounds)

    if isinstance(step, Number):
        step = np.array((step,) * ndim)

    assert np.all(bounds[0] < bounds[1]), f"Bounds must be sorted. Got {bounds=}."
    assert np.all(step > 0), f"Got negative step size: {step=}"

    grids = [CartesianGrid(ndim=ndim, origin=bounds[0], cell_size=step)]

    box_size = bounds[1] - bounds[0]

    subdivisions = 2
    level = 0
    block_indices = [np.arange(n) for n in box_size // step]

    def block_indices_n_cells(block_indices: list[NDArray[Any]]) -> int:
        return int(np.prod([len(indices) for indices in block_indices]))

    n_cells_naive = block_indices_n_cells(block_indices)
    logger.info("n_cells_naive=%d", n_cells_naive)

    # coarsen until the whole box fits into a single grid cell
    while block_indices_n_cells(block_indices) > 1:
        logger.debug("level %d, n_cells=%d", level, block_indices_n_cells(block_indices))
        new_block_indices = grids[-1].coarsen_block_indices(block_indices, subdivisions=subdivisions)
        assert block_indices_n_cells(new_block_indices) < block_indices_n_cells(block_indices)
        block_indices = new_block_indices
        grids.append(grids[-1].coarsen(subdivisions=subdivisions))
        level += 1

    indices = np.zeros((1, ndim), dtype=np.int32)
    for i, grid in enumerate(reversed(grids)):
        # filter indices
        pts = grid.cell_centers(indices)
        dists = f(pts)[:, 0]  # for some reason, the distances are returned in an array of shape (n, 1)
        mask = np.abs(dists) <= grid.cell_center_to_corner_distance
        logger.debug("level %d, mask.mean=%.3f, mask.sum=%d", len(grids) - 1 - i, mask.mean(), mask.sum())
        indices = indices[mask]

        if i != len(grids) - 1:
            # get indices in finder grid
            indices = grid.subdivide_indices(indices, subdivisions=subdivisions)

    logger.info("n_cells=%d", len(indices))
    return indices, grids[0]
