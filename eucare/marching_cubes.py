import logging
from numbers import Number

import numpy as np
import sdf

logger = logging.getLogger(__name__)

from skimage.measure import marching_cubes

"""

Inputs: 
- callable sdf which maps array of positions, shape (n, 3) to array of signed distances of shape (n).
- bounds (could be automatically estimated using ``sdf.mesh._estimate_bounds``) of shape ((x0, y0, z0), (x1, y1, z1))
- target resolution as (dx, dy, dz)

Algorithm:
1. Compute the final grid size nx, ny, nz.
2. Construct series of abstract grid objects that compose the oct-tree

"""


class CartesianGrid:
    """Axis aligned cartesian grid in n-dimensional euclidean space"""
    def __init__(self, ndim: int, origin: np.ndarray = None, cell_size: np.ndarray = None) -> None:
        self.ndim = ndim
        self.origin = np.zeros(self.ndim) if origin is None else origin
        self.cell_size = np.ones(self.ndim, dtype=np.float32) if cell_size is None else cell_size
        assert self.origin.shape == self.cell_size.shape == (self.ndim,), (
            f'Inconsistent shapes: {self.origin.shape=}, {self.cell_size.shape=}, {self.ndim=}'
        )
        self.cell_center_to_corner_distance = np.linalg.norm(self.cell_size) / 2

    def cell_centers(self, indices: np.ndarray) -> np.ndarray:
        """Get the centers of grid cells at specified indices

        Args:
            indices: Array of shape (n, dim), of indices along the different axes.

        Returns:
            center positions as array of shape (n, dim).
        """

        return (indices + 0.5) * self.cell_size[None]


    def _parse_subdivisions(self, subdivisions: int | tuple | np.ndarray):
        if isinstance(subdivisions, Number):
            subdivisions = np.ones(self.ndim, dtype=int) * subdivisions
        else:
            subdivisions = np.asarray(subdivisions, dtype=int)
            assert subdivisions.shape == (self.ndim,), f'{subdivisions.shape=} != {(self.ndim,)=}'
        return subdivisions


    def get_1d_axis_grids(self) -> list["CartesianGrid"]:
        """Get a list of 1D grids, corresponding to the axes of this grid"""
        return [
            CartesianGrid(ndim=1, origin=o, cell_size=s)
            for o, s in zip(self.origin[:, None], self.cell_size[:, None])
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
        assert len(indices.shape) == 2 and indices.shape[1] == self.ndim, f'{indices.shape=}, {self.ndim=}'
        return np.unique(indices // subdivisions, axis=0)


    def coarsen_block_indices(
            self, indices_per_dim: list[np.ndarray], subdivisions: int | tuple | np.ndarray
    ) -> list[np.ndarray]:
        subdivisions = self._parse_subdivisions(subdivisions)
        assert len(indices_per_dim) == self.ndim, f'{len(indices_per_dim)=} != {self.ndim=}'
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
        block_indices = (
            np.stack(
                np.meshgrid(*[np.arange(s) for s in subdivisions], indexing='ij'), axis=-1
            ).reshape(-1, self.ndim)
        )

        fine_indices = ((indices * subdivisions)[:, None] + block_indices[None]).reshape(-1, self.ndim)
        return fine_indices

    # def


def oct_tree_marching_cubes(f: callable, step: np.ndarray, bounds: np.ndarray | None = None):
    ndim = 3

    if bounds is None:
        bounds = sdf.mesh._estimate_bounds(f)
    bounds = np.asarray(bounds)

    if isinstance(step, Number):
        step = np.array((step,) * ndim)

    assert np.all(bounds[0] < bounds[1]), f'Bounds must be sorted. Got {bounds=}.'
    assert np.all(step > 0), f'Got negative step size: {step=}'

    grids = [CartesianGrid(ndim=ndim, origin=bounds[0], cell_size=step)]

    box_size = bounds[1] - bounds[0]


    subdivisions = 2
    level = 0
    block_indices = [np.arange(n) for n in box_size // step]

    def block_indices_n_cells(block_indices):
        return np.product([len(indices) for indices in block_indices])

    n_cells_naive = block_indices_n_cells(block_indices)
    logger.info('n_cells_naive=%d', n_cells_naive)

    # coarsen until the whole box fits into a single grid cell
    while block_indices_n_cells(block_indices) > 1:
        logger.debug('level %d, n_cells=%d', level, block_indices_n_cells(block_indices))
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
        logger.debug('level %d, mask.mean=%.3f, mask.sum=%d', len(grids) - 1 - i, mask.mean(), mask.sum())
        indices = indices[mask]

        if i != len(grids) - 1:
            # get indices in finder grid
            indices = grid.subdivide_indices(indices, subdivisions=subdivisions)

    logger.info('n_cells=%d', len(indices))
    return indices, grids[0]



