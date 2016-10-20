# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Partitons of interval products based on tensor grids.

A partition of a set is a finite collection of nonempty, pairwise
disjoint subsets whose union is the original set. The partitions
considered here are based on hypercubes, i.e. the tensor products
of partitions of intervals.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object, range, super, zip

import numpy as np

from odl.discr.grid import TensorGrid, RegularGrid, uniform_sampling_fromintv
from odl.set import IntervalProd
from odl.util.normalize import (
    normalized_index_expression, normalized_nodes_on_bdry,
    normalized_scalar_param_list, safe_int_conv)


__all__ = ('RectPartition', 'uniform_partition_fromintv',
           'uniform_partition_fromgrid', 'uniform_partition',
           'nonuniform_partition')


class RectPartition(object):

    """Rectangular partition by hypercubes based on `TensorGrid`.

    In 1d, a partition of an interval is implicitly defined by a
    collection of points x[0], ..., x[N-1] (a grid) which are chosen to
    lie in the center of the subintervals. The i-th subinterval is thus
    given by

        ``I[i] = [(x[i-1]+x[i])/2, (x[i]+x[i+1])/2]``

    """

    def __init__(self, intv_prod, grid):
        """Initialize a new instance.

        Parameters
        ----------
        intv_prod : `IntervalProd`
            Set to be partitioned
        grid : `TensorGrid`
            Spatial points supporting the partition. They must be
            contained in ``intv_prod``.
        """
        if not isinstance(intv_prod, IntervalProd):
            raise TypeError('{!r} is not an IntervalProd instance'
                            ''.format(intv_prod))
        if not isinstance(grid, TensorGrid):
            raise TypeError('{!r} is not a TensorGrid instance'
                            ''.format(grid))

        # More conclusive error than the one from contains_set
        if intv_prod.ndim != grid.ndim:
            raise ValueError('interval product {} is {}-dimensional while '
                             'grid {} is {}-dimensional'
                             ''.format(intv_prod, intv_prod.ndim,
                                       grid, grid.ndim))

        if not intv_prod.contains_set(grid):
            raise ValueError('{} is not contained in {}'
                             ''.format(grid, intv_prod))

        super().__init__()
        self.__set = intv_prod
        self.__grid = grid

        # Initialize the cell boundaries, the defining property of partitions
        bdry_vecs = []
        for ax, vec in enumerate(self.grid.coord_vectors):
            bdry = np.empty(len(vec) + 1)
            bdry[1:-1] = (vec[1:] + vec[:-1]) / 2.0
            bdry[0] = self.min()[ax]
            bdry[-1] = self.max()[ax]
            bdry_vecs.append(bdry)

        self.__cell_boundary_vecs = tuple(bdry_vecs)

    @property
    def cell_boundary_vecs(self):
        """Return the cell boundaries as coordinate vectors.

        Examples
        --------
        >>> rect = IntervalProd([0, -1], [1, 2])
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(rect, grid)
        >>> part.cell_boundary_vecs
        (array([ 0. ,  0.5,  1. ]), array([-1. , -0.5,  1. ,  2. ]))
        """
        return self.__cell_boundary_vecs

    @property
    def set(self):
        """Partitioned set, an `IntervalProd`."""
        return self.__set

    # IntervalProd related pass-through methods and derived properties
    # min, max and extent are for duck-typing purposes
    @property
    def min_pt(self):
        """Minimum coordinates of the partitioned set."""
        return self.set.min_pt

    @property
    def max_pt(self):
        """Maximum coordinates of the partitioned set."""
        return self.set.max_pt

    @property
    def mid_pt(self):
        """Midpoint of the partitioned set."""
        return self.set.mid_pt

    def min(self):
        """Return the minimum point of the partitioned set.

        See Also
        --------
        odl.set.domain.IntervalProd.min
        """
        return self.set.min()

    def max(self):
        """Return the maximum point of the partitioned set.

        See Also
        --------
        odl.set.domain.IntervalProd.max
        """
        return self.set.max()

    def extent(self):
        """Return a vector containing the total extent (max - min)."""
        return self.set.extent()

    @property
    def grid(self):
        """`TensorGrid` defining this partition."""
        return self.__grid

    # TensorGrid related pass-through methods and derived properties
    @property
    def is_uniform(self):
        """``True`` if ``self.grid`` is a `RegularGrid`."""
        return isinstance(self.grid, RegularGrid)

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.grid.ndim

    @property
    def shape(self):
        """Number of cells per axis, equal to ``self.grid.shape``."""
        return self.grid.shape

    @property
    def size(self):
        """Total number of cells, equal to ``self.grid.size``."""
        return self.grid.size

    def __len__(self):
        """Return ``len(self)``.

        Total number of cells along the first dimension.

        Examples
        --------
        >>> partition = uniform_partition([0, 0, 0], [1, 1, 1], [2, 3, 4])
        >>> len(partition)
        2

        See Also
        --------
        size : The total number of cells.
        """
        return len(self.grid)

    def points(self):
        """Return the sampling grid points."""
        return self.grid.points()

    @property
    def meshgrid(self):
        """Return the sparse meshgrid of sampling points."""
        return self.grid.meshgrid

    @property
    def coord_vectors(self):
        """Coordinate vectors of the grid."""
        return self.grid.coord_vectors

    # Further derived methods / properties
    @property
    def boundary_cell_fractions(self):
        """Return a tuple of contained fractions of boundary cells.

        Since the outermost grid points can have any distance to the
        boundary of the partitioned set, the "natural" outermost cell
        around these points can either be cropped or extended. This
        property is a tuple of (float, float) tuples, one entry per
        dimension, where the fractions of the left- and rightmost
        cells inside the set are stored. If a grid point lies exactly
        on the boundary, the value is 1/2 since the cell is cut in half.
        Otherwise, any value larger than 1/2 is possible.

        Returns
        -------
        on_bdry : tuple of 2-tuples of floats
            Each 2-tuple contains the fraction of the leftmost
            (first entry) and rightmost (second entry) cell in the
            partitioned set in the corresponding dimension.

        See Also
        --------
        cell_boundary_vecs

        Examples
        --------
        We create a partition of the rectangle [0, 1.5] x [-2, 2] with
        the grid points [0, 1] x [-1, 0, 2]. The "natural" cells at the
        boundary would be:

            [-0.5, 0.5] and [0.5, 1.5] in the first axis

            [-1.5, -0.5] and [1, 3] in the second axis

        Thus, in the first axis, the fractions contained in [0, 1.5]
        are 0.5 and 1, and in the second axis, [-2, 2] contains the
        fractions 1.5 and 0.5.

        >>> rect = IntervalProd([0, -2], [1.5, 2])
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(rect, grid)
        >>> part.boundary_cell_fractions
        ((0.5, 1.0), (1.5, 0.5))
        """
        frac_list = []
        for ax, (cvec, bmin, bmax) in enumerate(zip(
                self.grid.coord_vectors, self.set.min_pt, self.set.max_pt)):
            # Degenerate axes have a value of 1.0 (this is used as weight
            # in integration formulas later)
            if len(cvec) == 1:
                frac_list.append((1.0, 1.0))
            else:
                left_frac = 0.5 + (cvec[0] - bmin) / (cvec[1] - cvec[0])
                right_frac = 0.5 + (bmax - cvec[-1]) / (cvec[-1] - cvec[-2])
                frac_list.append((left_frac, right_frac))

        return tuple(frac_list)

    @property
    def cell_sizes_vecs(self):
        """Return the cell sizes as coordinate vectors.

        Returns
        -------
        csizes : tuple of `numpy.ndarray`'s
            The cell sizes per axis. The length of the vectors is the
            same as the corresponding ``grid.coord_vectors``.
            For axes with 1 grid point, cell size is set to 0.0.

        Examples
        --------
        We create a partition of the rectangle [0, 1] x [-1, 2] into
        2 x 3 cells with the grid points [0, 1] x [-1, 0, 2]. This
        implies that the cell boundaries are given as
        [0, 0.5, 1] x [-1, -0.5, 1, 2], hence the cell size vectors
        are [0.5, 0.5] x [0.5, 1.5, 1]:

        >>> rect = IntervalProd([0, -1], [1, 2])
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(rect, grid)
        >>> part.cell_boundary_vecs
        (array([ 0. ,  0.5,  1. ]), array([-1. , -0.5,  1. ,  2. ]))
        >>> part.cell_sizes_vecs
        (array([ 0.5,  0.5]), array([ 0.5,  1.5,  1. ]))
        """
        csizes = []
        for ax, cvec in enumerate(self.grid.coord_vectors):
            if len(cvec) == 1:
                csizes.append(np.array([0.0]))
            else:
                csize = np.empty_like(cvec)
                csize[1:-1] = (cvec[2:] - cvec[:-2]) / 2.0
                csize[0] = (cvec[0] + cvec[1]) / 2 - self.min()[ax]
                csize[-1] = self.max()[ax] - (cvec[-2] + cvec[-1]) / 2
                csizes.append(csize)

        return tuple(csizes)

    @property
    def cell_sides(self):
        """Side lengths of all 'inner' cells of a uniform partition.

        Only defined if ``self.grid`` is a `RegularGrid`.

        Examples
        --------
        We create a partition of the rectangle [0, 1] x [-1, 2] into
        3 x 3 cells, where the grid points lie on the boundary. This
        means that the grid points are [0, 0.5, 1] x [-1, 0.5, 2],
        i.e. the inner cell has side lengths 0.5 x 1.5:

        >>> rect = IntervalProd([0, -1], [1, 2])
        >>> grid = RegularGrid([0, -1], [1, 2], (3, 3))
        >>> part = RectPartition(rect, grid)
        >>> part.cell_sides
        array([ 0.5,  1.5])
        """
        if not self.is_uniform:
            raise NotImplementedError(
                'cell sides not defined for irregular partitions. Use '
                '`cell_sizes_vecs()` instead')

        sides = self.grid.stride
        sides[sides == 0] = self.extent()[sides == 0]
        return sides

    @property
    def cell_volume(self):
        """Volume of the 'inner' cells of a uniform partition.

        Only defined if ``self.grid`` is a `RegularGrid`.

        Examples
        --------
        We create a partition of the rectangle [0, 1] x [-1, 2] into
        3 x 3 cells, where the grid points lie on the boundary. This
        means that the grid points are [0, 0.5, 1] x [-1, 0.5, 2],
        i.e. the inner cell has side lengths 0.5 x 1.5:

        >>> rect = IntervalProd([0, -1], [1, 2])
        >>> grid = RegularGrid([0, -1], [1, 2], (3, 3))
        >>> part = RectPartition(rect, grid)
        >>> part.cell_sides
        array([ 0.5,  1.5])
        >>> part.cell_volume
        0.75
        """
        return float(np.prod(self.cell_sides))

    def approx_equals(self, other, atol):
        """Return ``True`` in case of approximate equality.

        Returns
        -------
        approx_eq : bool
            ``True`` if ``other`` is a `RectPartition` instance with
            ``self.set == other.set`` up to ``atol`` and
            ``self.grid == other.other`` up to ``atol``, ``False`` otherwise.
        """
        if other is self:
            return True
        elif not isinstance(other, RectPartition):
            return False
        else:
            return (self.set.approx_equals(other.set, atol=atol) and
                    self.grid.approx_equals(other.grid, atol=atol))

    def __eq__(self, other):
        """Return ``self == other``."""
        # Implemented separately for performance reasons
        if other is self:
            return True

        # Optimized version for exact equality
        return self.set == other.set and self.grid == other.grid

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Object determining which parts of the partition to extract.
            ``None`` (new axis) and empty axes are not supported.

        Examples
        --------
        >>> intvp = IntervalProd([-1, 1, 4, 2], [3, 6, 5, 7])
        >>> grid = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> part = RectPartition(intvp, grid)
        >>> part.cell_boundary_vecs
        (array([-1. , -0.5,  1.5,  3. ]),
         array([ 1.,  3.,  6.]),
         array([ 4.,  5.]),
         array([ 2. ,  3. ,  5.5,  7. ]))

        Indexing picks out sub-intervals (compare with the boundary
        vectors):

        >>> part[0, 0, 0, 0]
        RectPartition(
            IntervalProd([-1.0, 1.0, 4.0, 2.0], [-0.5, 3.0, 5.0, 3.0]),
            TensorGrid([-1.0], [2.0], [5.0], [2.0]))

        Taking an advanced slice (every second along the first axis,
        the last in the last axis and everything in between):

        >>> part[::2, ..., -1]
        RectPartition(
            IntervalProd([-1.0, 1.0, 4.0, 5.5], [3.0, 6.0, 5.0, 7.0]),
            TensorGrid([-1.0, 3.0], [2.0, 4.0], [5.0], [7.0]))

        Too few indices are filled up with an ellipsis from the right:

        >>> part[1]
        RectPartition(
            IntervalProd([-0.5, 1.0, 4.0, 2.0], [1.5, 6.0, 5.0, 7.0]),
            TensorGrid([0.0], [2.0, 4.0], [5.0], [2.0, 4.0, 7.0]))
        """
        # Special case of index list: slice along first axis
        if isinstance(indices, list):
            new_min_pt = [self.cell_boundary_vecs[0][:-1][indices][0]]
            new_max_pt = [self.cell_boundary_vecs[0][1:][indices][-1]]
            for cvec in self.cell_boundary_vecs[1:]:
                new_min_pt.append(cvec[0])
                new_max_pt.append(cvec[-1])

            new_intvp = IntervalProd(new_min_pt, new_max_pt)
            new_grid = self.grid[indices]
            return RectPartition(new_intvp, new_grid)

        indices = normalized_index_expression(indices, self.shape,
                                              int_to_slice=True)
        # Build the new partition
        new_min_pt, new_max_pt = [], []
        for cvec, idx in zip(self.cell_boundary_vecs, indices):
            # Determine the subinterval min_pt and max_pt vectors. Take the
            # first min_pt as new min_pt and the last max_pt as new max_pt.
            sub_min_pt = cvec[:-1][idx]
            sub_max_pt = cvec[1:][idx]
            new_min_pt.append(sub_min_pt[0])
            new_max_pt.append(sub_max_pt[-1])

        new_intvp = IntervalProd(new_min_pt, new_max_pt)
        new_grid = self.grid[indices]
        return RectPartition(new_intvp, new_grid)

    def insert(self, index, other):
        """Return a copy with ``other`` inserted before ``index``.

        Parameters
        ----------
        index : int
            Index of the dimension before which ``other`` is to
            be inserted. Negative indices count backwards from
            ``self.ndim``.
        other : `RectPartition`
            Partition to be inserted

        Returns
        -------
        newpart : `RectPartition`
            Partition with the inserted other partition

        Examples
        --------
        >>> part1 = uniform_partition([0, -1], [1, 2], (3, 3))
        >>> part2 = uniform_partition(0, 1, 5)
        >>> part1.insert(1, part2)
        uniform_partition([0.0, 0.0, -1.0], [1.0, 1.0, 2.0], [3, 5, 3])

        See Also
        --------
        append
        """
        newgrid = self.grid.insert(index, other.grid)
        newset = self.set.insert(index, other.set)
        return RectPartition(newset, newgrid)

    def append(self, other):
        """Insert at the end.

        Parameters
        ----------
        other : `RectPartition`,
            Set to be inserted.

        Examples
        --------
        >>> part1 = uniform_partition([0, -1], [1, 2], (3, 3))
        >>> part2 = uniform_partition(0, 1, 5)
        >>> part1.append(part2)
        uniform_partition([0.0, -1.0, 0.0], [1.0, 2.0, 1.0], [3, 3, 5])

        See Also
        --------
        insert
        """
        return self.insert(self.ndim, other)

    def squeeze(self):
        """Return the partition with removed degenerate (length 1) dimensions.

        Returns
        -------
        squeezed : `RectPartition`
            Squeezed partition.

        Examples
        --------
        >>> p = uniform_partition([0, -1], [1, 2], (3, 1))
        >>> p.squeeze()
        uniform_partition(0.0, 1.0, 3)

        Notes
        -----
        This is not equivalent to
        ``RectPartiton(self.set.squeeze(), self.grid.squeeze())`` since the
        definition of degenerate is different in sets and grids. This method
        follow the definition used in grids, that is, an axis is degenerate if
        it has only one element.

        See Also
        --------
        TensorGrid.squeeze
        IntervalProd.squeeze
        """
        newset = self.set[self.grid.inondeg]
        return RectPartition(newset, self.grid.squeeze())

    def __str__(self):
        """Return ``str(self)``."""
        return 'partition of {} using {}'.format(self.set, self.grid)

    def __repr__(self):
        """Return ``repr(self)``."""
        if uniform_partition_fromintv(self.set, self.shape) == self:

            if self.ndim == 1:
                inner_str = '{}, {}, {}'.format(float(self.set.min_pt),
                                                float(self.set.max_pt),
                                                self.size)
            else:
                inner_str = '{}, {}, {}'.format(list(self.set.min_pt),
                                                list(self.set.max_pt),
                                                list(self.shape))
            return 'uniform_partition({})'.format(inner_str)
        else:
            inner_str = '\n    {!r},\n    {!r}'.format(self.set, self.grid)
            return '{}({})'.format(self.__class__.__name__, inner_str)


def uniform_partition_fromintv(intv_prod, shape, nodes_on_bdry=False):
    """Return a partition of an interval product into equally sized cells.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Interval product to be partitioned
    shape : int or sequence of ints
        Number of nodes per axis. For 1d intervals, a single integer
        can be specified.
    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``intv_prod.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

    See Also
    --------
    uniform_partition_fromgrid

    Examples
    --------
    By default, no grid points are placed on the boundary:

    >>> interval = IntervalProd(0, 1)
    >>> part = uniform_partition_fromintv(interval, 4)
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]),)
    >>> part.grid.coord_vectors
    (array([ 0.125,  0.375,  0.625,  0.875]),)

    This can be changed with the nodes_on_bdry parameter:

    >>> part = uniform_partition_fromintv(interval, 3, nodes_on_bdry=True)
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.75,  1.  ]),)
    >>> part.grid.coord_vectors
    (array([ 0. ,  0.5,  1. ]),)

    We can specify this per axis, too. In this case we choose both
    in the first axis and only the rightmost in the second:

    >>> rect = IntervalProd([0, 0], [1, 1])
    >>> part = uniform_partition_fromintv(
    ...     rect, (3, 3), nodes_on_bdry=(True, (False, True)))
    ...
    >>> part.cell_boundary_vecs[0]  # first axis, as above
    array([ 0.  ,  0.25,  0.75,  1.  ])
    >>> part.grid.coord_vectors[0]
    array([ 0. ,  0.5,  1. ])
    >>> part.cell_boundary_vecs[1]  # second, asymmetric axis
    array([ 0. ,  0.4,  0.8,  1. ])
    >>> part.grid.coord_vectors[1]
    array([ 0.2,  0.6,  1. ])
    """

    grid = uniform_sampling_fromintv(intv_prod, shape,
                                     nodes_on_bdry=nodes_on_bdry)

    return RectPartition(intv_prod, grid)


def uniform_partition_fromgrid(grid, min_pt=None, max_pt=None):
    """Return a partition of an interval product based on a given grid.

    This method is complementary to `uniform_partition_fromintv` in that
    it infers the set to be partitioned from a given grid and optional
    parameters for ``min_pt`` and ``max_pt`` of the set.

    Parameters
    ----------
    grid : `TensorGrid`
        Grid on which the partition is based
    min_pt, max_pt : float, sequence of floats, or dict
        Spatial points defining the lower/upper limits of the intervals
        to be partitioned. The points can be specified in two ways:

        float or sequence: The values are used directly as ``min_pt``
        and/or ``max_pt``.

        dict: Index-value pairs specifying an axis and a spatial
        coordinate to be used in that axis. In axes which are not a key
        in the dictionary, the coordinate for the vector is calculated
        as::

            min_pt = x[0] - (x[1] - x[0]) / 2
            max_pt = x[-1] + (x[-1] - x[-2]) / 2

        See ``Examples`` below.

        In general, ``min_pt`` may not be larger than ``grid.min_pt``,
        and ``max_pt`` not smaller than ``grid.max_pt`` in any component.
        ``None`` is equivalent to an empty dictionary, i.e. the values
        are calculated in each dimension.

    See Also
    --------
    uniform_partition_fromintv

    Examples
    --------
    Have ``min_pt`` and ``max_pt`` of the bounding box automatically
    calculated:

    >>> grid = RegularGrid(0, 1, 3)
    >>> grid.coord_vectors
    (array([ 0. ,  0.5,  1. ]),)
    >>> part = uniform_partition_fromgrid(grid)
    >>> part.cell_boundary_vecs
    (array([-0.25,  0.25,  0.75,  1.25]),)

    ``min_pt`` and ``max_pt`` can be given explicitly:

    >>> part = uniform_partition_fromgrid(grid, min_pt=0, max_pt=1)
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.75,  1.  ]),)

    Using dictionaries, selective axes can be explicitly set. The
    keys refer to axes, the values to the coordinates to use:

    >>> grid = RegularGrid([0, 0], [1, 1], (3, 3))
    >>> part = uniform_partition_fromgrid(grid, min_pt={0: -1}, max_pt={-1: 3})
    >>> part.cell_boundary_vecs[0]
    array([-1.  ,  0.25,  0.75,  1.25])
    >>> part.cell_boundary_vecs[1]
    array([-0.25,  0.25,  0.75,  3.  ])
    """
    # Make dictionaries from min_pt and max_pt and fill with None where no
    # value is given.
    if min_pt is None:
        min_pt = {i: None for i in range(grid.ndim)}
    elif not hasattr(min_pt, 'items'):  # array-like
        min_pt = np.atleast_1d(min_pt)
        min_pt = {i: float(v) for i, v in enumerate(min_pt)}
    else:
        min_pt.update({i: None for i in range(grid.ndim) if i not in min_pt})

    if max_pt is None:
        max_pt = {i: None for i in range(grid.ndim)}
    elif not hasattr(max_pt, 'items'):
        max_pt = np.atleast_1d(max_pt)
        max_pt = {i: float(v) for i, v in enumerate(max_pt)}
    else:
        max_pt.update({i: None for i in range(grid.ndim) if i not in max_pt})

    # Set the values in the vectors by computing (None) or directly from the
    # given vectors (otherwise).
    min_pt_vec = np.empty(grid.ndim)
    for ax, xmin in min_pt.items():
        if xmin is None:
            cvec = grid.coord_vectors[ax]
            if len(cvec) == 1:
                raise ValueError('in axis {}: cannot calculate `min_pt` with '
                                 'only 1 grid point'.format(ax))
            min_pt_vec[ax] = cvec[0] - (cvec[1] - cvec[0]) / 2
        else:
            min_pt_vec[ax] = xmin

    max_pt_vec = np.empty(grid.ndim)
    for ax, xmax in max_pt.items():
        if xmax is None:
            cvec = grid.coord_vectors[ax]
            if len(cvec) == 1:
                raise ValueError('in axis {}: cannot calculate `max_pt` with '
                                 'only 1 grid point'.format(ax))
            max_pt_vec[ax] = cvec[-1] + (cvec[-1] - cvec[-2]) / 2
        else:
            max_pt_vec[ax] = xmax

    return RectPartition(IntervalProd(min_pt_vec, max_pt_vec), grid)


def uniform_partition(min_pt=None, max_pt=None, shape=None, cell_sides=None,
                      nodes_on_bdry=False):
    """Return a partition with equally sized cells.

    Parameters
    ----------
    min_pt, max_pt : float or sequence of float, optional
        Vectors defining the lower/upper limits of the intervals in an
        `IntervalProd` (a rectangular box). ``None`` entries mean
        "compute the value".
    shape : int or sequence of ints, optional
        Number of nodes per axis. ``None`` entries mean
        "compute the value".
    cell_sides : float or sequence of floats, optional
        Side length of the partition cells per axis. ``None`` entries mean
        "compute the value".
    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

    Notes
    -----
    In each axis, 3 of the 4 possible parameters ``min_pt``, ``max_pt``,
    ``shape`` and ``cell_sides`` must be given. If all four are
    provided, they are checked for consistency.

    See Also
    --------
    uniform_partition_fromintv : partition an existing set
    uniform_partition_fromgrid : use an existing grid as basis

    Examples
    --------
    Any combination of three of the four parameters can be used for
    creation of a partition:

    >>> part = uniform_partition(min_pt=0, max_pt=2, shape=4)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)
    >>> part = uniform_partition(min_pt=0, shape=4, cell_sides=0.5)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)
    >>> part = uniform_partition(max_pt=2, shape=4, cell_sides=0.5)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)
    >>> part = uniform_partition(min_pt=0, max_pt=2, cell_sides=0.5)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)

    In higher dimensions, the parameters can be given differently in
    each axis. Where ``None`` is given, the value will be computed:

    >>> part = uniform_partition(min_pt=[0, 0], max_pt=[1, 2], shape=[4, 2])
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]), array([ 0.,  1.,  2.]))
    >>> part = uniform_partition(min_pt=[0, 0], max_pt=[1, 2],
    ...                          shape=[None, 2], cell_sides=[0.25, None])
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]), array([ 0.,  1.,  2.]))
    >>> part = uniform_partition(min_pt=[0, None], max_pt=[None, 2],
    ...                          shape=[4, 2], cell_sides=[0.25, 1])
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]), array([ 0.,  1.,  2.]))

    By default, no grid points are placed on the boundary:

    >>> part = uniform_partition(0, 1, 4)
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]),)
    >>> part.grid.coord_vectors
    (array([ 0.125,  0.375,  0.625,  0.875]),)

    This can be changed with the nodes_on_bdry parameter:

    >>> part = uniform_partition(0, 1, 3, nodes_on_bdry=True)
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.75,  1.  ]),)
    >>> part.grid.coord_vectors
    (array([ 0. ,  0.5,  1. ]),)

    We can specify this per axis, too. In this case we choose both
    in the first axis and only the rightmost in the second:

    >>> part = uniform_partition([0, 0], [1, 1], (3, 3),
    ...                          nodes_on_bdry=(True, (False, True)))
    ...
    >>> part.cell_boundary_vecs[0]  # first axis, as above
    array([ 0.  ,  0.25,  0.75,  1.  ])
    >>> part.grid.coord_vectors[0]
    array([ 0. ,  0.5,  1. ])
    >>> part.cell_boundary_vecs[1]  # second, asymmetric axis
    array([ 0. ,  0.4,  0.8,  1. ])
    >>> part.grid.coord_vectors[1]
    array([ 0.2,  0.6,  1. ])
    """
    # Normalize partition parameters

    # np.size(None) == 1
    sizes = [np.size(p) for p in (min_pt, max_pt, shape, cell_sides)]
    ndim = int(np.max(sizes))

    min_pt = normalized_scalar_param_list(min_pt, ndim, param_conv=float,
                                          keep_none=True)
    max_pt = normalized_scalar_param_list(max_pt, ndim, param_conv=float,
                                          keep_none=True)
    shape = normalized_scalar_param_list(shape, ndim, param_conv=safe_int_conv,
                                         keep_none=True)
    cell_sides = normalized_scalar_param_list(cell_sides, ndim,
                                              param_conv=float, keep_none=True)

    nodes_on_bdry = normalized_nodes_on_bdry(nodes_on_bdry, ndim)

    # Calculate the missing parameters in min_pt, max_pt, shape
    for i, (xmin, xmax, n, dx, on_bdry) in enumerate(
            zip(min_pt, max_pt, shape, cell_sides, nodes_on_bdry)):
        num_params = sum(p is not None for p in (xmin, xmax, n, dx))
        if num_params < 3:
            raise ValueError('in axis {}: expected at least 3 of the '
                             'parameters `min_pt`, `max_pt`, `shape`, '
                             '`cell_sides`, got {}'
                             ''.format(i, num_params))

        # Unpack the tuple if possible, else use bool globally for this axis
        try:
            bdry_l, bdry_r = on_bdry
        except TypeError:
            bdry_l = bdry_r = on_bdry

        # For each node on the boundary, we subtract 1/2 from the number of
        # full cells between min_pt and max_pt.
        if xmin is None:
            min_pt[i] = xmax - (n - sum([bdry_l, bdry_r]) / 2.0) * dx
        elif xmax is None:
            max_pt[i] = xmin + (n - sum([bdry_l, bdry_r]) / 2.0) * dx
        elif n is None:
            # Here we add to n since (e-b)/s gives the reduced number of cells.
            n_calc = (xmax - xmin) / dx + sum([bdry_l, bdry_r]) / 2.0
            n_round = int(round(n_calc))
            if abs(n_calc - n_round) > 1e-5:
                raise ValueError('in axis {}: calculated number of nodes '
                                 '{} = ({} - {}) / {} too far from integer'
                                 ''.format(i, n_calc, xmax, xmin, dx))
            shape[i] = n_round
        elif dx is None:
            pass
        else:
            xmax_calc = xmin + (n - sum([bdry_l, bdry_r]) / 2.0) * dx
            if not np.isclose(xmax, xmax_calc):
                raise ValueError('in axis {}: calculated endpoint '
                                 '{} = {} + {} * {} too far from given '
                                 'endpoint {}.'
                                 ''.format(i, xmax_calc, xmin, n, dx, xmax))

    return uniform_partition_fromintv(
        IntervalProd(min_pt, max_pt), shape, nodes_on_bdry)


def nonuniform_partition(*coord_vecs, **kwargs):
    """Return a partition with un-equally sized cells.

    Parameters
    ----------
    coord_vecs1, ... coord_vecsN : `array-like`
        Arrays of coordinates of the mid-points of the partition cells.
    min_pt, max_pt : float or sequence of floats, optional
        Vectors defining the lower/upper limits of the intervals in an
        `IntervalProd` (a rectangular box). ``None`` entries mean
        "compute the value".
    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

        Cannot be given with both min_pt and max_pt since they determine the
        same thing.

    See Also
    --------
    uniform_partition : uniformly spaced points
    uniform_partition_fromintv : partition an existing set
    uniform_partition_fromgrid : use an existing grid as basis

    Examples
    --------
    With uniformly spaced points the result is the same as a uniform partition:

    >>> nonuniform_partition([0, 1, 2, 3])
    uniform_partition(-0.5, 3.5, 4)
    >>> nonuniform_partition([0, 1, 2, 3], [1, 2])
    uniform_partition([-0.5, 0.5], [3.5, 2.5], [4, 2])

    If the points are not uniformly spaced a nonuniform partition is created.
    Note that the containing interval is calculated by assuming that the points
    are in the middle of the sub-intervals:

    >>> nonuniform_partition([0, 1, 3])
    RectPartition(
        IntervalProd(-0.5, 4.0),
        TensorGrid([0.0, 1.0, 3.0]))

    Higher dimensional partitions are created by specifying the gridpoints
    along each dimension:

    >>> nonuniform_partition([0, 1, 3], [1, 2])
    RectPartition(
        IntervalProd([-0.5, 0.5], [4.0, 2.5]),
        TensorGrid([0.0, 1.0, 3.0], [1.0, 2.0]))

    If the endpoints should be on the boundary, the ``nodes_on_bdry`` parameter
    can be used:

    >>> nonuniform_partition([0, 1, 3], nodes_on_bdry=True)
    RectPartition(
        IntervalProd(0.0, 3.0),
        TensorGrid([0.0, 1.0, 3.0]))

    Users can also manually specify the containing intervals dimensions by
    using the ``min_pt`` and ``max_pt`` arguments:

    >>> nonuniform_partition([0, 1, 3], min_pt=-2, max_pt=3)
    RectPartition(
        IntervalProd(-2.0, 3.0),
        TensorGrid([0.0, 1.0, 3.0]))
    """
    # Get parameters from kwargs
    min_pt = kwargs.pop('min_pt', None)
    max_pt = kwargs.pop('max_pt', None)
    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)

    # np.size(None) == 1
    sizes = [len(coord_vecs)] + [np.size(p) for p in (min_pt, max_pt)]
    ndim = int(np.max(sizes))

    min_pt = normalized_scalar_param_list(min_pt, ndim, param_conv=float,
                                          keep_none=True)
    max_pt = normalized_scalar_param_list(max_pt, ndim, param_conv=float,
                                          keep_none=True)
    nodes_on_bdry = normalized_nodes_on_bdry(nodes_on_bdry, ndim)

    # Calculate the missing parameters in min_pt, max_pt
    for i, (xmin, xmax, (bdry_l, bdry_r), coords) in enumerate(
            zip(min_pt, max_pt, nodes_on_bdry, coord_vecs)):
        # Check input for redundancy
        if xmin is not None and bdry_l:
            raise ValueError('in axis {}: got both `min_pt` and '
                             '`nodes_on_bdry=True`'.format(i))
        if xmax is not None and bdry_r:
            raise ValueError('in axis {}: got both `max_pt` and '
                             '`nodes_on_bdry=True`'.format(i))

        # Compute boundary position if not given by user
        if xmin is None:
            if bdry_l:
                min_pt[i] = coords[0]
            else:
                min_pt[i] = coords[0] - (coords[1] - coords[0]) / 2.0
        if xmax is None:
            if bdry_r:
                max_pt[i] = coords[-1]
            else:
                max_pt[i] = coords[-1] + (coords[-1] - coords[-2]) / 2.0

    interval = IntervalProd(min_pt, max_pt)
    grid = TensorGrid(*coord_vecs)
    return RectPartition(interval, grid)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
