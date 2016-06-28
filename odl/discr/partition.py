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
from odl.set.domain import IntervalProd
from odl.util.normalize import normalized_index_expression


__all__ = ('RectPartition', 'uniform_partition_fromintv',
           'uniform_partition_fromgrid', 'uniform_partition')

_POINT_POSITIONS = ('center', 'left')


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
        """The partitioned set, an `IntervalProd`."""
        return self.__set

    # IntervalProd related pass-through methods and derived properties
    # min, max and extent are for duck-typing purposes
    @property
    def begin(self):
        """Minimum coordinates of the partitioned set."""
        return self.set.begin

    @property
    def end(self):
        """Maximum coordinates of the partitioned set."""
        return self.set.end

    def min(self):
        """Return the minimum point of the partitioned set.

        See also
        --------
        odl.set.domain.IntervalProd.min
        """
        return self.set.min()

    def max(self):
        """Return the maximum point of the partitioned set.

        See also
        --------
        odl.set.domain.IntervalProd.max
        """
        return self.set.max()

    def extent(self):
        """Return a vector containing the total extent (max - min)."""
        return self.set.extent()

    @property
    def grid(self):
        """The `TensorGrid` defining this partition."""
        return self.__grid

    # TensorGrid related pass-through methods and derived properties
    @property
    def is_uniform(self):
        """Return `True` if ``self.grid`` is a `RegularGrid`."""
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
        """The coordinate vectors of the grid."""
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
        on_bdry : `tuple` of 2-tuple of `float`
            Each 2-tuple contains the fraction of the leftmost
            (first entry) and rightmost (second entry) cell in the
            partitioned set in the corresponding dimension.

        See also
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
                self.grid.coord_vectors, self.set.begin, self.set.end)):
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
        csizes : `tuple` of `numpy.ndarray`
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
        """Side lengths of all 'inner' cells of a regular partition.

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
        """Volume of the 'inner' cells, regardless of begin and end.

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
        """Return `True` in case of approximate equality.

        Returns
        -------
        approx_eq : `bool`
            `True` if ``other`` is a `RectPartition` instance with
            ``self.set == other.set`` up to ``atol`` and
            ``self.grid == other.other`` up to ``atol``.
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
        # Optimized version for exact equality
        return self.set == other.set and self.grid == other.grid

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Object determining which parts of the partition to extract.
            `None` (new axis) and empty axes are not supported.

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
            new_begin = [self.cell_boundary_vecs[0][:-1][indices][0]]
            new_end = [self.cell_boundary_vecs[0][1:][indices][-1]]
            for cvec in self.cell_boundary_vecs[1:]:
                new_begin.append(cvec[0])
                new_end.append(cvec[-1])

            new_intvp = IntervalProd(new_begin, new_end)
            new_grid = self.grid[indices]
            return RectPartition(new_intvp, new_grid)

        indices = normalized_index_expression(indices, self.shape,
                                              int_to_slice=True)
        # Build the new partition
        new_begin, new_end = [], []
        for cvec, idx in zip(self.cell_boundary_vecs, indices):
            # Determine the subinterval begin and end vectors. Take the
            # first begin as new begin and the last end as new end.
            sub_begin = cvec[:-1][idx]
            sub_end = cvec[1:][idx]
            new_begin.append(sub_begin[0])
            new_end.append(sub_end[-1])

        new_intvp = IntervalProd(new_begin, new_end)
        new_grid = self.grid[indices]
        return RectPartition(new_intvp, new_grid)

    def insert(self, index, other):
        """Return a copy with ``other`` inserted before ``index``.

        Parameters
        ----------
        index : `int`
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
            The set to be inserted.

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
                inner_str = '{}, {}, {}'.format(float(self.set.begin),
                                                float(self.set.end),
                                                self.size)
            else:
                inner_str = '{}, {}, {}'.format(list(self.set.begin),
                                                list(self.set.end),
                                                list(self.shape))
            return 'uniform_partition({})'.format(inner_str)
        else:
            inner_str = '\n    {!r},\n    {!r}'.format(self.set, self.grid)
            return '{}({})'.format(self.__class__.__name__, inner_str)


def uniform_partition_fromintv(intv_prod, num_nodes, nodes_on_bdry=False):
    """Return a partition of an interval product into equally sized cells.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Interval product to be partitioned
    num_nodes : `int` or `sequence` of `int`
        Number of nodes per axis. For 1d intervals, a single integer
        can be specified.
    nodes_on_bdry : `bool` or `sequence`, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (True) or shift it
        by half a cell size into the interior (False). In each axis,
        an entry may consist in a single `bool` or a 2-tuple of
        `bool`. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

    See also
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

    grid = uniform_sampling_fromintv(intv_prod, num_nodes,
                                     nodes_on_bdry=nodes_on_bdry)

    return RectPartition(intv_prod, grid)


def uniform_partition_fromgrid(grid, begin=None, end=None):
    """Return a partition of an interval product based on a given grid.

    This method is complementary to
    `uniform_partition_fromintv` in that it infers the
    set to be partitioned from a given grid and optional parameters
    for the begin and the end of the set.

    Parameters
    ----------
    grid : `TensorGrid`
        Grid on which the partition is based
    begin, end : `array-like` or `dictionary`
        Spatial points defining the begin and end of an interval
        product to be partitioned. The points can be specified in
        two ways:

        array-like: These values are used directly as begin and/or end.

        dictionary: Index-value pairs specifying an axis and a spatial
        coordinate to be used in that axis. In axes which are not a key
        in the dictionary, the coordinate for the vector is calculated
        as::

            begin = x[0] - (x[1] - x[0]) / 2
            end = x[-1] + (x[-1] - x[-2]) / 2

        See ``Examples`` below.

        In general, ``begin`` may not be larger than ``grid.min_pt``,
        and ``end`` not smaller than ``grid.max_pt`` in any component.
        `None` is equivalent to an empty dictionary, i.e. the values
        are calculated in each dimension.

    See also
    --------
    uniform_partition_fromintv

    Examples
    --------
    Have begin and end of the bounding box automatically calculated:

    >>> grid = RegularGrid(0, 1, 3)
    >>> grid.coord_vectors
    (array([ 0. ,  0.5,  1. ]),)
    >>> part = uniform_partition_fromgrid(grid)
    >>> part.cell_boundary_vecs
    (array([-0.25,  0.25,  0.75,  1.25]),)

    Begin and end can be given explicitly as array-like:

    >>> part = uniform_partition_fromgrid(grid, begin=0, end=1)
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.75,  1.  ]),)

    Using dictionaries, selective axes can be explicitly set. The
    keys refer to axes, the values to the coordinates to use:

    >>> grid = RegularGrid([0, 0], [1, 1], (3, 3))
    >>> part = uniform_partition_fromgrid(grid, begin={0: -1}, end={-1: 3})
    >>> part.cell_boundary_vecs[0]
    array([-1.  ,  0.25,  0.75,  1.25])
    >>> part.cell_boundary_vecs[1]
    array([-0.25,  0.25,  0.75,  3.  ])
    """
    # Make dictionaries from begin and end and fill with None where no value
    # is given.
    if begin is None:
        begin = {i: None for i in range(grid.ndim)}
    elif not hasattr(begin, 'items'):  # array-like
        begin = np.atleast_1d(begin)
        begin = {i: float(v) for i, v in enumerate(begin)}
    else:
        begin.update({i: None for i in range(grid.ndim) if i not in begin})

    if end is None:
        end = {i: None for i in range(grid.ndim)}
    elif not hasattr(end, 'items'):
        end = np.atleast_1d(end)
        end = {i: float(v) for i, v in enumerate(end)}
    else:
        end.update({i: None for i in range(grid.ndim) if i not in end})

    # Set the values in the vectors by computing (None) or directly from the
    # given vectors (otherwise).
    begin_vec = np.empty(grid.ndim)
    for ax, beg_val in begin.items():
        if beg_val is None:
            cvec = grid.coord_vectors[ax]
            if len(cvec) == 1:
                raise ValueError('cannot calculate begin in axis {} with '
                                 'only 1 grid point'.format(ax))
            begin_vec[ax] = cvec[0] - (cvec[1] - cvec[0]) / 2
        else:
            begin_vec[ax] = beg_val

    end_vec = np.empty(grid.ndim)
    for ax, end_val in end.items():
        if end_val is None:
            cvec = grid.coord_vectors[ax]
            if len(cvec) == 1:
                raise ValueError('cannot calculate end in axis {} with '
                                 'only 1 grid point'.format(ax))
            end_vec[ax] = cvec[-1] + (cvec[-1] - cvec[-2]) / 2
        else:
            end_vec[ax] = end_val

    return RectPartition(IntervalProd(begin_vec, end_vec), grid)


def uniform_partition(begin=None, end=None, num_nodes=None,
                      cell_sides=None, nodes_on_bdry=False):
    """Return a partition with equally sized cells.

    Parameters
    ----------
    begin, end : float or array-like, optional
        Vectors defining the begin end end points of an `IntervalProd`
        (a rectangular box). For one-dimensional partitions, single
        floats can be provided. ``None`` entries mean "compute the value".
    num_nodes : int or sequence of int, optional
        Number of nodes per axis. For 1d intervals, a single integer
        can be specified. ``None`` entries mean "compute the value".
    cell_sides : float or array-like, optional
        Side length of the partition cells per axis. For 1d intervals,
        a single integer can be specified. ``None`` entries mean
        "compute the value".
    nodes_on_bdry : `bool` or `sequence`, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (True) or shift it
        by half a cell size into the interior (False). In each axis,
        an entry may consist in a single `bool` or a 2-tuple of
        `bool`. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

    Notes
    -----
    In each axis, 3 of the 4 possible parameters ``begin``, ``end``,
    ``num_nodes`` and ``cell_sides`` must be given. If all four are
    provided, they are checked for consistency.

    See also
    --------
    uniform_partition_fromintv : partition an existing set
    uniform_partition_fromgrid : use an existing grid as basis

    Examples
    --------
    Any combination of three of the four parameters can be used for
    creation of a partition:

    >>> part = uniform_partition(begin=0, end=2, num_nodes=4)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)
    >>> part = uniform_partition(begin=0, num_nodes=4, cell_sides=0.5)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)
    >>> part = uniform_partition(end=2, num_nodes=4, cell_sides=0.5)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)
    >>> part = uniform_partition(begin=0, end=2, cell_sides=0.5)
    >>> part.cell_boundary_vecs
    (array([ 0. ,  0.5,  1. ,  1.5,  2. ]),)

    In higher dimensions, the parameters can be given differently in
    each axis. Where None is given, the value will be computed:

    >>> part = uniform_partition(begin=[0, 0], end=[1, 2],
    ...                          num_nodes=[4, 2])
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]), array([ 0.,  1.,  2.]))
    >>> part = uniform_partition(begin=[0, 0], end=[1, 2],
    ...                          num_nodes=[None, 2],
    ...                          cell_sides=[0.25, None])
    >>> part.cell_boundary_vecs
    (array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]), array([ 0.,  1.,  2.]))
    >>> part = uniform_partition(begin=[0, None], end=[None, 2],
    ...                          num_nodes=[4, 2],
    ...                          cell_sides=[0.25, 1])
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
    sizes = [np.size(p) for p in (begin, end, num_nodes, cell_sides)]
    ndim = int(np.max(sizes))

    if ndim == 1:
        begin = [begin]
        end = [end]
        num_nodes = [num_nodes]
        cell_sides = [cell_sides]
    else:
        if begin is None:
            begin = [None] * ndim
        if end is None:
            end = [None] * ndim
        if num_nodes is None:
            num_nodes = [None] * ndim
        if cell_sides is None:
            cell_sides = [None] * ndim

        begin = list(begin)
        end = list(end)
        num_nodes = list(num_nodes)
        cell_sides = list(cell_sides)
        sizes = [len(p) for p in (begin, end, num_nodes, cell_sides)]

        if not all(s == ndim for s in sizes):
            raise ValueError('inconsistent sizes {}, {}, {}, {} of '
                             'arguments `begin`, `end`, `num_nodes`, '
                             '`cell_sides`'.format(*sizes))

    # Normalize nodes_on_bdry
    if np.shape(nodes_on_bdry) == ():
        nodes_on_bdry = ([(bool(nodes_on_bdry), bool(nodes_on_bdry))] * ndim)
    elif ndim == 1 and len(nodes_on_bdry) == 2:
        nodes_on_bdry = [nodes_on_bdry]
    elif len(nodes_on_bdry) != ndim:
        raise ValueError('nodes_on_bdry has length {}, expected {}.'
                         ''.format(len(nodes_on_bdry), ndim))

    # Calculate the missing parameters in begin, end, num_nodes
    for i, (b, e, n, s, on_bdry) in enumerate(zip(begin, end, num_nodes,
                                                  cell_sides, nodes_on_bdry)):
        num_params = sum(p is not None for p in (b, e, n, s))
        if num_params < 3:
            raise ValueError('in axis {}: expected at least 3 of the '
                             'parameters `begin`, `end`, `num_nodes`, '
                             '`cell_sides`, got {}'
                             ''.format(i, num_params))

        # Unpack the tuple if possible, else use bool globally for this axis
        try:
            bdry_l, bdry_r = on_bdry
        except TypeError:
            bdry_l = bdry_r = on_bdry

        # For each node on the boundary, we subtract 1/2 from the number of
        # full cells between begin and end.
        if b is None:
            begin[i] = e - (n - sum([bdry_l, bdry_r]) / 2.0) * s
        elif e is None:
            end[i] = b + (n - sum([bdry_l, bdry_r]) / 2.0) * s
        elif n is None:
            # Here we add to n since (e-b)/s gives the reduced number of cells.
            n_calc = (e - b) / s + sum([bdry_l, bdry_r]) / 2.0
            n_round = int(round(n_calc))
            if abs(n_calc - n_round) > 1e-5:
                raise ValueError('in axis {}: calculated number of nodes '
                                 '{} = ({} - {}) / {} too far from integer'
                                 ''.format(i, n_calc, e, b, s))
            num_nodes[i] = n_round
        elif s is None:
            pass
        else:
            e_calc = b + (n - sum([bdry_l, bdry_r]) / 2.0) * s
            if not np.isclose(e, e_calc):
                raise ValueError('in axis {}: calculated endpoint '
                                 '{} = {} + {} * {} too far from given '
                                 'endpoint {}.'
                                 ''.format(i, e_calc, b, n, s, e))

    return uniform_partition_fromintv(
        IntervalProd(begin, end), num_nodes, nodes_on_bdry)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
