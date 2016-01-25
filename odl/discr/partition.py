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

# External module imports
import numpy as np

# ODL imports
from odl.discr.grid import TensorGrid, RegularGrid
from odl.set.domain import IntervalProd
from odl.util.utility import array1d_repr


__all__ = ('RectPartition',)

_POINT_POSITIONS = ('center', 'left')


class RectPartition(object):

    """Rectangular partition by hypercubes based on `TensorGrid`s.

    In 1d, a partition of an interval is implicitly defined by a
    collection of points x[0], ..., x[N-1] (a grid) which are chosen to
    lie in the center of the subintervals. The i-th subinterval is thus
    given by
    ::
        I[i] = [(x[i-1]+x[i])/2, (x[i]+x[i+1])/2]
    """

    def __init__(self, grid, begin=None, end=None, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        grid : `TensorGrid`
            Spatial points supporting the partition
        begin, end : array-like, shape ``(grid.ndim,)``
            Spatial points defining the begin and end of an interval
            product to be partitioned. ``begin`` must be component-wise
            at most ``grid.min_pt``, and ``end`` at least
            ``grid.max_pt``. If not provided, they are inferred from
            the grid.
        begin_axes, end_axes : sequence of `int`, optional
            Dimensions in which to apply ``begin`` and ``end``. If
            given, only these dimensions in ``begin`` or ``end``, resp.,
            are considered - other entries are ignored.

        Notes
        -----
        The endpoints ``begin`` and ``end`` are determined from the grid
        as follows::

            begin = x[0] - (x[1] - x[0]) / 2
            end = x[N-1] + (x[N-1] - x[N-2]) / 2
        """
        if not isinstance(grid, TensorGrid):
            raise TypeError('{!r} is not a TensorGrid instance.'
                            ''.format(grid))

        # Begin and end: check correct shapes and values if given,
        # calculate otherwise.
        begin_axes = kwargs.pop('begin_axes', None)
        end_axes = kwargs.pop('end_axes', None)

        begin = self._calc_begin_end(grid, begin_axes, begin, 'begin')
        end = self._calc_begin_end(grid, end_axes, end, 'end')

        super().__init__()
        self._bbox = IntervalProd(begin, end)
        self._grid = grid

    def _calc_begin_end(self, grid, vec_axes, vector, which):
        """Return the interval product begin/end from ``grid``.

        Parameters
        ----------
        grid : `TensorGrid`
            Grid from which to infer
        vec_axes : sequence of `int`
            Dimensions in which to take the values from ``vector``
        vector : array-like, shape ``(grid.ndim,)``
            Array from which to take the values corresponding to the
            dimensions in ``vec_axes``
        which : {'begin', 'end'}
            Which one of the endpoints to calculate

        Returns
        -------
        newvec : `numpy.ndarray`
            The new begin or end vector, depending on ``which``
        """
        # Check axes sanity
        if vector is None:
            vector = np.zeros(grid.ndim)
            if vec_axes is None:
                vec_axes = []
            else:
                raise ValueError('{which}_axes cannot be given without '
                                 '{which} parameter.'.format(which=which))
        elif vec_axes is None:
            vec_axes = list(range(grid.ndim))

        # Normalize negative values and check for errors
        vec_axes = [i if i >= 0 else grid.ndim + i
                    for i in vec_axes]
        if any(i < 0 or i >= grid.ndim for i in vec_axes):
            raise IndexError('{}_axes {} contains out-ouf-bounds indices.'
                             ''.format(which, vec_axes))

        if len(set(vec_axes)) != len(vec_axes):
            raise ValueError('{}_axes {} contains duplicate indices.'
                             ''.format(which, vec_axes))

        # Set private attribute to be used in __repr__
        # TODO: find good way to handle this without running into slight
        # rounding errors. Best would be to determine in __repr__ if
        # begin/end are as calculated, but it may be better to store if
        # begin/end was given and just always add it to the __repr__ output
        # in that case, regardless of other factors
        if which == 'begin':
            if vec_axes:
                self._custom_begin = True
            else:
                self._custom_begin = False
        else:
            if vec_axes:
                self._custom_end = True
            else:
                self._custom_end = False

        # The other axes
        calc_axes = [i for i in range(grid.ndim) if i not in vec_axes]

        # Check vector sanity (shape and bounds in relevant axes)
        vector = np.asarray(vector, dtype='float64')
        if vector.shape != (grid.ndim,):
            raise ValueError('{} has shape {}, expected {}.'
                             ''.format(which, vector.shape, (grid.ndim,)))
        if (which == 'begin' and
                np.any(vector[vec_axes] > grid.min_pt[vec_axes])):
            raise ValueError('begin {} has entries larger than the '
                             'minimum grid point {} in one of the axes {}.'
                             ''.format(vector, grid.min_pt, vec_axes))
        elif (which == 'end' and
              np.any(vector[vec_axes] < grid.max_pt[vec_axes])):
            raise ValueError('end {} has entries smaller than the '
                             'maximum grid point {} in one of the axes {}.'
                             ''.format(vector, grid.max_pt, vec_axes))

        # Create the new vector
        new_vec = np.empty(grid.ndim, dtype='float64')
        new_vec[vec_axes] = vector[vec_axes]
        for ax in calc_axes:
            cvec = grid.coord_vectors[ax]
            if len(cvec) == 1:
                raise ValueError(
                    'Degenerate dimension {} requires explicit begin and '
                    'end.'.format(ax))

            if which == 'begin':
                new_vec[ax] = cvec[0] - (cvec[1] - cvec[0]) / 2
            else:
                new_vec[ax] = cvec[-1] + (cvec[-1] - cvec[-2]) / 2

        return new_vec

    @property
    def grid(self):
        """The `TensorGrid` defining this partition."""
        return self._grid

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.grid.ndim

    @property
    def shape(self):
        """Number of cells per axis, equal to ``self.grid.shape``."""
        return self.grid.shape

    @property
    def order(self):
        """Return axis ordering, equal to that of the grid."""
        return self.grid.order

    @property
    def bbox(self):
        """The bounding box, i.e. the `IntervalProd` being partitioned.

        Examples
        --------
        By default, the bounding box is inferred from the grid and
        extends beyond the grid boundaries:

        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(grid)  # Implicit begin and end
        >>> part.bbox
        Rectangle([-0.5, -1.5], [1.5, 3.0])

        This behavior can be changed by explicitly giving begin and
        end:

        >>> part = RectPartition(grid, begin=[0, -1], end=[1, 2])
        >>> part.bbox
        Rectangle([0.0, -1.0], [1.0, 2.0])
        """
        return self._bbox

    @property
    def begin(self):
        """Vector with minimum bounding box coordinates."""
        return self.bbox.begin

    @property
    def end(self):
        """Vector with maximum bounding box coordinates."""
        return self.bbox.end

    def min(self):
        """Return the 'minimum corner' of ``self.interv_prod``.

        Examples
        --------
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(grid)  # Implicit begin and end
        >>> part.min()
        array([-0.5, -1.5])

        >>> part = RectPartition(grid, begin=[0, -1], end=[1, 2])
        >>> part.min()
        array([ 0., -1.])

        See also
        --------
        bbox
        """
        return self.bbox.min()

    def max(self):
        """Return the 'minimum corner' of ``self.interv_prod``.

        Examples
        --------
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(grid)  # Implicit begin and end
        >>> part.max()
        array([ 1.5,  3. ])

        >>> part = RectPartition(grid, begin=[0, -1], end=[1, 2])
        >>> part.max()
        array([ 1.,  2.])

        See also
        --------
        bbox
        """
        return self.bbox.max()

    def extent(self):
        """Return a vector containing the total extent (max - min)."""
        return self.max() - self.min()

    def sampling_points(self):
        """Return the grid sampling points."""
        return self.grid.points()

    def cell_boundaries(self):
        """Return the cell boundaries as coordinate vectors.

        Examples
        --------
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(grid)  # Implicit begin and end
        >>> part.cell_boundaries()
        (array([-0.5,  0.5,  1.5]), array([-1.5, -0.5,  1. ,  3. ]))

        >>> part = RectPartition(grid, begin=[0, -1], end=[1, 2])
        >>> part.cell_boundaries()
        (array([ 0. ,  0.5,  1. ]), array([-1. , -0.5,  1. ,  2. ]))
        """
        bdry_vecs = []
        for ax, vec in enumerate(self.grid.coord_vectors):
            bdry = np.empty(len(vec) + 1)
            bdry[1:-1] = (vec[1:] + vec[:-1]) / 2.0
            bdry[0] = self.min()[ax]
            bdry[-1] = self.max()[ax]
            bdry_vecs.append(bdry)

        return tuple(bdry_vecs)

    def cell_sizes(self):
        """Return the cell sizes as coordinate vectors.

        Returns
        -------
        csizes : `tuple` of `numpy.ndarray`
            The cell sizes per axis. The length of the vectors is the
            same as the corresponding ``grid.coord_vectors``.
            For axes with 1 grid point, cell size is set to 0.0.

        Examples
        --------
        >>> grid = TensorGrid([0, 1], [-1, 0, 2])
        >>> part = RectPartition(grid)  # Implicit begin and end
        >>> part.cell_sizes()
        (array([ 1.,  1.]), array([ 1. ,  1.5,  2. ]))

        >>> part = RectPartition(grid, begin=[0, -1], end=[1, 2])
        >>> part.cell_sizes()
        (array([ 0.5,  0.5]), array([ 0.5,  1.5,  1. ]))
        """
        csizes = []
        for ax, vec in enumerate(self.grid.coord_vectors):
            if len(vec) == 1:
                csizes.append(np.array([0.0]))
            else:
                csize = np.empty_like(vec)
                csize[1:-1] = (vec[2:] - vec[:-2]) / 2.0
                csize[0] = (vec[0] + vec[1]) / 2 - self.min()[ax]
                csize[-1] = self.max()[ax] - (vec[-2] + vec[-1]) / 2
                csizes.append(csize)

        return tuple(csizes)

    @property
    def cell_volume(self):
        """Volume of the 'inner' cells, regardless of begin and end.

        Only defined if ``self.grid`` is a `RegularGrid`.

        Examples
        --------
        >>> grid = RegularGrid([0, 0], [1, 1], (3, 3))
        >>> part = RectPartition(grid)
        >>> part.cell_volume
        0.25
        """
        if not isinstance(self.grid, RegularGrid):
            raise TypeError('cell_volume not defined for non-regular grids.')
        return float(np.prod(self.extent() / self.shape))

    def approx_equals(self, other, atol):
        """Return `True` in case of approximate equality.

        Returns
        -------
        approx_eq : `bool`
            `True` if ``other`` is a `RectPartition` instance with
            ``self.bbox == other.bbox`` up to ``atol`` and
            ``self.grid == other.other`` up to ``atol``.
        """
        if other is self:
            return True
        elif not isinstance(other, RectPartition):
            return False
        else:
            return (self.bbox.approx_equals(other.bbox, atol=atol) and
                    self.grid.approx_equals(other.grid, atol=atol))

    def __eq__(self, other):
        """Return ``self == other``."""
        return self.bbox == other.bbox and self.grid == other.grid

    def insert(self, index, other):
        """Return a copy with ``other`` inserted before ``index``.

        Parameters
        ----------
        index : `int`
            Index of the dimension before which ``other`` is to
            be inserted. Must fulfill ``0 <= index <= ndim``.
        other : `RectPartition`
            Partition to be inserted

        Returns
        -------
        newpart : `RectPartition`
            Partition with the inserted other partition

        Examples
        --------
        """
        newgrid = self.grid.insert(index, other.grid)
        newbbox = self.bbox.insert(index, other.bbox)
        new_b = newbbox.begin
        new_e = newbbox.end
        return RectPartition(newgrid, begin=new_b, end=new_e)

    # TODO: pretty-print
    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n {!r}'.format(self.grid)
        sep = ',\n '
        if self._custom_begin:
            inner_str += sep + 'begin={}'.format(
                array1d_repr(self.bbox.begin))
            sep = ', '
        if self._custom_end:
            inner_str += sep + 'end={}'.format(
                array1d_repr(self.bbox.end))

        return '{}({})'.format(self.__class__.__name__, inner_str)


def uniform_partition(intv_prod, num_nodes, order='C', node_at_bdry=True):
    """Return a partition of ``intv_prod`` by a regular grid.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Interval product to be partitioned
    num_nodes : `int` or sequence of `int`
        Number of nodes per axis. For 1d intervals, a single integer
        can be specified.
    order : {'C', 'F'}
        Ordering of the generated grid
    node_at_bdry : `bool` or array-like
        If `True`, the bounding box ends exactly at the boundary grid
        nodes. For `False`, it extends the grid by half a cell size.
        If an array-like is given, it must have shape ``(ndim, 2)``,
        where ``ndim`` is the number of dimensions. It defines per axis
        if the bounding box ends at the left node (first column) and
        at the right node (second column).

    Examples
    --------
    """
    # Sanity checks
    if np.shape(num_nodes) == ():
        num_nodes = (num_nodes,)
    elif len(num_nodes) != intv_prod.ndim:
        raise ValueError('num_nodes has length {}, expected {}.'
                         ''.format(len(num_nodes), (intv_prod.ndim)))

    if np.shape(node_at_bdry) == ():
        node_at_bdry = ([(bool(node_at_bdry), bool(node_at_bdry))] *
                        intv_prod.ndim)
    elif np.shape(node_at_bdry) == (intv_prod.ndim, 2):
        pass
    else:
        raise ValueError('node_at_bdry has shape {}, expected {}.'
                         ''.format(np.shape(node_at_bdry),
                                   (intv_prod.ndim, 2)))

    gmin, gmax = [], []
    for n, beg, end, (bdry_l, bdry_r) in zip(num_nodes, intv_prod.begin,
                                             intv_prod.end, node_at_bdry):
        # Shift left and right boundary grid node if necessary.
        # The conditions to be met are:
        # 1. The node should be half a stride away from the boundary
        # 2. Adding (n-1) * stride in the corresponding direction should
        #    give the other boundary.
        # From these conditions, it follows that stride = (b - a) / (n - 1/2)
        # in the asymmetric cases and stride = (b - a) / n for both shifted.
        if bdry_l and bdry_r:
            gmin.append(beg)
            gmax.append(end)
        elif bdry_l and not bdry_r:
            gmin.append(beg)
            gmax.append(end - (end - beg) / (2 * n - 1))
        elif not bdry_l and bdry_r:
            gmin.append(beg + (end - beg) / (2 * n - 1))
            gmax.append(end)
        else:
            gmin.append(beg + (end - beg) / (2 * n))
            gmax.append(end - (end - beg) / (2 * n))

    grid = RegularGrid(gmin, gmax, num_nodes, order=order)
    return RectPartition(grid, begin=intv_prod.begin, end=intv_prod.end)

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
