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

"""Partitons of interval products based on tensor grid.

A partition of a set is a finite collection of nonempty, pairwise
disjoint subsets whose union is the original set. The partitions
considered here are based on hypercubes, i.e. the tensor products
of partitions of intervals.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super, str, zip

# External module imports
import numpy as np

# ODL imports
from odl.discr.grid import TensorGrid
from odl.set.domain import IntervalProd
from odl.util.utility import array1d_repr


__all__ = ('GridBasedPartition',)

_POINT_POSITIONS = ('center', 'left')


class GridBasedPartition(object):

    """Abstract partition class based on `TensorGrid`s.

    In 1d, a partition of an interval is implicitly defined by a
    collection of points x[0], ..., x[N-1] (a grid) and the information
    where in the subintervals the grid points lie. If, e.g. the
    points are to lie in the center of the subintervals, these
    subintervals are given as ``[(x[i-1]-x[i])/2, (x[i]+x[i+1])/2]``.
    """

    def __init__(self, grid, begin=None, end=None, point_pos='center'):
        """Initialize a new instance.

        Parameters
        ----------
        grid : `TensorGrid`
            Spatial points supporting the partition
        begin, end : array-like
            Spatial points defining the begin and end of an interval
            product to be partitioned. ``begin`` must be component-wise
            at most ``grid.min_pt``, and ``end`` at least
            ``grid.max_pt``. If not provided, they are inferred
            from the grid.
        point_pos : {'center', 'left'} or sequence of `str`
            Where the grid points lie in the subintervals. A sequence
            is interpreted per dimension, its length must be equal
            to ``grid.ndim``. If a single string is given, that option
            is applied in all dimensions.

        Notes
        -----
        The ``begin`` and ``end`` points are determined from the grid
        as follows:

        - In dimensions with ``point_pos='left'``, it is ``begin = x[0]``
          and ``end = x[N-1]``, where ``x`` is the length ``N`` vector of
          grid coodinates in that dimension.

        - In dimensions with ``point_pos='center'``, the left endpoint
          is given as ``begin = x[0] - (x[1]-x[0])/2``, the right
          endpoint is ``end = x[N-1] + (x[N-1]-x[N-2])/2``.
        """
        if not isinstance(grid, TensorGrid):
            raise TypeError('{!r} is not a TensorGrid instance.'
                            ''.format(grid))

        # Make a grid.ndim - length list from the point_pos, raise if
        # bad length sequence.
        try:
            # String check by concatenation with ''
            pos_lst = [str(point_pos + '').lower()] * grid.ndim
        except TypeError:
            # Sequence given
            pos_lst = [str(pos).lower() for pos in point_pos]
            if len(pos_lst) != grid.ndim:
                raise ValueError('expected {} entries in point_pos, got {}.'
                                 ''.format(grid.ndim, len(pos_lst)))

        for pos, orig_pos in zip(pos_lst, point_pos):
            if pos not in _POINT_POSITIONS:
                raise ValueError('point position {} not understood.'
                                 ''.format(orig_pos))

        # Begin and end: check correct shapes and values if given,
        # calculate otherwise.
        if begin is not None:
            begin = np.asarray(begin, dtype='float64')
            if begin.shape != (grid.ndim,):
                raise ValueError('begin has shape {}, expected {}.'
                                 ''.format(begin.shape, (grid.ndim,)))
            if np.any(begin > grid.min_pt):
                raise ValueError('begin {} has entries larger than the '
                                 'minimum grid point {}.'
                                 ''.format(begin, grid.min_pt))
            self._custom_begin = True

        else:
            begin = np.empty(grid.ndim, dtype='float64')
            for i, (pos, coo_vec) in enumerate(zip(pos_lst,
                                                   grid.coord_vectors)):
                if len(coo_vec) == 1:
                    raise ValueError(
                        'cannot infer begin and end in degenerate dimension '
                        '{}.'.format(i))
                if pos == 'left':
                    begin[i] = coo_vec[0]
                else:  # pos == 'center'
                    begin[i] = coo_vec[0] - (coo_vec[1] - coo_vec[0]) / 2
            self._custom_begin = False

        if end is not None:
            end = np.asarray(end, dtype='float64')
            if end.shape != (grid.ndim,):
                raise ValueError('end has shape {}, expected {}.'
                                 ''.format(end.shape, (grid.ndim,)))
            if np.any(end < grid.max_pt):
                raise ValueError('end {} has entries smaller than the '
                                 'maximum grid point {}.'
                                 ''.format(end, grid.max_pt))
            self._custom_end = True
        else:
            end = np.empty(grid.ndim, dtype='float64')
            for i, (pos, coo_vec) in enumerate(zip(pos_lst,
                                                   grid.coord_vectors)):
                if pos == 'left':
                    end[i] = coo_vec[-1]
                else:  # pos == 'center'
                    end[i] = coo_vec[-1] + (coo_vec[-1] - coo_vec[-2]) / 2
            self._custom_end = False

        super().__init__()
        self._interv_prod = IntervalProd(begin, end)
        self._grid = grid
        self._point_pos = pos_lst

    @property
    def grid(self):
        """The `TensorGrid` defining this partition."""
        return self._grid

    @property
    def interv_prod(self):
        """The `IntervalProd` partitioned by this partition."""
        return self._interv_prod

    @property
    def point_pos(self):
        """Position of the grid points in the subsets (per axis)."""
        return self._point_pos

    # TODO: pretty-print
    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n {!r}'.format(self.grid)
        sep = ',\n '
        if self._custom_begin:
            inner_str += sep + 'begin={}'.format(
                array1d_repr(self.interv_prod.begin))
            sep = ', '
        if self._custom_end:
            inner_str += sep + 'end={}'.format(
                array1d_repr(self.interv_prod.end))
            sep = ', '
        single_pos = all(pos == self.point_pos[0]
                         for pos in self.point_pos[1:])
        if single_pos and self.point_pos[0] != 'center':
            inner_str += sep + "point_pos='{}'".format(self.point_pos[0])
        elif not single_pos:
            inner_str += sep + 'point_pos={}'.format(self.point_pos)

        return '{}({})'.format(self.__class__.__name__, inner_str)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
