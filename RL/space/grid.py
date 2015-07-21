# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.

"""Efficient implementation of n-dimensional sampling grids.

Includes TensorGrid, RegularGrid, UniformGrid.

# TODO: document public interface
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, super
from future import standard_library

# External module imports
import numpy as np
from scipy.lib.blas import get_blas_funcs
from numbers import Integral, Real
from math import sqrt

# RL imports
from RL.space.set import Set
from RL.utility.utility import errfmt

standard_library.install_aliases()


class TensorGrid(Set):

    """An n-dimensional tensor grid.

    This is a sparse representation of a collection of n-dimensional points
    defined by the tensor product of n coordinate vectors.

    Example: x = (x_1, x_2), y = (y_1, y_2, y_3, y_4), z = (z_1, z_2, z_3).
    The resulting grid consists of all possible combinations
    p = (x_i, y_j, z_k), hence 2 * 4 * 3 = 24 points in total.

    Attributes
    ----------

    ============= ======================= ===========
    Name          Type                    Description
    ============= ======================= ===========
    coord_vectors list of numpy.ndarray's Vectors containing\
    the grid point coordinates along each axis
    dim           int                     Grid dimension
    shape         tuple of int's          Number of grid points per\
    axis
    ntotal        int                     Total number of grid points
    min           numpy.ndarray           Grid point with minimal\
    coordinates
    max           numpy.ndarray           Grid point with maximal\
    coordinates
    ============= ======================= ===========

    Methods
    -------

    ===================== ======================== ===========
    Signature             Return type              Description
    ===================== ======================== ===========
    equals(other)         boolean                  Equality test,\
    equivalent to 'self == other'
    contains(point)       boolean                  Membership test,\
    equivalent to 'point in self'
    points(order='C')     numpy.ndarray            All grid points as\
    a single array
    corners(order='C')    tuple of numpy.ndarray's The corner points\
    of the grid
    meshgrid(sparse=True) tuple of numpy.ndarray's Efficient grid\
    for function evaluation (see numpy.meshgrid)
    ===================== ======================== ===========
    """

    def __init__(self, *coord_vectors):
        """
        Parameters
        ----------

        v1,...,vn : array-like
            The coordinate vectors defining the grid points. They must be
            sorted in ascending order and may not contain duplicates.
            Empty vectors are not allowed.
        """
        if not coord_vectors:
            raise ValueError('No coordinate vectors given.')

        vecs = tuple(np.atleast_1d(vec).astype(float)
                     for vec in coord_vectors)
        for i, vec in enumerate(vecs):

            if len(vec) == 0:
                raise ValueError('Vector {} has zero length.'.format(i+1))

            if np.nan in vec:
                raise ValueError(errfmt('''
                Vector {} contains NaNs or garbage entries'''.format(i+1)))

            if np.inf in vec or -np.inf in vec:
                raise ValueError('Vector {} contains +-inf'.format(i+1))

            if vec.ndim != 1:
                raise ValueError(errfmt('''
                Dimension of vector {} is {} instead of 1.
                '''.format(i+1, vec.ndim)))

            sorted_vec = np.sort(vec)
            if np.any(vec != sorted_vec):
                raise ValueError(errfmt('''
                Vector {} not sorted.'''.format(i+1)))

            for j in range(len(vec) - 1):
                if vec[j] == vec[j+1]:
                    raise ValueError(errfmt('''
                    Vector {} contains duplicates.'''.format(i+1)))

        self._coord_vectors = vecs

    @property
    def coord_vectors(self):
        """The coordinate vectors of the grid."""
        return self._coord_vectors

    @property
    def dim(self):
        """The dimension of the grid."""
        return len(self.coord_vectors)

    @property
    def shape(self):
        """The number of grid points per axis."""
        return tuple(len(vec) for vec in self.coord_vectors)

    @property
    def ntotal(self):
        """The total number of grid points."""
        return np.prod(self.shape)

    @property
    def min(self):
        """Vector containing the minimal coordinate per axis."""
        return np.array([vec[0] for vec in self.coord_vectors])

    @property
    def max(self):
        """Vector containing the maximal coordinate per axis."""
        return np.array([vec[-1] for vec in self.coord_vectors])

    def equals(self, other):
        """Test if this grid is equal to another grid.

        Return True if all coordinate vectors are equal, otherwise
        False.
        """
        return (isinstance(other, TensorGrid) and
                self.dim == other.dim and
                all(np.all(vec_s == vec_o) for (vec_s, vec_o) in zip(
                    self.coord_vectors, other.coord_vectors)))

    def contains(self, point):
        """Test if this grid contains the given point.

        Be aware that rounding errors may lead to misleading results.
        """
        point = np.atleast_1d(point)
        return (point.shape == (self.dim,) and
                all(point[i] in self.coord_vectors[i]
                    for i in range(self.dim)))

    def points(self, order='C'):
        """All grid points in a single array.

        Parameters
        ----------

        order : 'C' or 'F'
            The ordering of the axes in which the points appear in
            the output.

        Returns
        -------

        points : numpy.ndarray
            The size of the array is ntotal x dim, i.e. the points are
            stored as rows.
        """
        if order not in ('C', 'F'):
            raise ValueError(errfmt('''
            Value of 'order' ({}) must be 'C' or 'F'.'''.format(order)))

        axes = range(self.dim) if order == 'C' else reversed(range(self.dim))
        point_arr = np.empty((self.ntotal, self.dim))

        nrepeats = self.ntotal
        ntiles = 1
        for axis in axes:
            nrepeats //= self.shape[axis]
            point_arr[:, axis] = np.repeat(
                np.tile(self.coord_vectors[axis], ntiles), nrepeats)
            ntiles *= self.shape[axis]

        return point_arr

    def corners(self, order='C'):
        """The corner points of the grid.

        Parameters
        ----------

        order : 'C' or 'F'
            The ordering of the axes in which the corners appear in
            the output.

        Returns
        -------

        out : tuple of numpy.ndarray's
            The number of arrays is 2^m, where m is the number of
            non-degenerate axes.
        """
        if order not in ('C', 'F'):
            raise ValueError(errfmt('''
            Value of 'order' ({}) must be 'C' or 'F'.'''.format(order)))

        minmax_vecs = []
        for axis in range(self.dim):
            if self.shape[axis] == 1:
                minmax_vecs.append(self.coord_vectors[axis][0])
            else:
                minmax_vecs.append((self.coord_vectors[axis][0],
                                    self.coord_vectors[axis][-1]))

        minmax_grid = TensorGrid(*minmax_vecs)
        return tuple(minmax_grid.points(order=order))

    def meshgrid(self, sparse=True):
        """A grid suitable for function evaluation.

        Parameters
        ----------

        sparse : boolean, optional
            If True, the grid is not 'fleshed out' to save memory.

        Returns
        -------

        meshgrid : tuple of numpy.ndarray's

        See also
        --------

        numpy.meshgrid (we use indexing='ij' and copy=True)
        """
        return tuple(np.meshgrid(*self.coord_vectors, indexing='ij',
                                 sparse=sparse, copy=True))

    def __repr__(self):
        """repr(self) implementation"""
        return 'TensorGrid(' + ', '.join([repr(tuple(vec))
                                         for vec in self.coord_vectors]) + ')'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
