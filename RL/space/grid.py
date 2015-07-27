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

=========== ===========
Class name  Description
=========== ===========
TensorGrid  Tensor product of coordinate vectors, possibly non-uniform
RegularGrid Tensor product of vectors with regularly spaced coordinates
=========== ===========
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library

# External module imports
import numpy as np

# RL imports
from RL.space.set import Set
from RL.utility.utility import errfmt, array1d_repr

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

    ========================== ======================== ===========
    Signature                  Return type              Description
    ========================== ======================== ===========
    equals(other, tol=0.0)     boolean                  Equality test,\
    equivalent to 'self == other' for 'tol'==0.0
    contains(point, tol=0.0)   boolean                  Membership\
    test, equivalent to 'point in self' for 'tol'==0
    is_subgrid(other, tol=0.0) boolean                  Subgrid test
    points(order='C')          numpy.ndarray            All grid\
    points as a single array
    corners(order='C')         tuple of numpy.ndarray's The corner\
    points of the grid
    meshgrid(sparse=True)      tuple of numpy.ndarray's Efficient grid\
    for function evaluation (see numpy.meshgrid)
    ========================== ======================== ===========
    """

    def __init__(self, *coord_vectors):
        """
        Parameters
        ----------

        v1,...,vn : array-like
            The coordinate vectors defining the grid points. They must be
            sorted in ascending order and may not contain duplicates.
            Empty vectors are not allowed.

        Examples
        --------

        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g
        TensorGrid([1.0, 2.0, 5.0], [-2.0, 1.5, 2.0])
        >>> print(g)
        [1.0, 2.0, 5.0] x [-2.0, 1.5, 2.0]
        >>> g.dim  # dimension = number of axes
        2
        >>> g.shape  # points per axis
        (3, 3)
        >>> g.ntotal  # total number of points
        9
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

    # Attributes
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
        """Vector containing the minimal coordinate per axis.

        Example
        -------

        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.min
        array([ 1., -2.])
        """
        return np.array([vec[0] for vec in self.coord_vectors])

    @property
    def max(self):
        """Vector containing the maximal coordinate per axis.

        Example
        -------

        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.max
        array([ 5., 2.])
        """
        return np.array([vec[-1] for vec in self.coord_vectors])

    # Methods
    def equals(self, other, tol=0.0):
        """Test if this grid is equal to another grid.

        Return True if all coordinate vectors are equal (up to the
        given tolerance), otherwise False.

        Parameters
        ----------

        tol : float
            Allow deviations up to this number in absolute value
            per vector entry.

        Examples
        --------

        >>> g1 = TensorGrid([0, 1], [-1, 0, 2])
        >>> g2 = TensorGrid([-0.1, 1.1], [-1, 0.1, 2])
        >>> g1.equals(g2)
        False
        >>> g1 == g2  # equivalent
        False
        >>> g1.equals(g2, tol=0.1)
        True
        """
        return (isinstance(other, TensorGrid) and
                self.dim == other.dim and
                self.shape == other.shape and
                all(np.allclose(vec_s, vec_o, atol=tol, rtol=0.0)
                    for (vec_s, vec_o) in zip(self.coord_vectors,
                                              other.coord_vectors)))

    def contains(self, point, tol=0.0):
        """Test if a point belongs to the grid.

        Parameters
        ----------

        tol : float
            Allow deviations up to this number in absolute value
            per vector entry.
        """
        point = np.atleast_1d(point)
        return (point.shape == (self.dim,) and
                all(np.any(np.isclose(vector, coord, atol=tol, rtol=0.0))
                    for vector, coord in zip(self.coord_vectors, point)))

    def is_subgrid(self, other, tol=0.0):
        """Test if this grid is contained in another grid.

        Parameters
        ----------

        tol : float
            Allow deviations up to this number in absolute value
            per coordinate vector entry.
        """
        # TODO: this is quite inefficient, think of a better solution!
        # This version tests each coordinate for fuzzy membership in
        # the corresponding other coordinate vector -> O(n^2)
        return(isinstance(other, TensorGrid) and
               np.all(self.shape <= other.shape) and
               np.all(self.min >= other.min - tol) and
               np.all(self.max <= other.max + tol) and
               all(np.any(np.isclose(vector_o, coord, atol=tol, rtol=0.0))
                   for vector_o, vector_s in zip(other.coord_vectors,
                                                 self.coord_vectors)
                   for coord in vector_s))

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

    def __getitem__(self, slc):
        """self[slc] implementation.

        Parameters
        ----------

        slc : int or slice
            Negative indices are not supported.
        """
        slc_list = list(np.s_[slc])

        if Ellipsis in slc_list:
            if slc_list.count(Ellipsis) > 1:
                raise ValueError("Cannot use more than one ellipsis.")
            if len(slc_list) == self.dim + 1:
                ellipsis_idx = self.dim
                num_after_ellipsis = 0
            else:
                ellipsis_idx = slc_list.index(Ellipsis)
                num_after_ellipsis = len(slc_list) - ellipsis_idx - 1
            slc_list.remove(Ellipsis)

        else:
            if len(slc_list) < self.dim:
                raise IndexError(errfmt('''
                Too few axes ({} < {}).'''.format(len(slc_list, self.dim))))
            ellipsis_idx = self.dim
            num_after_ellipsis = 0

        if len(slc_list) > self.dim:
            raise IndexError(errfmt('''
            Too many axes ({} > {}).'''.format(len(slc_list, self.dim))))

        new_vecs = []
        for i in range(ellipsis_idx):
            new_vecs.append(self.coord_vectors[i][slc_list[i]])
        for i in range(ellipsis_idx, self.dim - num_after_ellipsis):
            new_vecs.append(self.coord_vectors[i])
        for i in reversed(range(1, num_after_ellipsis + 1)):
            new_vecs.append(self.coord_vectors[-i][slc_list[-i]])

        return TensorGrid(*new_vecs)

    def __repr__(self):
        """repr(self) implementation."""
        return 'TensorGrid(' + ', '.join(
            array1d_repr(vec) for vec in self.coord_vectors) + ')'

    def __str__(self):
        """str(self) implementation."""
        return ' x '.join(array1d_repr(vec) for vec in self.coord_vectors)


class RegularGrid(TensorGrid):
    """An n-dimensional tensor grid with equidistant coordinates.

    This is a sparse representation of an n-dimensional grid defined
    as the tensor product of n coordinate vectors with equidistant
    nodes. The grid points are calculated according to the rule

    x_j = center + (j - (shape-1)/2 * stride)

    with elementwise addition and multiplication. Note that the
    division is a true division (no rounding), thus if there is an
    axis with an even number of nodes, the center is not a grid point.

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

    ========================== ======================== ===========
    Signature                  Return type              Description
    ========================== ======================== ===========
    equals(other, tol=0.0)     boolean                  Equality test,\
    equivalent to 'self == other' for 'tol'==0.0
    contains(point, tol=0.0)   boolean                  Membership\
    test, equivalent to 'point in self' for 'tol'==0
    is_subgrid(other, tol=0.0) boolean                  Subgrid test
    points(order='C')          numpy.ndarray            All grid\
    points as a single array
    corners(order='C')         tuple of numpy.ndarray's The corner\
    points of the grid
    meshgrid(sparse=True)      tuple of numpy.ndarray's Efficient grid\
    for function evaluation (see numpy.meshgrid)
    ========================== ======================== ===========
    """

    def __init__(self, shape, center=None, stride=None):
        """
        Parameters
        ----------

        shape : array-like or int
            The number of grid points per axis. For 1D grids, a single
            integer may be given.
        center : array-like or float, optional
            The center of the grid (may not be a grid point). For 1D
            grids, a single float may be given.
            Default: (0.0,...,0.0)
        stride : array-like or float, optional
            Vector pointing from x_j to x_(j+1). For 1D grids, a single
            float may be given.
            Default: (1.0,...,1.0)
        """
        shape = np.atleast_1d(shape, dtype=int)
        if not np.all(shape > 0):
            raise ValueError("'shape' may only have positive entries.")

        if center is None:
            center = np.zeros_like(shape, dtype=float)
        else:
            center = np.atleast_1d(center, dtype=float)
            if len(center) != len(shape):
                raise ValueError(errfmt('''
                'center' ({}) must have the same length as 'shape' ({}).
                '''.format(len(center), len(shape))))

        if stride is None:
            stride = np.ones_like(shape, dtype=float)
        else:
            stride = np.atleast_1d(stride, dtype=float)
            if len(stride) != len(shape):
                raise ValueError(errfmt('''
                'stride' ({}) must have the same length as 'shape' ({}).
                '''.format(len(stride), len(shape))))
        if not np.all(stride > 0):
            raise ValueError("'stride' may only have positive entries.")

        coord_vecs = [ct + (shape-1)/2 * st for ct, st in zip(center, stride)]
        super().__init__(*coord_vecs)

        self._center = center
        self._stride = stride

    @property
    def center(self):
        """The center of the grid. Not necessarily a grid point."""
        return self._center

    @property
    def stride(self):
        """The step per axis between two neighboring grid points."""
        return self._stride

    def is_subgrid(self, other, tol=0.0):
        """Test if this grid is contained in another grid.

        Parameters
        ----------

        tol : float
            Allow deviations up to this number in absolute value
            per coordinate vector entry.
        """
        # Optimize one more common case
        if isinstance(other, RegularGrid):
            idcs = np.where(self.shape > 2)
            if np.any(self.stride[idcs] > other.stride[idcs] + tol):
                return False
        return(isinstance(other, TensorGrid) and
               np.all(self.shape <= other.shape) and
               np.all(self.min >= other.min - tol) and
               np.all(self.max <= other.max + tol) and
               all(np.any(np.isclose(vector_o, coord, atol=tol, rtol=0.0))
                   for vector_o, vector_s in zip(other.coord_vectors,
                                                 self.coord_vectors)
                   for coord in vector_s))


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
