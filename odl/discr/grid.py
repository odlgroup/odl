# Copyright 2014, 2015 The ODL development group
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

"""Sparse implementations of n-dimensional sampling grids.

Sampling grids are collections of points in an n-dimensional coordinate
space with a certain structure which is exploited to minimize storage.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import range, super, zip

# External module imports
import numpy as np

# ODL imports
from odl.set.domain import IntervalProd
from odl.set.sets import Set, Integers
from odl.util.utility import array1d_repr, array1d_str


__all__ = ('TensorGrid', 'RegularGrid', 'uniform_sampling')


class TensorGrid(Set):

    """An n-dimensional tensor grid.

    This is a sparse representation of a collection of n-dimensional points
    defined by the tensor product of n coordinate vectors.

    Example: x = (x_1, x_2), y = (y_1, y_2, y_3, y_4), z = (z_1, z_2, z_3).
    The resulting grid consists of all possible combinations
    p = (x_i, y_j, z_k), hence 2 * 4 * 3 = 24 points in total.
    """

    def __init__(self, *coord_vectors, **kwargs):
        """Initialize a TensorGrid instance.

        Parameters
        ----------
        v1,...,vn : array-like
            The coordinate vectors defining the grid points. They must
            be sorted in ascending order and may not contain
            duplicates. Empty vectors are not allowed.
        kwargs : {'as_midp'}
            'as_midp' : bool, optional  (Default: `False`)
                Treat grid points as midpoints of rectangular cells.
                This influences the behavior of `min`, `max` and
                `cell_sizes`.

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g
        TensorGrid([1.0, 2.0, 5.0], [-2.0, 1.5, 2.0])
        >>> print(g)
        grid [1.0, 2.0, 5.0] x [-2.0, 1.5, 2.0]
        >>> g.ndim  # number of axes
        2
        >>> g.shape  # points per axis
        (3, 3)
        >>> g.ntotal  # total number of points
        9

        Grid points can be extracted with index notation (NOTE: This is
        slow, do not loop over the grid using indices!):

        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> g[0, 0, 0, 0]
        array([-1., 2., 5., 2.])

        Slices and ellipsis are also supported:

        >>> g[:, 0, 0, 0]
        TensorGrid([-1.0, 0.0, 3.0], [2.0], [5.0], [2.0])
        >>> g[0, ..., 1:]
        TensorGrid([-1.0], [2.0, 4.0], [5.0], [4.0, 7.0])
        """
        if not coord_vectors:
            raise ValueError('No coordinate vectors given.')

        vecs = tuple(np.atleast_1d(vec).astype(float)
                     for vec in coord_vectors)
        for i, vec in enumerate(vecs):

            if len(vec) == 0:
                raise ValueError('vector {} has zero length.'.format(i+1))

            if not np.all(np.isfinite(vec)):
                raise ValueError('vector {} contains invalid entries.'
                                 ''.format(i+1))

            if vec.ndim != 1:
                raise ValueError('vector {} has {} dimensions instead of 1.'
                                 ''.format(i+1, vec.ndim))

            sorted_vec = np.sort(vec)
            if np.any(vec != sorted_vec):
                raise ValueError('vector {} not sorted.'.format(i+1))

            if np.any(np.diff(vec) == 0):
                raise ValueError('vector {} contains duplicates.'.format(i+1))

        self._coord_vectors = vecs
        self._as_midp = bool(kwargs.pop('as_midp', False))
        self._ideg = np.array([i for i in range(len(vecs))
                               if len(vecs[i]) == 1])
        self._inondeg = np.array([i for i in range(len(vecs))
                                  if len(vecs[i]) != 1])

        # Thes args are not public and thus not checked for consistency!
        _exact_min = kwargs.pop('_exact_min', None)
        _exact_max = kwargs.pop('_exact_max', None)
        if _exact_min is not None:
            _exact_min = np.atleast_1d(_exact_min)
        if _exact_max is not None:
            _exact_max = np.atleast_1d(_exact_max)
        self._exact_min = _exact_min
        self._exact_max = _exact_max

    # Attributes
    @property
    def coord_vectors(self):
        """The coordinate vectors of the grid."""
        return self._coord_vectors

    @property
    def ndim(self):
        """The number of dimensions of the grid."""
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
    def min_pt(self):
        """Vector containing the minimal grid coordinates per axis.

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.min_pt
        array([ 1., -2.])
        """
        return np.array([vec[0] for vec in self.coord_vectors])

    @property
    def max_pt(self):
        """Vector containing the maximal grid coordinates per axis.

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.max_pt
        array([ 5., 2.])
        """
        return np.array([vec[-1] for vec in self.coord_vectors])

    def min(self):
        """Return vector with minimal cell coordinates per axis.

        This is relevant if the grid was initialized with
        `as_midp=True`, in which case the minimum is half a cell
        smaller than the minimum grid point.

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2], as_midp=False)
        >>> g.min()
        array([ 1., -2.])
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2], as_midp=True)
        >>> g.min()
        array([ 0.5 , -3.75])
        """
        if not self._as_midp:
            return self.min_pt
        elif self._exact_min is not None:
            return self._exact_min
        else:
            minpt_cell_size = np.array([cs[0] for cs in self.cell_sizes()])
            return self.min_pt - minpt_cell_size / 2

    def max(self):
        """Return vector with maximal cell coordinates per axis.

        This is relevant if the grid was initialized with
        `as_midp=True`, in which case the maximum is half a cell
        larger than the maximum grid point.

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2], as_midp=False)
        >>> g.max()
        array([ 5., 2.])
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2], as_midp=True)
        >>> g.max()
        array([ 6.5 ,  2.25])
        """
        if not self._as_midp:
            return self.max_pt
        elif self._exact_max is not None:
            return self._exact_max
        else:
            maxpt_cell_size = np.array([cs[-1] for cs in self.cell_sizes()])
            return self.max_pt + maxpt_cell_size / 2

    # Methods
    def element(self):
        """An arbitrary element, the minimum coordinates."""
        return self.min_pt

    def approx_equals(self, other, tol):
        """Test if this grid is equal to another grid.

        Parameters
        ----------
        other : object
            Object to be tested
        tol : float
            Allow deviations up to this number in absolute value
            per vector entry.

        Returns
        -------
        equals : bool
            `True` if `other` is a `TensorGrid` instance with all
            coordinate vectors equal (up to the given tolerance), to
            the ones of this grid, otherwise `False`.

        Examples
        --------
        >>> g1 = TensorGrid([0, 1], [-1, 0, 2])
        >>> g2 = TensorGrid([-0.1, 1.1], [-1, 0.1, 2])
        >>> g1.approx_equals(g2, tol=0)
        False
        >>> g1.approx_equals(g2, tol=0.15)
        True
        """
        if other is self:
            return True

        # pylint: disable=arguments-differ
        return (type(self) == type(other) and
                self.ndim == other.ndim and
                self.shape == other.shape and
                all(np.allclose(vec_s, vec_o, atol=tol, rtol=0.0)
                    for (vec_s, vec_o) in zip(self.coord_vectors,
                                              other.coord_vectors)))

    def __eq__(self, other):
        """`g.__eq__(other) <==> g == other`."""
        return self.approx_equals(other, tol=0.0)

    def approx_contains(self, other, tol):
        """Test if `other` belongs to this grid up to a tolerance.

        Parameters
        ----------
        other : array-like or `float`
            The object to test for membership in this grid
        tol : float
            Allow deviations up to this number in absolute value
            per vector entry.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.approx_contains([0, 0], tol=0.0)
        True
        >>> [0, 0] in g  # equivalent
        True
        >>> g.approx_contains([0.1, -0.1], tol=0.0)
        False
        >>> g.approx_contains([0.1, -0.1], tol=0.15)
        True
        """
        other = np.atleast_1d(other)
        return (other.shape == (self.ndim,) and
                all(np.any(np.isclose(vector, coord, atol=tol, rtol=0.0))
                    for vector, coord in zip(self.coord_vectors, other)))

    def __contains__(self, other):
        """`g.__contains__(other) <==> other in g`."""
        other = np.atleast_1d(other)
        return (other.shape == (self.ndim,) and
                all(coord in vector
                    for vector, coord in zip(self.coord_vectors, other)))

    def is_subgrid(self, other, tol=0.0):
        """Test if this grid is contained in another grid.

        Parameters
        ----------
        other : `TensorGrid`
            The other grid which is supposed to contain this grid
        tol : float
            Allow deviations up to this number in absolute value
            per coordinate vector entry.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g_sub = TensorGrid([0], [-1, 2])
        >>> g_sub.is_subgrid(g)
        True
        >>> g_sub = TensorGrid([0.1], [-1.05, 2.1])
        >>> g_sub.is_subgrid(g)
        False
        >>> g_sub.is_subgrid(g, tol=0.15)
        True
        """
        # Optimization for some common cases
        if other is self:
            return True
        if not (isinstance(other, TensorGrid) and
                np.all(self.shape <= other.shape) and
                np.all(self.min_pt >= other.min_pt - tol) and
                np.all(self.max_pt <= other.max_pt + tol)):
            return False

        # Array version of the fuzzy subgrid test, about 3 times faster
        # than the loop version.
        for vec_o, vec_s in zip(other.coord_vectors, self.coord_vectors):
            # pylint: disable=unbalanced-tuple-unpacking
            vec_o_mg, vec_s_mg = np.meshgrid(vec_o, vec_s, sparse=True,
                                             copy=True, indexing='ij')
            if not np.all(np.any(np.abs(vec_s_mg - vec_o_mg) <= tol, axis=0)):
                return False

        return True

    def insert(self, other, index):
        """Insert another grid before the given index.

        The given grid (`m` dimensions) is inserted into the current
        one (`n` dimensions) before the given index, resulting in a new
        `TensorGrid` with `n + m` dimensions.
        Note that no changes are made in-place.

        Parameters
        ----------
        other : `TensorGrid`, float or array-like
            The grid to be inserted. A float or array `a` is treated as
            `TensorGrid(a)`.
        index : `Integral`
            The index of the dimension before which `other` is to
            be inserted. Must fulfill `0 <= index <= ndim`.

        Returns
        -------
        newgrid : `TensorGrid`
            The enlarged grid

        Examples
        --------
        >>> g1 = TensorGrid([0, 1], [-1, 2])
        >>> g2 = TensorGrid([1], [-6, 15])
        >>> g1.insert(g2, 1)
        TensorGrid([0.0, 1.0], [1.0], [-6.0, 15.0], [-1.0, 2.0])
        """
        if index not in Integers():
            raise TypeError('{} is not an integer.'.format(index))
        if not 0 <= index <= self.ndim:
            raise IndexError('index {} out of valid range 0 -> {}.'
                             ''.format(index, self.ndim))

        if not isinstance(other, TensorGrid):
            other = TensorGrid(other)

        new_vecs = (self.coord_vectors[:index] + other.coord_vectors +
                    self.coord_vectors[index:])
        return TensorGrid(*new_vecs)

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
            The size of the array is ntotal x ndim, i.e. the points are
            stored as rows.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.points()
        array([[ 0., -1.],
               [ 0.,  0.],
               [ 0.,  2.],
               [ 1., -1.],
               [ 1.,  0.],
               [ 1.,  2.]])
        >>> g.points(order='F')
        array([[ 0., -1.],
               [ 1., -1.],
               [ 0.,  0.],
               [ 1.,  0.],
               [ 0.,  2.],
               [ 1.,  2.]])
        """
        if order not in ('C', 'F'):
            raise ValueError('order {!r} not recognized.'.format(order))

        axes = range(self.ndim) if order == 'C' else reversed(range(self.ndim))
        point_arr = np.empty((self.ntotal, self.ndim))

        nrepeats = self.ntotal
        ntiles = 1
        for axis in axes:
            nrepeats //= self.shape[axis]
            point_arr[:, axis] = np.repeat(
                np.tile(self.coord_vectors[axis], ntiles), nrepeats)
            ntiles *= self.shape[axis]

        return point_arr

    def corners(self, order='C'):
        """The corner points of the grid in a single array.

        Parameters
        ----------
        order : {'C', 'F'}
            The ordering of the axes in which the corners appear in
            the output.

        Returns
        -------
        out : numpy.ndarray
            The size of the array is 2^m x ndim, where m is the number
            of non-degenerate axes, i.e. the corners are stored as rows.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.corners()
        array([[ 0., -1.],
               [ 0.,  2.],
               [ 1., -1.],
               [ 1.,  2.]])
        >>> g.corners(order='F')
        array([[ 0., -1.],
               [ 1., -1.],
               [ 0.,  2.],
               [ 1.,  2.]])
        """
        if order not in ('C', 'F'):
            raise ValueError('order {!r} not recognized.'.format(order))

        minmax_vecs = []
        for axis in range(self.ndim):
            if self.shape[axis] == 1:
                minmax_vecs.append(self.coord_vectors[axis][0])
            else:
                minmax_vecs.append((self.coord_vectors[axis][0],
                                    self.coord_vectors[axis][-1]))

        minmax_grid = TensorGrid(*minmax_vecs)
        return minmax_grid.points(order=order)

    def cell_sizes(self):
        """The grid cell sizes as coordinate vectors.

        Returns
        -------
        csizes : tuple of numpy.ndarray
            The cell sizes per axis. The length of the vectors will be
            one less than `coord_vectors` if `as_midp == False`,
            otherwise they will have the same length.
            For axes with 1 grid point, cell size is set to 0.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.cell_sizes()
        (array([ 1.]), array([ 1.,  2.]))
        >>> g = TensorGrid([0, 1], [-1, 0, 2], as_midp=True)
        >>> g.cell_sizes()
        (array([ 1.,  1.]), array([ 1. ,  1.5,  2. ]))
        """
        csizes = []
        for vec in self.coord_vectors:
            if len(vec) == 1:
                csizes.append(np.array([0.0]))
            else:
                if self._as_midp:
                    csize = np.empty_like(vec)
                    csize[1:-1] = (vec[2:] - vec[:-2]) / 2.0
                    csize[0] = vec[1] - vec[0]
                    csize[-1] = vec[-1] - vec[-2]
                else:
                    csize = vec[1:] - vec[:-1]
                csizes.append(csize)

        return tuple(csizes)

    def meshgrid(self, sparse=True):
        """A grid suitable for function evaluation.

        Parameters
        ----------
        sparse : bool, optional
            If True, the grid is not "fleshed out" to save memory.

        Returns
        -------
        meshgrid : tuple of numpy.ndarray's
            Function evaluation grid with size-1 axes if `sparse=True`,
            otherwise with "fleshed out" axes

        See also
        --------
        numpy.meshgrid : coordinate matrices from coordinate vectors
            We use `indexing='ij'` and `copy=True`

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> x, y = g.meshgrid()
        >>> x
        array([[ 0.],
               [ 1.]])
        >>> y
        array([[-1.,  0.,  2.]])

        Easy function evaluation via broadcasting:

        >>> x**2 - y**2
        array([[-1.,  0., -4.],
               [ 0.,  1., -3.]])

        Alternatively, the grids can be 'fleshed out', using
        significantly more memory

        >>> x, y = g.meshgrid(sparse=False)
        >>> x
        array([[ 0.,  0.,  0.],
               [ 1.,  1.,  1.]])
        >>> y
        array([[-1.,  0.,  2.],
               [-1.,  0.,  2.]])
        >>> x**2 - y**2
        array([[-1.,  0., -4.],
               [ 0.,  1., -3.]])
        """
        return tuple(np.meshgrid(*self.coord_vectors, indexing='ij',
                                 sparse=sparse, copy=True))

    def convex_hull(self):
        """The "inner" of the grid, an `IntervalProd`.

        The convex hull of a set is the union of all line segments
        between points in the set. For a tensor grid, it is the
        interval product given by the extremal coordinates.

        Returns
        -------
        chull : `IntervalProd`
            Interval product defined by the minimum and maximum of
            the grid (depends on `as_midp`)

        Examples
        --------
        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> g.convex_hull()
        IntervalProd([-1.0, 2.0, 5.0, 2.0], [3.0, 4.0, 5.0, 7.0])
        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7],
        ...                as_midp=True)
        >>> g.convex_hull()
        IntervalProd([-1.5, 1.0, 5.0, 1.0], [4.5, 5.0, 5.0, 8.5])
        """
        return IntervalProd(self.min(), self.max())

    def __getitem__(self, slc):
        """self[slc] implementation.

        Parameters
        ----------
        slc : int or slice
            Negative indices and `None` (new axis) are not supported.

        Examples
        --------
        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> g[0, 0, 0, 0]
        array([-1., 2., 5., 2.])
        >>> g[:, 0, 0, 0]
        TensorGrid([-1.0, 0.0, 3.0], [2.0], [5.0], [2.0])
        >>> g[0, ..., 1:]
        TensorGrid([-1.0], [2.0, 4.0], [5.0], [4.0, 7.0])
        >>> g[::2, ..., ::2]
        TensorGrid([-1.0, 3.0], [2.0, 4.0], [5.0], [2.0, 7.0])
        """
        slc_list = list(np.atleast_1d(np.s_[slc]))
        if None in slc_list:
            raise IndexError('Creation of new axes not supported.')

        try:
            idx = np.array(slc_list, dtype=int)  # All single indices
            if len(idx) < self.ndim:
                raise IndexError('too few indices ({} < {}).'
                                 ''.format(len(idx), self.ndim))
            elif len(idx) > self.ndim:
                raise IndexError('too many indices ({} > {}).'
                                 ''.format(len(idx), self.ndim))

            return np.array([v[i] for i, v in zip(idx, self.coord_vectors)])

        except TypeError:
            pass

        if Ellipsis in slc_list:
            if slc_list.count(Ellipsis) > 1:
                raise IndexError("Cannot use more than one ellipsis.")
            if len(slc_list) == self.ndim + 1:  # Ellipsis without effect
                ellipsis_idx = self.ndim
                num_after_ellipsis = 0
            else:
                ellipsis_idx = slc_list.index(Ellipsis)
                num_after_ellipsis = len(slc_list) - ellipsis_idx - 1
            slc_list.remove(Ellipsis)

        else:
            if len(slc_list) < self.ndim:
                raise IndexError('too few axes ({} < {}).'
                                 ''.format(len(slc_list), self.ndim))
            ellipsis_idx = self.ndim
            num_after_ellipsis = 0

        if any(s.start == s.stop and s.start is not None
               for s in slc_list if isinstance(s, slice)):
            raise IndexError('Slices with empty axes not allowed.')

        if len(slc_list) > self.ndim:
            raise IndexError('too many axes ({} > {}).'
                             ''.format(len(slc_list), self.ndim))

        new_vecs = []
        for i in range(ellipsis_idx):
            new_vecs.append(self.coord_vectors[i][slc_list[i]])
        for i in range(ellipsis_idx, self.ndim - num_after_ellipsis):
            new_vecs.append(self.coord_vectors[i])
        for i in reversed(list(range(1, num_after_ellipsis + 1))):
            new_vecs.append(self.coord_vectors[-i][slc_list[-i]])

        return TensorGrid(*new_vecs)

    def __repr__(self):
        """g.__repr__() <==> repr(g)."""
        vec_str = ', '.join(array1d_repr(vec) for vec in self.coord_vectors)
        if self._as_midp:
            return 'TensorGrid({}, as_midp=True)'.format(vec_str)
        else:
            return 'TensorGrid({})'.format(vec_str)

    def __str__(self):
        """g.__str__() <==> str(g)."""
        grid_str = ' x '.join(array1d_str(vec) for vec in self.coord_vectors)
        if self._as_midp:
            return 'midp grid {}'.format(grid_str)
        else:
            return 'grid {}'.format(grid_str)


class RegularGrid(TensorGrid):

    """An n-dimensional tensor grid with equidistant coordinates.

    This is a sparse representation of an n-dimensional grid defined
    as the tensor product of n coordinate vectors with equidistant
    nodes. The grid points are calculated according to the rule

    x_j = center + (j - (shape-1)/2 * stride)

    with elementwise addition and multiplication. Note that the
    division is a true division (no rounding), thus if there is an
    axis with an even number of nodes, the center is not a grid point.
    """

    def __init__(self, min_pt, max_pt, shape, **kwargs):
        """Initialize a new instance.
        Parameters
        ----------
        min_pt : array-like or float
            Grid point with minimum coordinates, can be a single float
            for 1D grids
        max_pt : array-like or float
            Grid point with maximum coordinates, can be a single float
            for 1D grids
        shape : array-like or int
            The number of grid points per axis, can be an integer for
            1D grids
        kwargs : {'as_midp'}
            'as_midp' : bool, optional  (Default: `False`)
                Treat grid points as midpoints of rectangular cells.
                This influences the behavior of `min`, `max` and
                `cell_sizes`.

        Examples
        --------
        >>> rg = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg
        RegularGrid([-1.5, -1.0], [-0.5, 3.0], [2, 3])
        >>> rg.coord_vectors
        (array([-1.5, -0.5]), array([-1.,  1.,  3.]))
        >>> rg.ndim, rg.ntotal
        (2, 6)
        """
        min_pt = np.atleast_1d(min_pt).astype(np.float64)
        max_pt = np.atleast_1d(max_pt).astype(np.float64)
        shape = np.atleast_1d(shape).astype(np.int64)

        if any(x.ndim != 1 for x in (min_pt, max_pt, shape)):
            raise ValueError('input arrays have dimensions {}, {}, {} '
                             'instead of 1.'
                             ''.format(min_pt.ndim, max_pt.ndim, shape.ndim))

        if not min_pt.shape == max_pt.shape == shape.shape:
            raise ValueError('input array shapes are {}, {}, {} but '
                             'should be equal.'
                             ''.format(min_pt.shape, max_pt.shape,
                                       shape.shape))

        if not np.all(np.isfinite(min_pt)):
            raise ValueError('minimum point {} has invalid entries.'
                             ''.format(min_pt))

        if not np.all(np.isfinite(max_pt)):
            raise ValueError('maximum point {} has invalid entries.'
                             ''.format(max_pt))

        if not np.all(min_pt <= max_pt):
            raise ValueError('minimum point {} has not strictly smaller '
                             'entries than maximum point {}.'
                             ''.format(min_pt, max_pt))

        if np.any(shape <= 0):
            raise ValueError('shape parameter is {} but should only contain '
                             'positive values.'.format(shape))

        degen = np.where(min_pt == max_pt)[0]
        if np.any(shape[degen] != 1):
            raise ValueError('degenerated axes {} with shapes {}, expected '
                             '{}.'.format(tuple(degen[:]), tuple(shape[degen]),
                                          len(degen) * (1,)))

        coord_vecs = [np.linspace(mi, ma, num, endpoint=True, dtype=np.float64)
                      for mi, ma, num in zip(min_pt, max_pt, shape)]
        super().__init__(*coord_vecs, **kwargs)
        self._center = (self.max_pt + self.min_pt) / 2
        self._stride = np.ones(len(shape), dtype='float64')
        idcs = np.where(shape > 1)
        self._stride[idcs] = ((self.max_pt - self.min_pt)[idcs] /
                              (shape[idcs] - 1))

    @property
    def center(self):
        """The center of the grid. Not necessarily a grid point.

        Examples
        --------
        >>> rg = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg.center
        array([-1.,  1.])
        """
        return self._center

    @property
    def stride(self):
        """The step per axis between two neighboring grid points.

        Examples
        --------
        >>> rg = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg.stride
        array([ 1.,  2.])
        """
        return self._stride

    def is_subgrid(self, other, tol=0.0):
        """Test if this grid is contained in another grid.

        Parameters
        ----------

        tol : float
            Allow deviations up to this number in absolute value
            per coordinate vector entry.

        Examples
        --------
        >>> rg = RegularGrid([-2, -2], [0, 4], (3, 4))
        >>> rg.coord_vectors
        (array([-2., -1.,  0.]), array([-2.,  0.,  2.,  4.]))
        >>> rg_sub = RegularGrid([-1, 2], [0, 4], (2, 2))
        >>> rg_sub.coord_vectors
        (array([-1.,  0.]), array([ 2.,  4.]))
        >>> rg_sub.is_subgrid(rg)
        True

        Fuzzy check is also possible. Note that the tolerance still
        applies to the coordinate vectors.

        >>> rg_sub = RegularGrid([-1.015, 2], [0, 3.99], (2, 2))
        >>> rg_sub.is_subgrid(rg, tol=0.01)
        False
        >>> rg_sub.is_subgrid(rg, tol=0.02)
        True
        """
        # Optimize some common cases
        if other is self:
            return True
        if not (isinstance(other, TensorGrid) and
                np.all(self.shape <= other.shape) and
                np.all(self.min_pt >= other.min_pt - tol) and
                np.all(self.max_pt <= other.max_pt + tol)):
            return False

        # For regular grids, it suffices to show that min_pt, max_pt
        # and g[1,...,1] are contained in the other grid. For axes
        # with less than 2 points, this reduces to min_pt and max_pt,
        # and the corresponding indices in the other check point are
        # set to 0.
        if isinstance(other, RegularGrid):
            minmax_contained = (other.approx_contains(self.min_pt, tol=tol) and
                                other.approx_contains(self.max_pt, tol=tol))
            check_idx = np.zeros(self.ndim, dtype=int)
            check_idx[np.array(self.shape) >= 3] = 1
            checkpt_contained = other.approx_contains(self[check_idx.tolist()],
                                                      tol=tol)

            return minmax_contained and checkpt_contained

        elif isinstance(other, TensorGrid):
            # other is a TensorGrid, we need to fall back to full check
            self_tg = TensorGrid(*self.coord_vectors)
            return self_tg.is_subgrid(other)

    def insert(self, grid, index):
        """Insert another regular grid before the given index.

        The given grid (`m` dimensions) is inserted into the current
        one (`n` dimensions) before the given index, resulting in a new
        `RegularGrid` with `n + m` dimensions.
        Note that no changes are made in-place.

        Parameters
        ----------
        grid : `RegularGrid`
            The grid to be inserted.
        index : `Integral`
            The index of the dimension before which 'other' is to
            be inserted. Must fulfill 0 <= index <= ndim.

        Returns
        -------
        newgrid : `RegularGrid`
            The enlarged grid

        Examples
        --------
        >>> rg1 = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg2 = RegularGrid(-3, 7, 6)
        >>> rg1.insert(rg2, 1)
        RegularGrid([-1.5, -3.0, -1.0], [-0.5, 7.0, 3.0], [2, 6, 3])
        """
        if index not in Integers():
            raise TypeError('{} is not an integer.'.format(index))
        if not 0 <= index <= self.ndim:
            raise IndexError('index {} out of valid range 0 -> {}.'
                             ''.format(index, self.ndim))

        if not isinstance(grid, RegularGrid):
            raise TypeError('{} is not a regular grid.'.format(grid))

        new_shape = self.shape[:index] + grid.shape + self.shape[index:]
        new_minpt = (self.min_pt[:index].tolist() + grid.min_pt.tolist() +
                     self.min_pt[index:].tolist())
        new_maxpt = (self.max_pt[:index].tolist() + grid.max_pt.tolist() +
                     self.max_pt[index:].tolist())
        return RegularGrid(new_minpt, new_maxpt, new_shape)

    def __getitem__(self, slc):
        """self[slc] implementation.

        Parameters
        ----------
        slc : int or slice
            Negative indices and 'None' (new axis) are not supported.

        Examples
        --------

        >>> g = RegularGrid([-1.5, -3, -1], [-0.5, 7, 3], (2, 3, 6))
        >>> g[0, 0, 0]
        array([-1.5, -3. , -1. ])
        >>> g[:, 0, 0]
        RegularGrid([-1.5, -3.0, -1.0], [-0.5, -3.0, -1.0], [2, 1, 1])

        Ellipsis can be used, too:

        >>> g[..., ::3]
        RegularGrid([-1.5, -3.0, -1.0], [-0.5, 7.0, 3.0], [2, 3, 2])
        """
        from math import ceil

        slc_list = list(np.atleast_1d(np.s_[slc]))
        if None in slc_list:
            raise IndexError('Creation of new axes not supported.')

        try:
            idx = np.array(slc_list, dtype=int)  # All single indices
            if len(idx) < self.ndim:
                raise IndexError('too few indices ({} < {}).'
                                 ''.format(len(idx), self.ndim))
            elif len(idx) > self.ndim:
                raise IndexError('too many indices ({} > {}).'
                                 ''.format(len(idx), self.ndim))

            return np.array([v[i] for i, v in zip(idx, self.coord_vectors)])

        except TypeError:
            pass

        if Ellipsis in slc_list:
            if slc_list.count(Ellipsis) > 1:
                raise IndexError('Cannot use more than one ellipsis.')
            if len(slc_list) == self.ndim + 1:  # Ellipsis without effect
                ellipsis_idx = self.ndim
                num_after_ellipsis = 0
            else:
                ellipsis_idx = slc_list.index(Ellipsis)
                num_after_ellipsis = len(slc_list) - ellipsis_idx - 1
            slc_list.remove(Ellipsis)

        else:
            if len(slc_list) < self.ndim:
                raise IndexError('too few axes ({} < {}).'
                                 ''.format(len(slc_list), self.ndim))
            ellipsis_idx = self.ndim
            num_after_ellipsis = 0

        if any(s.start == s.stop and s.start is not None
               for s in slc_list if isinstance(s, slice)):
            raise IndexError('Slices with empty axes not allowed.')

        if len(slc_list) > self.ndim:
            raise IndexError('too many axes ({} > {}).'
                             ''.format(len(slc_list), self.ndim))

        new_shape = np.ones(self.ndim, dtype=int)
        new_minpt, new_maxpt = -np.ones((2, self.ndim), dtype='float64')

        # Copy axes corresponding to ellipsis
        ell_idcs = np.arange(ellipsis_idx, self.ndim - num_after_ellipsis)
        new_shape[ell_idcs] = np.array(self.shape)[ell_idcs]
        new_minpt[ell_idcs] = self.min_pt[ell_idcs]
        new_maxpt[ell_idcs] = self.max_pt[ell_idcs]

        # The other indices
        for i in range(ellipsis_idx):
            if isinstance(slc_list[i], slice):
                istart, istop, istep = slc_list[i].indices(self.shape[i])
            else:
                istart, istop, istep = slc_list[i], slc_list[i]+1, 1
            new_shape[i] = ceil((istop - istart) / istep)
            new_minpt[i] = self.min_pt[i] + istart * self.stride[i]
            new_maxpt[i] = self.max_pt[i] - ((self.shape[i] - istop) *
                                             self.stride[i])

        for i in range(1, num_after_ellipsis + 1):
            i = -i
            if isinstance(slc_list[i], slice):
                istart, istop, istep = slc_list[i].indices(self.shape[i])
            else:
                istart, istop, istep = slc_list[i], slc_list[i]+1, 1
            new_shape[i] = ceil((istop - istart) / istep)
            new_minpt[i] = self.min_pt[i] + istart * self.stride[i]
            new_maxpt[i] = self.max_pt[i] - ((self.shape[i] - istop) *
                                             self.stride[i])

        return RegularGrid(new_minpt, new_maxpt, new_shape)

    def __repr__(self):
        """g.__repr__() <==> repr(g)."""
        if self._as_midp:
            return 'RegularGrid({}, {}, {}, as_midp=True)'.format(
                list(self.min_pt), list(self.max_pt), list(self.shape))
        else:
            return 'RegularGrid({}, {}, {})'.format(
                list(self.min_pt), list(self.max_pt), list(self.shape))

    def __str__(self):
        """g.__str__() <==> str(g)."""
        str_lst = []
        for vec in self.coord_vectors:
            if len(vec) <= 3:
                str_lst.append('{!r}'.format(vec.tolist()))
            else:
                str_lst.append('[{}, {}, ..., {}]'.format(vec[0], vec[1],
                                                          vec[-1]))
        if self._as_midp:
            return 'midp regular grid ' + ' x '.join(str_lst)
        else:
            return 'regular grid ' + ' x '.join(str_lst)


def uniform_sampling(intv_prod, num_nodes, as_midp=True):
    """Sample an interval product uniformly.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Set to be sampled
    num_nodes : int or tuple of int
        Number of nodes per axis. For dimension >= 2, a tuple
        is required. All entries must be positive. Entries
        corresponding to degenerate axes must be equal to 1.
    as_midp : bool, optional
        If True, the midpoints of an interval partition will be
        returned, which excludes the endpoints. Otherwise,
        equispaced nodes including the endpoints are generated.
        Note that the resulting strides are different.
        Default: `True`.

    Returns
    -------
    sampling : `RegularGrid`
        Uniform sampling grid for the interval product

    Examples
    --------
    >>> from odl import IntervalProd
    >>> rbox = IntervalProd([-1.5, 2], [-0.5, 3])
    >>> grid = uniform_sampling(rbox, [2, 5])
    >>> grid.coord_vectors
    (array([-1.25, -0.75]), array([ 2.1,  2.3,  2.5,  2.7,  2.9]))
    >>> grid = uniform_sampling(rbox, [2, 5], as_midp=False)
    >>> grid.coord_vectors
    (array([-1.5, -0.5]), array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ]))
    """
    num_nodes = np.atleast_1d(num_nodes).astype(np.int64)

    if not isinstance(intv_prod, IntervalProd):
        raise TypeError('interval product {!r} not an `IntervalProd` instance.'
                        ''.format(intv_prod))

    if np.any(np.isinf(intv_prod.begin)) or np.any(np.isinf(intv_prod.end)):
        raise ValueError('uniform sampling undefined for infinite '
                         'domains.')

    if num_nodes.shape != (intv_prod.ndim,):
        raise ValueError('number of nodes {} has wrong shape '
                         '({} != ({},)).'
                         ''.format(num_nodes, num_nodes.shape, intv_prod.ndim))

    if np.any(num_nodes <= 0):
        raise ValueError('number of nodes {} has non-positive entries.'
                         ''.format(num_nodes))

    if np.any(num_nodes[intv_prod._ideg] > 1):
        raise ValueError('degenerate axes {} cannot be sampled with more '
                         'than one node.'.format(tuple(intv_prod._ideg)))

    if as_midp:
        grid_min = intv_prod.begin + intv_prod.size / (2*num_nodes)
        grid_max = intv_prod.end - intv_prod.size / (2*num_nodes)
        return RegularGrid(grid_min, grid_max, num_nodes, as_midp=as_midp,
                           _exact_min=intv_prod.begin,
                           _exact_max=intv_prod.end)
    else:
        grid_min = intv_prod.begin
        grid_max = intv_prod.end
        return RegularGrid(grid_min, grid_max, num_nodes, as_midp=as_midp)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
