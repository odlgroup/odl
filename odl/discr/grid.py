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

"""Sparse implementations of n-dimensional sampling grids.

Sampling grids are collections of points in an n-dimensional coordinate
space with a certain structure which is exploited to minimize storage.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super, range, str, zip

import numpy as np

from odl.set import Set, IntervalProd
from odl.util.normalize import normalized_index_expression
from odl.util.utility import array1d_repr, array1d_str


__all__ = ('TensorGrid', 'RegularGrid', 'uniform_sampling_fromintv',
           'uniform_sampling')


def sparse_meshgrid(*x):
    """Make a sparse `meshgrid` by adding empty dimensions.

    Parameters
    ----------
    x1,...,xN : `array-like`
        Input arrays to turn into sparse meshgrid vectors.

    Returns
    -------
    meshgrid : tuple of `numpy.ndarray`'s
        Sparse coordinate vectors representing an N-dimensional grid.

    See Also
    --------
    numpy.meshgrid : dense or sparse meshgrids

    Examples
    --------
    >>> x, y = [0, 1], [2, 3, 4]
    >>> mesh = sparse_meshgrid(x, y)
    >>> sum(xi for xi in mesh).ravel()  # first axis slowest
    array([2, 3, 4, 3, 4, 5])
    """
    n = len(x)
    mesh = []
    for ax, xi in enumerate(x):
        xi = np.asarray(xi)
        slc = [None] * n
        slc[ax] = slice(None)
        mesh.append(np.ascontiguousarray(xi[slc]))

    return tuple(mesh)


class TensorGrid(Set):

    """An n-dimensional tensor grid.

    A tensor grid is the set of points defined by all possible
    combination of coordinates taken from fixed coordinate vectors.

    In 2 dimensions, for example, given two coordinate vectors::

        coord_vec1 = [0, 1]
        coord_vec2 = [-1, 0, 2]

    the corresponding tensor grid is the set of all 2d points whose
    first component is from ``coord_vec1`` and the second component from
    ``coord_vec2``. The total number of such points is 2 * 3 = 6::

        points = [[0, -1], [0, 0], [0, 2],
                  [1, -1], [1, 0], [1, 2]]

    Note that this is the standard 'C' ordering where the first axis
    (``coord_vec1``) varies slowest. Ordering is only relevant when
    the point array is actually created; the grid itself is independent
    of this ordering.

    The storage need for a tensor grid is only the sum of the lengths
    of the coordinate vectors, while the total number of points is
    the product of these lengths. This class makes use of this
    sparse storage.
    """

    def __init__(self, *coord_vectors):
        """Initialize a new instance.

        Parameters
        ----------
        vec1,...,vecN : `array-like`
            The coordinate vectors defining the grid points. They must
            be sorted in ascending order and may not contain
            duplicates. Empty vectors are not allowed.

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
        >>> g.size  # total number of points
        9

        Grid points can be extracted with index notation (NOTE: This is
        slow, do not loop over the grid using indices!):

        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> g[0, 0, 0, 0]
        array([-1.,  2.,  5.,  2.])

        Slices and ellipsis are also supported:

        >>> g[:, 0, 0, 0]
        TensorGrid([-1.0, 0.0, 3.0], [2.0], [5.0], [2.0])
        >>> g[0, ..., 1:]
        TensorGrid([-1.0], [2.0, 4.0], [5.0], [4.0, 7.0])
        """
        if not coord_vectors:
            raise ValueError('no coordinate vectors given')

        vecs = tuple(np.atleast_1d(vec).astype('float64')
                     for vec in coord_vectors)
        for i, vec in enumerate(vecs):

            if len(vec) == 0:
                raise ValueError('vector {} has zero length'
                                 ''.format(i + 1))

            if not np.all(np.isfinite(vec)):
                raise ValueError('vector {} contains invalid entries'
                                 ''.format(i + 1))

            if vec.ndim != 1:
                raise ValueError('vector {} has {} dimensions instead of 1'
                                 ''.format(i + 1, vec.ndim))

            sorted_vec = np.sort(vec)
            if np.any(vec != sorted_vec):
                raise ValueError('vector {} not sorted'
                                 ''.format(i + 1))

            if np.any(np.diff(vec) == 0):
                raise ValueError('vector {} contains duplicates'
                                 ''.format(i + 1))

        self.__coord_vectors = vecs
        self.__ideg = np.array([i for i in range(len(vecs))
                               if len(vecs[i]) == 1])
        self.__inondeg = np.array([i for i in range(len(vecs))
                                  if len(vecs[i]) != 1])

    # Attributes
    @property
    def coord_vectors(self):
        """Coordinate vectors of the grid.

        Returns
        -------
        coord_vectors : tuple of `numpy.ndarray`'s

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> x, y = g.coord_vectors
        >>> x
        array([ 0.,  1.])
        >>> y
        array([-1.,  0.,  2.])

        See Also
        --------
        meshgrid : Same result but with nd arrays
        """
        return self.__coord_vectors

    @property
    def ndim(self):
        """Number of dimensions of the grid."""
        return len(self.coord_vectors)

    @property
    def shape(self):
        """Number of grid points per axis."""
        return tuple(len(vec) for vec in self.coord_vectors)

    @property
    def size(self):
        """Total number of grid points."""
        return np.prod(self.shape)

    def __len__(self):
        """Return ``len(self)``.

        The length along the first dimension.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2], [4, 5, 6])
        >>> len(g)
        2

        See Also
        --------
        size : The total number of elements.
        """
        return self.shape[0]

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
        array([ 5.,  2.])
        """
        return np.array([vec[-1] for vec in self.coord_vectors])

    @property
    def ideg(self):
        """Indices of the degenerate axes of this grid."""
        return self.__ideg

    @property
    def inondeg(self):
        """Indices of the non-degenerate axes of this grid."""
        return self.__inondeg

    # min, max and extent are for set duck-typing
    def min(self, **kwargs):
        """Return `min_pt`.

        Parameters
        ----------
        kwargs
            For duck-typing with `numpy.amin`

        See Also
        --------
        max
        odl.set.domain.IntervalProd.min

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.min()
        array([ 1., -2.])

        Also works with numpy

        >>> np.min(g)
        array([ 1., -2.])
        """
        out = kwargs.get('out', None)
        if out is not None:
            out[:] = self.min_pt
            return out
        else:
            return self.min_pt

    def max(self, **kwargs):
        """Return `max_pt`.

        Parameters
        ----------
        kwargs
            For duck-typing with `numpy.amax`

        See Also
        --------
        min
        odl.set.domain.IntervalProd.max

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.max()
        array([ 5.,  2.])

        Also works with numpy

        >>> np.max(g)
        array([ 5.,  2.])
        """
        out = kwargs.get('out', None)
        if out is not None:
            out[:] = self.max_pt
            return out
        else:
            return self.max_pt

    def extent(self):
        """Return the edge lengths of this grid's minimal bounding box.

        Examples
        --------
        >>> g = TensorGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.extent()
        array([ 4.,  4.])
        """
        return self.max_pt - self.min_pt

    def convex_hull(self):
        """Return the smallest `IntervalProd` containing this grid.

        The convex hull of a set is the union of all line segments
        between points in the set. For a tensor grid, it is the
        interval product given by the extremal coordinates.

        Returns
        -------
        chull : `IntervalProd`
            Interval product defined by the minimum and maximum of
            the grid

        Examples
        --------
        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> g.convex_hull()
        IntervalProd([-1.0, 2.0, 5.0, 2.0], [3.0, 4.0, 5.0, 7.0])
        """
        return IntervalProd(self.min(), self.max())

    def element(self):
        """An arbitrary element, the minimum coordinates."""
        return self.min_pt

    def approx_equals(self, other, atol):
        """Test if this grid is equal to another grid.

        Parameters
        ----------
        other :
            Object to be tested
        atol : float
            Allow deviations up to this number in absolute value
            per vector entry.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `TensorGrid` instance with all
            coordinate vectors equal (up to the given tolerance), to
            the ones of this grid, ``False`` otherwise.

        Examples
        --------
        >>> g1 = TensorGrid([0, 1], [-1, 0, 2])
        >>> g2 = TensorGrid([-0.1, 1.1], [-1, 0.1, 2])
        >>> g1.approx_equals(g2, atol=0)
        False
        >>> g1.approx_equals(g2, atol=0.15)
        True
        """
        if other is self:
            return True

        return (isinstance(other, TensorGrid) and
                self.ndim == other.ndim and
                self.shape == other.shape and
                all(np.allclose(vec_s, vec_o, atol=atol, rtol=0.0)
                    for (vec_s, vec_o) in zip(self.coord_vectors,
                                              other.coord_vectors)))

    def __eq__(self, other):
        """Return ``self == other``.
        """
        # Implemented separately for performance reasons
        if other is self:
            return True

        return (isinstance(other, TensorGrid) and
                self.shape == other.shape and
                all(np.array_equal(vec_s, vec_o)
                    for (vec_s, vec_o) in zip(self.coord_vectors,
                                              other.coord_vectors)))

    def approx_contains(self, other, atol):
        """Test if ``other`` belongs to this grid up to a tolerance.

        Parameters
        ----------
        other : `array-like` or float
            The object to test for membership in this grid
        atol : float
            Allow deviations up to this number in absolute value
            per vector entry.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.approx_contains([0, 0], atol=0.0)
        True
        >>> [0, 0] in g  # equivalent
        True
        >>> g.approx_contains([0.1, -0.1], atol=0.0)
        False
        >>> g.approx_contains([0.1, -0.1], atol=0.15)
        True
        """
        other = np.atleast_1d(other)
        return (other.shape == (self.ndim,) and
                all(np.any(np.isclose(vector, coord, atol=atol, rtol=0.0))
                    for vector, coord in zip(self.coord_vectors, other)))

    def __contains__(self, other):
        """Return ``other in self``."""
        other = np.atleast_1d(other)
        if other.dtype == np.dtype(object):
            return False
        return (other.shape == (self.ndim,) and
                all(coord in vector
                    for vector, coord in zip(self.coord_vectors, other)))

    def is_subgrid(self, other, atol=0.0):
        """Test if this grid is contained in another grid.

        Parameters
        ----------
        other :  `TensorGrid`
            The other grid which is supposed to contain this grid
        atol : float
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
        >>> g_sub.is_subgrid(g, atol=0.15)
        True
        """
        # Optimization for some common cases
        if other is self:
            return True
        if not (isinstance(other, TensorGrid) and
                np.all(self.shape <= other.shape) and
                np.all(self.min_pt >= other.min_pt - atol) and
                np.all(self.max_pt <= other.max_pt + atol)):
            return False

        # Array version of the fuzzy subgrid test, about 3 times faster
        # than the loop version.
        for vec_o, vec_s in zip(other.coord_vectors, self.coord_vectors):
            # Create array of differences of all entries in vec_o and vec_s.
            # If there is no almost zero entry in each row, return False.
            vec_o_mg, vec_s_mg = sparse_meshgrid(vec_o, vec_s)
            if not np.all(np.any(np.abs(vec_s_mg - vec_o_mg) <= atol, axis=0)):
                return False

        return True

    def insert(self, index, other):
        """Return a copy with ``other`` inserted before ``index``.

        The given grid (``m`` dimensions) is inserted into the current
        one (``n`` dimensions) before the given index, resulting in a new
        `TensorGrid` with ``n + m`` dimensions.
        Note that no changes are made in-place.

        Parameters
        ----------
        index : int
            The index of the dimension before which ``other`` is to
            be inserted. Negative indices count backwards from
            ``self.ndim``.
        other :  `TensorGrid`
            The grid to be inserted

        Returns
        -------
        newgrid : `TensorGrid`
            The enlarged grid

        Examples
        --------
        >>> g1 = TensorGrid([0, 1], [-1, 2])
        >>> g2 = TensorGrid([1], [-6, 15])
        >>> g1.insert(1, g2)
        TensorGrid([0.0, 1.0], [1.0], [-6.0, 15.0], [-1.0, 2.0])

        See Also
        --------
        append
        """
        idx = int(index)
        # Support backward indexing
        if idx < 0:
            idx = self.ndim + idx
        if not 0 <= idx <= self.ndim:
            raise IndexError('index {} outside the valid range 0 ... {}'
                             ''.format(idx, self.ndim))

        new_vecs = (self.coord_vectors[:idx] + other.coord_vectors +
                    self.coord_vectors[idx:])
        return TensorGrid(*new_vecs)

    def append(self, other):
        """Insert grid ``other`` at the end.

        Parameters
        ----------
        other : `TensorGrid`
            Set to be inserted.

        Examples
        --------
        >>> g1 = TensorGrid([0, 1], [-1, 2])
        >>> g2 = TensorGrid([1], [-6, 15])
        >>> g1.append(g2)
        TensorGrid([0.0, 1.0], [-1.0, 2.0], [1.0], [-6.0, 15.0])

        See Also
        --------
        insert
        """
        return self.insert(self.ndim, other)

    def squeeze(self):
        """Return the grid with removed degenerate (length 1) dimensions.

        Returns
        -------
        squeezed : `TensorGrid`
            Squeezed grid.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1], [-1, 0, 2])
        >>> g.squeeze()
        TensorGrid([0.0, 1.0], [-1.0, 0.0, 2.0])
        """
        coord_vecs = [self.coord_vectors[axis] for axis in self.inondeg]
        return TensorGrid(*coord_vecs)

    def points(self, order='C'):
        """All grid points in a single array.

        Parameters
        ----------
        order : {'C', 'F'}
            Axis ordering in the resulting point array

        Returns
        -------
        points : `numpy.ndarray`
            The shape of the array is ``size x ndim``, i.e. the points
            are stored as rows.

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
        if str(order).upper() not in ('C', 'F'):
            raise ValueError('order {!r} not recognized'.format(order))
        else:
            order = str(order).upper()

        axes = range(self.ndim) if order == 'C' else reversed(range(self.ndim))
        shape = self.shape if order == 'C' else tuple(reversed(self.shape))
        point_arr = np.empty((self.size, self.ndim))

        for i, axis in enumerate(axes):
            view = point_arr[:, axis].reshape(shape)
            coord_shape = (1,) * i + (-1,) + (1,) * (self.ndim - i - 1)
            view[:] = self.coord_vectors[axis].reshape(coord_shape)

        return point_arr

    def corner_grid(self):
        """Return a grid with only the corner points.

        Returns
        -------
        cgrid : `TensorGrid`
            Grid with size 2 in non-degenerate dimensions and 1
            in degenerate ones

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.corner_grid()
        TensorGrid([0.0, 1.0], [-1.0, 2.0])
        """
        minmax_vecs = []
        for axis in range(self.ndim):
            if self.shape[axis] == 1:
                minmax_vecs.append(self.coord_vectors[axis][0])
            else:
                minmax_vecs.append((self.coord_vectors[axis][0],
                                    self.coord_vectors[axis][-1]))

        return TensorGrid(*minmax_vecs)

    def corners(self, order='C'):
        """Corner points of the grid in a single array.

        Parameters
        ----------
        order : {'C', 'F'}
            Axis ordering in the resulting point array

        Returns
        -------
        corners : `numpy.ndarray`
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
        return self.corner_grid().points(order=order)

    @property
    def meshgrid(self):
        """A grid suitable for function evaluation.

        Returns
        -------
        meshgrid : tuple of `numpy.ndarray`'s
            Function evaluation grid with ``ndim`` axes

        See Also
        --------
        numpy.meshgrid
            Coordinate matrices from coordinate vectors.
            We use ``indexing='ij'`` and ``copy=True``

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> x, y = g.meshgrid
        >>> x
        array([[ 0.],
               [ 1.]])
        >>> y
        array([[-1.,  0.,  2.]])

        Easy function evaluation via broadcasting:

        >>> x**2 - y**2
        array([[-1.,  0., -4.],
               [ 0.,  1., -3.]])
        """
        return sparse_meshgrid(*self.coord_vectors)

    @property
    def is_uniform(self):
        """Return ``True`` if this grid is a `RegularGrid`."""
        return isinstance(self, RegularGrid)

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Object determining which parts of the grid to extract.
            ``None`` (new axis) and empty axes are not supported.

        Examples
        --------
        Indexing with integers along all axes produces an array (a point):

        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> g[0, 0, 0, 0]
        array([-1.,  2.,  5.,  2.])

        Otherwise, a new TensorGrid is returned:

        >>> g[:, 0, 0, 0]
        TensorGrid([-1.0, 0.0, 3.0], [2.0], [5.0], [2.0])
        >>> g[0, ..., 1:]
        TensorGrid([-1.0], [2.0, 4.0], [5.0], [4.0, 7.0])
        >>> g[::2, ..., ::2]
        TensorGrid([-1.0, 3.0], [2.0, 4.0], [5.0], [2.0, 7.0])

        Too few indices are filled up with an ellipsis from the right:

        >>> g[0]
        TensorGrid([-1.0], [2.0, 4.0], [5.0], [2.0, 4.0, 7.0])
        >>> g[0] == g[0, :, :, :] == g[0, ...]
        True
        """
        if isinstance(indices, list):
            new_coord_vecs = [self.coord_vectors[0][indices]]
            new_coord_vecs += self.coord_vectors[1:]
            return TensorGrid(*new_coord_vecs)

        indices = normalized_index_expression(indices, self.shape,
                                              int_to_slice=False)

        # If all indices are integers, return an array (a point). Otherwise,
        # create a new grid.
        if all(np.isscalar(idx) for idx in indices):
            return np.fromiter(
                (v[int(idx)] for idx, v in zip(indices, self.coord_vectors)),
                dtype=float)
        else:
            new_coord_vecs = [vec[idx]
                              for idx, vec in zip(indices, self.coord_vectors)]
            return TensorGrid(*new_coord_vecs)

    def __array__(self, dtype=None):
        """Used with ``numpy``. Returns `points`.

        This allows usage of tensorgrid with some numpy functions.

        Parameters
        ----------
        dtype : `numpy.dtype`
            The Numpy data type of the result array. ``None`` means `float`.

        Examples
        --------
        >>> g = TensorGrid([0, 1], [-2, 0, 2])

        Convert to an array:

        >>> np.asarray(g)
        array([[ 0., -2.],
               [ 0.,  0.],
               [ 0.,  2.],
               [ 1., -2.],
               [ 1.,  0.],
               [ 1.,  2.]])

        Calculate the midpoint:

        >>> np.mean(g, axis=0)
        array([ 0.5,  0. ])
        """
        return self.points().astype(dtype)

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = ', '.join(array1d_repr(vec) for vec in self.coord_vectors)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        grid_str = ' x '.join(array1d_str(vec) for vec in self.coord_vectors)
        return 'grid ' + grid_str


class RegularGrid(TensorGrid):

    """An n-dimensional tensor grid with equidistant coordinates.

    This is a sparse representation of an n-dimensional grid defined
    as the tensor product of n coordinate vectors with equidistant
    nodes. The grid points are calculated according to the rule::

        x_j = min_pt + j * (max_pt - min_pt) / (shape - 1)

    with elementwise addition and multiplication.
    """

    def __init__(self, min_pt, max_pt, shape):
        """Initialize a new instance.

        Parameters
        ----------
        min_pt, max_pt : float or sequence of floats
            Points defining the minimum/maximum grid coordinates.
        shape : int or sequence of ints
            Number of grid points per axis.

        Examples
        --------
        >>> rg = odl.RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg
        RegularGrid([-1.5, -1.0], [-0.5, 3.0], (2, 3))
        >>> rg.coord_vectors
        (array([-1.5, -0.5]), array([-1.,  1.,  3.]))
        >>> rg.ndim, rg.size
        (2, 6)

        In 1D, we don't need sequences:

        >>> rg = odl.RegularGrid(0, 1, 10)
        >>> rg.shape
        (10,)
        """
        min_pt = np.atleast_1d(min_pt).astype('float64')
        max_pt = np.atleast_1d(max_pt).astype('float64')
        shape = np.atleast_1d(shape).astype('int64', casting='safe')

        if any(x.ndim != 1 for x in (min_pt, max_pt, shape)):
            raise ValueError('input arrays have dimensions {}, {}, {} '
                             'instead of 1'
                             ''.format(min_pt.ndim, max_pt.ndim, shape.ndim))

        if not min_pt.shape == max_pt.shape == shape.shape:
            raise ValueError('input array shapes {}, {}, {} are not equal'
                             ''.format(min_pt.shape, max_pt.shape,
                                       shape.shape))

        if not np.all(np.isfinite(min_pt)):
            raise ValueError('minimum point {} has invalid entries'
                             ''.format(min_pt))

        if not np.all(np.isfinite(max_pt)):
            raise ValueError('maximum point {} has invalid entries'
                             ''.format(max_pt))

        if not np.all(min_pt <= max_pt):
            raise ValueError('minimum point {} has entries larger than '
                             'those of maximum point {}'
                             ''.format(min_pt, max_pt))

        if np.any(shape <= 0):
            raise ValueError('shape parameter is {} but should only contain '
                             'positive values'.format(shape))

        degen = np.where(min_pt == max_pt)[0]
        if np.any(shape[degen] != 1):
            raise ValueError('degenerated axes {} with shapes {}, expected '
                             '{}'.format(tuple(degen[:]), tuple(shape[degen]),
                                         len(degen) * (1,)))

        coord_vecs = [np.linspace(mi, ma, num, endpoint=True, dtype=np.float64)
                      for mi, ma, num in zip(min_pt, max_pt, shape)]
        TensorGrid.__init__(self, *coord_vecs)
        self.__mid_pt = (self.max_pt + self.min_pt) / 2
        self.__stride = np.zeros(len(shape), dtype='float64')
        idcs = np.where(shape > 1)
        self.__stride[idcs] = ((self.max_pt - self.min_pt)[idcs] /
                               (shape[idcs] - 1))

    @property
    def mid_pt(self):
        """Midpoint of the grid. Not necessarily a grid point.

        Examples
        --------
        >>> rg = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg.mid_pt
        array([-1.,  1.])
        """
        return self.__mid_pt

    @property
    def stride(self):
        """Step per axis between two neighboring grid points.

        Examples
        --------
        >>> rg = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg.stride
        array([ 1.,  2.])
        """
        return self.__stride

    def is_subgrid(self, other, atol=0.0):
        """Test if this grid is contained in another grid.

        Parameters
        ----------
        other :
            Check if this object is a subgrid
        atol : float
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
        >>> rg_sub.is_subgrid(rg, atol=0.01)
        False
        >>> rg_sub.is_subgrid(rg, atol=0.02)
        True
        """
        # Optimize some common cases
        if other is self:
            return True
        if not (isinstance(other, TensorGrid) and
                np.all(self.shape <= other.shape) and
                np.all(self.min_pt >= other.min_pt - atol) and
                np.all(self.max_pt <= other.max_pt + atol)):
            return False

        # For regular grids, it suffices to show that min_pt, max_pt
        # and g[1,...,1] are contained in the other grid. For axes
        # with less than 2 points, this reduces to min_pt and max_pt,
        # and the corresponding indices in the other check point are
        # set to 0.
        if isinstance(other, RegularGrid):
            minmax_contained = (
                other.approx_contains(self.min_pt, atol=atol) and
                other.approx_contains(self.max_pt, atol=atol))
            check_idx = np.zeros(self.ndim, dtype=int)
            check_idx[np.array(self.shape) >= 3] = 1
            checkpt_contained = other.approx_contains(self[tuple(check_idx)],
                                                      atol=atol)

            return minmax_contained and checkpt_contained

        elif isinstance(other, TensorGrid):
            # other is a TensorGrid, we need to fall back to full check
            self_tg = TensorGrid(*self.coord_vectors)
            return self_tg.is_subgrid(other)

    def insert(self, index, other):
        """Insert another regular grid before the given index.

        The given grid (``m`` dimensions) is inserted into the current
        one (``n`` dimensions) before the given index, resulting in a new
        `RegularGrid` with ``n + m`` dimensions.
        Note that no changes are made in-place.

        Parameters
        ----------
        index : int
            Index of the dimension before which ``other`` is to
            be inserted. Negative indices count backwards from
            ``self.ndim``.
        other : `TensorGrid`
            Grid to be inserted. If a `RegularGrid` is given,
            the output will be a `RegularGrid`.

        Returns
        -------
        newgrid : `TensorGrid` or `RegularGrid`
            The enlarged grid. If the inserted grid is a `RegularGrid`,
            so is the return value.

        Examples
        --------
        >>> rg1 = RegularGrid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg2 = RegularGrid(-3, 7, 6)
        >>> rg1.insert(1, rg2)
        RegularGrid([-1.5, -3.0, -1.0], [-0.5, 7.0, 3.0], (2, 6, 3))

        If other is a TensorGrid, so is the result:

        >>> tg = TensorGrid([0, 1, 2])
        >>> rg1.insert(2, tg)
        TensorGrid([-1.5, -0.5], [-1.0, 1.0, 3.0], [0.0, 1.0, 2.0])
        """
        if isinstance(other, RegularGrid):
            idx = int(index)

            # Support backward indexing
            if idx < 0:
                idx = self.ndim + idx
            if not 0 <= idx <= self.ndim:
                raise IndexError('index out of valid range 0 -> {}'
                                 ''.format(self.ndim))

            new_shape = self.shape[:idx] + other.shape + self.shape[idx:]
            new_min_pt = (list(self.min_pt[:idx]) + list(other.min_pt) +
                          list(self.min_pt[idx:]))
            new_max_pt = (list(self.max_pt[:idx]) + list(other.max_pt) +
                          list(self.max_pt[idx:]))
            return RegularGrid(new_min_pt, new_max_pt, new_shape)

        elif isinstance(other, TensorGrid):
            self_tg = TensorGrid(*self.coord_vectors)
            return self_tg.insert(index, other)

        else:
            raise TypeError('{!r} is not a TensorGrid instance'.format(other))

    def squeeze(self):
        """Return the grid with removed degenerate dimensions.

        Returns
        -------
        squeezed : `RegularGrid`
            Squeezed grid

        Examples
        --------
        >>> g = RegularGrid([0, 0, 0], [1, 0, 1], (5, 1, 5))
        >>> g.squeeze()
        RegularGrid([0.0, 0.0], [1.0, 1.0], (5, 5))
        """
        sq_minpt = [self.min_pt[axis] for axis in self.inondeg]
        sq_maxpt = [self.max_pt[axis] for axis in self.inondeg]
        sq_shape = [self.shape[axis] for axis in self.inondeg]
        return RegularGrid(sq_minpt, sq_maxpt, sq_shape)

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Object determining which parts of the grid to extract.
            ``None`` (new axis) and empty axes are not supported.

        Examples
        --------
        Indexing with integers along all axes produces an array (a point):

        >>> g = RegularGrid([-1.5, -3, -1], [-0.5, 7, 4], (2, 3, 6))
        >>> g[0, 0, 0]
        array([-1.5, -3. , -1. ])

        Otherwise, a new RegularGrid is returned:

        >>> g[:, 0, 0]
        RegularGrid([-1.5, -3.0, -1.0], [-0.5, -3.0, -1.0], (2, 1, 1))

        Ellipsis can be used, too:

        >>> g[..., ::3]
        RegularGrid([-1.5, -3.0, -1.0], [-0.5, 7.0, 2.0], (2, 3, 2))
        """
        # Index list results in a tensor grid
        if isinstance(indices, list):
            return super().__getitem__(indices)

        indices = normalized_index_expression(indices, self.shape,
                                              int_to_slice=False)

        # If all indices are integers, return an array (a point). Otherwise,
        # create a new grid.
        if all(np.isscalar(idx) for idx in indices):
            return np.fromiter(
                (v[int(idx)] for idx, v in zip(indices, self.coord_vectors)),
                dtype=float)
        else:
            new_min_pt = []
            new_max_pt = []
            new_shape = []

            for idx, xmin, xmax, n, cvec in zip(
                    indices, self.min_pt, self.max_pt, self.shape,
                    self.coord_vectors):
                if np.isscalar(idx):
                    idx = slice(idx, idx + 1)

                if idx == slice(None):  # Take all along this axis
                    new_min_pt.append(xmin)
                    new_max_pt.append(xmax)
                    new_shape.append(n)
                else:  # Slice
                    istart, istop, istep = idx.indices(n)

                    if istart == istop:
                        num = 1
                        last = istart
                    else:
                        num = int(np.ceil((istop - istart) / istep))
                        last = istart + (num - 1) * istep

                    new_min_pt.append(cvec[istart])
                    new_max_pt.append(cvec[last])
                    new_shape.append(num)

            return RegularGrid(new_min_pt, new_max_pt, new_shape)

    def __str__(self):
        """Return ``str(self)``."""
        grid_str = ' x '.join(array1d_str(vec) for vec in self.coord_vectors)
        return 'regular grid ' + grid_str

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '{}, {}, {}'.format(list(self.min_pt), list(self.max_pt),
                                        tuple(self.shape))
        return '{}({})'.format(self.__class__.__name__, inner_str)


def uniform_sampling_fromintv(intv_prod, shape, nodes_on_bdry=True):
    """Sample an interval product uniformly.

    The resulting grid will include ``intv_prod.min_pt`` and
    ``intv_prod.max_pt`` as grid points. If you want a subdivision into
    equally sized cells with grid points in the middle, use
    `uniform_partition` instead.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Set to be sampled
    shape : int or sequence of ints
        Number of nodes per axis. Entries corresponding to degenerate axes
        must be equal to 1.
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

    Returns
    -------
    sampling : `RegularGrid`
        Uniform sampling grid for the interval product

    See Also
    --------
    uniform_sampling:
        sample an implicitly defined interval product
    odl.discr.partition.uniform_partition_fromintv :
        divide interval product into equally sized subsets

    Examples
    --------
    >>> rbox = odl.IntervalProd([-1.5, 2], [-0.5, 3])
    >>> grid = uniform_sampling_fromintv(rbox, (3, 3))
    >>> grid.coord_vectors
    (array([-1.5, -1. , -0.5]), array([ 2. ,  2.5,  3. ]))

    To have the nodes in the "middle", use ``nodes_on_bdry=False``:

    >>> grid = uniform_sampling_fromintv(rbox, (2, 2), nodes_on_bdry=False)
    >>> grid.coord_vectors
    (array([-1.25, -0.75]), array([ 2.25,  2.75]))


    See Also
    --------
    uniform_sampling : Sample an implicitly created `IntervalProd`
    """
    shape = np.atleast_1d(shape).astype('int64', casting='safe')

    if not isinstance(intv_prod, IntervalProd):
        raise TypeError('{!r} is not an `IntervalProd` instance'
                        ''.format(intv_prod))

    if np.shape(nodes_on_bdry) == ():
        nodes_on_bdry = ([(bool(nodes_on_bdry), bool(nodes_on_bdry))] *
                         intv_prod.ndim)
    elif intv_prod.ndim == 1 and len(nodes_on_bdry) == 2:
        nodes_on_bdry = [nodes_on_bdry]
    elif len(nodes_on_bdry) != intv_prod.ndim:
        raise ValueError('`nodes_on_bdry` has length {}, expected {}'
                         ''.format(len(nodes_on_bdry), intv_prod.ndim))
    else:
        shape = tuple(int(n) for n in shape)

    # We need to determine the placement of the grid minimum and maximum
    # points based on the choices in nodes_on_bdry. If in a given axis,
    # and for a given side (left or right), the entry is True, the node lies
    # on the boundary, so this coordinate can simply be taken as-is.
    #
    # Otherwise, the following conditions must be met:
    #
    # 1. The node should be half a stride s away from the boundary
    # 2. Adding or subtracting (n-1)*s should give the other extremal node.
    #
    # If both nodes are to be shifted half a stride inside,
    # the second condition yields
    # a + s/2 + (n-1)*s = b - s/2 => s = (b - a) / n,
    # hence the extremal grid points are
    # gmin = a + s/2 = a + (b - a) / (2 * n),
    # gmax = b - s/2 = b - (b - a) / (2 * n).
    #
    # In the case where one node, say the rightmost, lies on the boundary,
    # the condition 2. reads as
    # a + s/2 + (n-1)*s = b => s = (b - a) / (n - 1/2),
    # thus
    # gmin = a + (b - a) / (2 * n - 1).

    gmin, gmax = [], []
    for n, xmin, xmax, on_bdry in zip(shape, intv_prod.min_pt,
                                      intv_prod.max_pt, nodes_on_bdry):

        # Unpack the tuple if possible, else use bool globally for this axis
        try:
            bdry_l, bdry_r = on_bdry
        except TypeError:
            bdry_l = bdry_r = on_bdry

        if bdry_l and bdry_r:
            gmin.append(xmin)
            gmax.append(xmax)
        elif bdry_l and not bdry_r:
            gmin.append(xmin)
            gmax.append(xmax - (xmax - xmin) / (2 * n - 1))
        elif not bdry_l and bdry_r:
            gmin.append(xmin + (xmax - xmin) / (2 * n - 1))
            gmax.append(xmax)
        else:
            gmin.append(xmin + (xmax - xmin) / (2 * n))
            gmax.append(xmax - (xmax - xmin) / (2 * n))

    return RegularGrid(gmin, gmax, shape)


def uniform_sampling(min_pt, max_pt, shape, nodes_on_bdry=True):
    """Sample an implicitly defined interval product uniformly.

    Parameters
    ----------
    min_pt, max_pt : float or sequence of float
        Vectors of lower/upper ends of the intervals in the product.
    shape : int or sequence of ints
        Number of nodes per axis. Entries corresponding to degenerate axes
        must be equal to 1.
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

    See Also
    --------
    uniform_sampling_fromintv :
        sample a given interval product
    odl.discr.partition.uniform_partition :
        divide implicitly defined interval product into equally
        sized subsets

    Examples
    --------
    >>> grid = odl.uniform_sampling([-1.5, 2], [-0.5, 3], (3, 3))
    >>> grid.coord_vectors
    (array([-1.5, -1. , -0.5]), array([ 2. ,  2.5,  3. ]))

    To have the nodes in the "middle", use ``nodes_on_bdry=False``:

    >>> grid = odl.uniform_sampling([-1.5, 2], [-0.5, 3], (2, 2),
    ...                             nodes_on_bdry=False)
    >>> grid.coord_vectors
    (array([-1.25, -0.75]), array([ 2.25,  2.75]))

    In 1D, we don't need sequences:

    >>> grid = odl.uniform_sampling(0, 1, 3)
    >>> grid.coord_vectors
    (array([ 0. ,  0.5,  1. ]),)
    """
    return uniform_sampling_fromintv(IntervalProd(min_pt, max_pt), shape,
                                     nodes_on_bdry=nodes_on_bdry)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
