# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
from odl.util import (
    normalized_index_expression, normalized_scalar_param_list, safe_int_conv,
    array1d_repr, array1d_str, signature_string, indent_rows)


__all__ = ('RectGrid', 'uniform_grid', 'uniform_grid_fromintv')


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


class RectGrid(Set):

    """An n-dimensional rectilinear grid.

    A rectilinear grid is the set of points defined by all possible
    combination of coordinates taken from fixed coordinate vectors.

    The storage need for a rectilinear grid is only the sum of the lengths
    of the coordinate vectors, while the total number of points is
    the product of these lengths. This class makes use of that
    sparse storage scheme.

    See ``Notes`` for details.
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
        >>> g = RectGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g
        RectGrid(
            [1.0, 2.0, 5.0],
            [-2.0, 1.5, 2.0]
        )
        >>> g.ndim  # number of axes
        2
        >>> g.shape  # points per axis
        (3, 3)
        >>> g.size  # total number of points
        9

        Grid points can be extracted with index notation (NOTE: This is
        slow, do not loop over the grid using indices!):

        >>> g = RectGrid([-1, 0, 3], [2, 4, 5], [5], [2, 4, 7])
        >>> g[0, 0, 0, 0]
        array([-1.,  2.,  5.,  2.])

        Slices and ellipsis are also supported:

        >>> g[:, 0, 0, 0]
        RectGrid(
            [-1.0, 0.0, 3.0],
            [2.0],
            [5.0],
            [2.0]
        )
        >>> g[0, ..., 1:]
        RectGrid(
            [-1.0],
            [2.0, 4.0, 5.0],
            [5.0],
            [4.0, 7.0]
        )

        Notes
        -----
        In 2 dimensions, for example, given two coordinate vectors

        .. math::
            v_1 = (-1, 0, 2),\ v_2 = (0, 1)

        the corresponding rectilinear grid :math:`G` is the set of all
        2d points whose first component is from :math:`v_1` and the
        second component from :math:`v_2`:

        .. math::
            G = \{(-1, 0), (-1, 1), (0, 0), (0, 1), (2, 0), (2, 1)\}

        Here is a graphical representation::

               :    :        :
               :    :        :
            1 -x----x--------x-...
               |    |        |
            0 -x----x--------x-...
               |    |        |
              -1    0        2

        Apparently, this structure can represent grids with arbitrary step
        sizes in each axis.

        Note that the above ordering of points is the standard ``'C'``
        ordering where the first axis (:math:`v_1`) varies slowest.
        Ordering is only relevant when the point array is actually created;
        the grid itself is independent of this ordering.
        """
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

        # Non-degenerate axes
        self.__nondegen_byaxis = tuple(len(v) > 1 for v in self.coord_vectors)

        # Uniformity, setting True in degenerate axes
        diffs = [np.diff(v) for v in self.coord_vectors]
        self.__is_uniform_byaxis = tuple(
            (diff.size == 0) or np.allclose(diff, diff[0])
            for diff in diffs)

    # Attributes
    @property
    def coord_vectors(self):
        """Coordinate vectors of the grid.

        Returns
        -------
        coord_vectors : tuple of `numpy.ndarray`'s

        Examples
        --------
        >>> g = RectGrid([0, 1], [-1, 0, 2])
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
        # Since np.prod(()) == 1.0 we need to handle that by ourselves
        return 0 if self.shape == () else np.prod(self.shape)

    def __len__(self):
        """Return ``len(self)``.

        The length along the first dimension.

        Examples
        --------
        >>> g = RectGrid([0, 1], [-1, 0, 2], [4, 5, 6])
        >>> len(g)
        2

        See Also
        --------
        size : The total number of elements.
        """
        return 0 if self.shape == () else self.shape[0]

    @property
    def min_pt(self):
        """Vector containing the minimal grid coordinates per axis.

        Examples
        --------
        >>> g = RectGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.min_pt
        array([ 1., -2.])
        """
        return np.array([vec[0] for vec in self.coord_vectors])

    @property
    def max_pt(self):
        """Vector containing the maximal grid coordinates per axis.

        Examples
        --------
        >>> g = RectGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.max_pt
        array([ 5.,  2.])
        """
        return np.array([vec[-1] for vec in self.coord_vectors])

    @property
    def nondegen_byaxis(self):
        """Boolean array with ``True`` entries for non-degenerate axes.

        Examples
        --------
        >>> g = uniform_grid([0, 0], [1, 1], (5, 1))
        >>> g.nondegen_byaxis
        (True, False)
        """
        return self.__nondegen_byaxis

    @property
    def is_uniform_byaxis(self):
        """Boolean tuple showing uniformity of this grid per axis."""
        return self.__is_uniform_byaxis

    @property
    def is_uniform(self):
        """``True`` if this grid is uniform in all axes, else ``False``."""
        return all(self.is_uniform_byaxis)

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
        >>> g = RectGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.min()
        array([ 1., -2.])

        Also works with Numpy:

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
        >>> g = RectGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.max()
        array([ 5.,  2.])

        Also works with Numpy:

        >>> np.max(g)
        array([ 5.,  2.])
        """
        out = kwargs.get('out', None)
        if out is not None:
            out[:] = self.max_pt
            return out
        else:
            return self.max_pt

    @property
    def mid_pt(self):
        """Midpoint of the grid, not necessarily a grid point.

        Examples
        --------
        >>> rg = uniform_grid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg.mid_pt
        array([-1.,  1.])
        """
        return (self.max_pt + self.min_pt) / 2

    @property
    def stride(self):
        """Step per axis between neighboring points of a uniform grid.

        If the grid contains axes that are not uniform, ``stride`` has
        a ``NaN`` entry.

        Examples
        --------
        >>> rg = uniform_grid([-1.5, -1], [-0.5, 3], (2, 3))
        >>> rg.stride
        array([ 1.,  2.])
        >>> g = RectGrid([0, 1, 2], [0, 1, 4])
        >>> g.stride
        array([  1.,  nan])
        """
        # Cache for efficiency instead of re-computing
        try:
            strd = self.__stride
        except AttributeError:
            strd = []
            for i in range(self.ndim):
                if not self.is_uniform_byaxis[i]:
                    strd.append(float('nan'))
                elif self.nondegen_byaxis[i]:
                    strd.append(self.extent[i] / (self.shape[i] - 1.0))
                else:
                    strd.append(0.0)
            self.__stride = np.array(strd)
            return self.__stride.copy()
        else:
            return strd.copy()

    @property
    def extent(self):
        """Return the edge lengths of this grid's minimal bounding box.

        Examples
        --------
        >>> g = RectGrid([1, 2, 5], [-2, 1.5, 2])
        >>> g.extent
        array([ 4.,  4.])
        """
        return self.max_pt - self.min_pt

    def convex_hull(self):
        """Return the smallest `IntervalProd` containing this grid.

        The convex hull of a set is the union of all line segments
        between points in the set. For a rectilinear grid, it is the
        interval product given by the extremal coordinates.

        Returns
        -------
        convex_hull : `IntervalProd`
            Interval product defined by the minimum and maximum points
            of the grid.

        Examples
        --------
        >>> g = RectGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
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
            ``True`` if ``other`` is a `RectGrid` instance with all
            coordinate vectors equal (up to the given tolerance), to
            the ones of this grid, ``False`` otherwise.

        Examples
        --------
        >>> g1 = RectGrid([0, 1], [-1, 0, 2])
        >>> g2 = RectGrid([-0.1, 1.1], [-1, 0.1, 2])
        >>> g1.approx_equals(g2, atol=0)
        False
        >>> g1.approx_equals(g2, atol=0.15)
        True
        """
        if other is self:
            return True

        return (isinstance(other, RectGrid) and
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

        return (isinstance(other, RectGrid) and
                self.shape == other.shape and
                all(np.array_equal(vec_s, vec_o)
                    for (vec_s, vec_o) in zip(self.coord_vectors,
                                              other.coord_vectors)))

    def __hash__(self):
        """Return ``hash(self)``."""
        # TODO: update with #841
        coord_vec_str = tuple(cv.tostring() for cv in self.coord_vectors)
        return hash((type(self), coord_vec_str))

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
        >>> g = RectGrid([0, 1], [-1, 0, 2])
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
        """Return ``True`` if this grid is a subgrid of ``other``.

        Parameters
        ----------
        other :  `TensorGrid`
            The other grid which is supposed to contain this grid
        atol : float, optional
            Allow deviations up to this number in absolute value
            per coordinate vector entry.

        Returns
        -------
        is_subgrid : bool
            ``True`` if all coordinate vectors of ``self`` are within
            absolute distance ``atol`` of the other grid, else ``False``.

        Examples
        --------
        >>> rg = uniform_grid([-2, -2], [0, 4], (3, 4))
        >>> rg.coord_vectors
        (array([-2., -1.,  0.]), array([-2.,  0.,  2.,  4.]))
        >>> rg_sub = uniform_grid([-1, 2], [0, 4], (2, 2))
        >>> rg_sub.coord_vectors
        (array([-1.,  0.]), array([ 2.,  4.]))
        >>> rg_sub.is_subgrid(rg)
        True

        Fuzzy check is also possible. Note that the tolerance still
        applies to the coordinate vectors.

        >>> rg_sub = uniform_grid([-1.015, 2], [0, 3.99], (2, 2))
        >>> rg_sub.is_subgrid(rg, atol=0.01)
        False
        >>> rg_sub.is_subgrid(rg, atol=0.02)
        True
        """
        # Optimization for some common cases
        if other is self:
            return True
        if not isinstance(other, RectGrid):
            return False
        if not all(self.shape[i] <= other.shape[i] and
                   self.min_pt[i] >= other.min_pt[i] - atol and
                   self.max_pt[i] <= other.max_pt[i] + atol
                   for i in range(self.ndim)):
            return False
        if self.size == 0:
            return True

        if self.is_uniform and other.is_uniform:
            # For uniform grids, it suffices to show that min_pt, max_pt
            # and g[1,...,1] are contained in the other grid. For axes
            # with less than 2 points, this reduces to min_pt and max_pt,
            # and the corresponding indices in the other check point are
            # set to 0.
            minmax_contained = (
                other.approx_contains(self.min_pt, atol=atol) and
                other.approx_contains(self.max_pt, atol=atol))
            check_idx = np.zeros(self.ndim, dtype=int)
            check_idx[np.array(self.shape) >= 3] = 1
            checkpt_contained = other.approx_contains(self[tuple(check_idx)],
                                                      atol=atol)
            return minmax_contained and checkpt_contained

        else:
            # Array version of the fuzzy subgrid test, about 3 times faster
            # than the loop version.
            for vec_o, vec_s in zip(other.coord_vectors, self.coord_vectors):
                # Create array of differences of all entries in vec_o and
                # vec_s. If there is no almost zero entry in each row,
                # return False.
                vec_o_mg, vec_s_mg = sparse_meshgrid(vec_o, vec_s)
                if not np.all(np.any(np.isclose(vec_s_mg, vec_o_mg, atol=atol),
                                     axis=0)):
                    return False
            return True

    def insert(self, index, *grids):
        """Return a copy with ``grids`` inserted before ``index``.

        The given grids are inserted (as a block) into ``self``, yielding
        a new grid whose number of dimensions is the sum of the numbers of
        dimensions of all involved grids.
        Note that no changes are made in-place.

        Parameters
        ----------
        index : int
            The index of the dimension before which ``grids`` are to
            be inserted. Negative indices count backwards from
            ``self.ndim``.
        grid1, ..., gridN :  `RectGrid`
            The grids to be inserted into ``self``.

        Returns
        -------
        newgrid : `RectGrid`
            The enlarged grid.

        Examples
        --------
        >>> g1 = RectGrid([0, 1], [-1, 0, 2])
        >>> g2 = RectGrid([1], [-6, 15])
        >>> g1.insert(1, g2)
        RectGrid(
            [0.0, 1.0],
            [1.0],
            [-6.0, 15.0],
            [-1.0, 0.0, 2.0]
        )
        >>> g1.insert(1, g2, g2)
        RectGrid(
            [0.0, 1.0],
            [1.0],
            [-6.0, 15.0],
            [1.0],
            [-6.0, 15.0],
            [-1.0, 0.0, 2.0]
        )

        See Also
        --------
        append
        """
        index, index_in = safe_int_conv(index), index
        if not -self.ndim <= index <= self.ndim:
            raise IndexError('index {0} outside the valid range -{1} ... {1}'
                             ''.format(index_in, self.ndim))
        if index < 0:
            index += self.ndim

        if len(grids) > 1:
            return self.insert(index, grids[0]).insert(
                index + grids[0].ndim, *(grids[1:]))
        else:
            grid = grids[0]
            if not isinstance(grid, RectGrid):
                raise TypeError('{!r} is not a `RectGrid` instance'
                                ''.format(grid))
            new_vecs = (self.coord_vectors[:index] + grid.coord_vectors +
                        self.coord_vectors[index:])
            return RectGrid(*new_vecs)

    def append(self, *grids):
        """Insert ``grids`` at the end as a block.

        Parameters
        ----------
        grid1, ..., gridN :  `RectGrid`
            The grids to be appended to ``self``.

        Returns
        -------
        newgrid : `RectGrid`
            The enlarged grid.

        Examples
        --------
        >>> g1 = RectGrid([0, 1], [-1, 0, 2])
        >>> g2 = RectGrid([1], [-6, 15])
        >>> g1.append(g2)
        RectGrid(
            [0.0, 1.0],
            [-1.0, 0.0, 2.0],
            [1.0],
            [-6.0, 15.0]
        )
        >>> g1.append(g2, g2)
        RectGrid(
            [0.0, 1.0],
            [-1.0, 0.0, 2.0],
            [1.0],
            [-6.0, 15.0],
            [1.0],
            [-6.0, 15.0]
        )

        See Also
        --------
        insert
        """
        return self.insert(self.ndim, *grids)

    def squeeze(self):
        """Return the grid with removed degenerate (length 1) dimensions.

        Returns
        -------
        squeezed : `RectGrid`
            Squeezed grid.

        Examples
        --------
        >>> g = RectGrid([0, 1], [-1], [-1, 0, 2])
        >>> g.squeeze()
        RectGrid(
            [0.0, 1.0],
            [-1.0, 0.0, 2.0]
        )

        """
        nondegen_indcs = [i for i in range(self.ndim)
                          if self.nondegen_byaxis[i]]
        coord_vecs = [self.coord_vectors[axis] for axis in nondegen_indcs]
        return RectGrid(*coord_vecs)

    def points(self, order='C'):
        """All grid points in a single array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Axis ordering in the resulting point array

        Returns
        -------
        points : `numpy.ndarray`
            The shape of the array is ``size x ndim``, i.e. the points
            are stored as rows.

        Examples
        --------
        >>> g = RectGrid([0, 1], [-1, 0, 2])
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
        cgrid : `RectGrid`
            Grid with size 2 in non-degenerate dimensions and 1
            in degenerate ones

        Examples
        --------
        >>> g = RectGrid([0, 1], [-1, 0, 2])
        >>> g.corner_grid()
        uniform_grid([0.0, -1.0], [1.0, 2.0], (2, 2))
        """
        minmax_vecs = []
        for axis in range(self.ndim):
            if self.shape[axis] == 1:
                minmax_vecs.append(self.coord_vectors[axis][0])
            else:
                minmax_vecs.append((self.coord_vectors[axis][0],
                                    self.coord_vectors[axis][-1]))

        return RectGrid(*minmax_vecs)

    def corners(self, order='C'):
        """Corner points of the grid in a single array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Axis ordering in the resulting point array

        Returns
        -------
        corners : `numpy.ndarray`
            The size of the array is 2^m x ndim, where m is the number
            of non-degenerate axes, i.e. the corners are stored as rows.

        Examples
        --------
        >>> g = RectGrid([0, 1], [-1, 0, 2])
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
        >>> g = RectGrid([0, 1], [-1, 0, 2])
        >>> x, y = g.meshgrid
        >>> x
        array([[ 0.],
               [ 1.]])
        >>> y
        array([[-1.,  0.,  2.]])

        Easy function evaluation via broadcasting:

        >>> x ** 2 - y ** 2
        array([[-1.,  0., -4.],
               [ 0.,  1., -3.]])
        """
        return sparse_meshgrid(*self.coord_vectors)

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

        >>> g = RectGrid([-1, 0, 3], [2, 4, 5], [5], [2, 4, 7])
        >>> g[0, 0, 0, 0]
        array([-1.,  2.,  5.,  2.])

        Otherwise, a new RectGrid is returned:

        >>> g[:, 0, 0, 0]
        RectGrid(
            [-1.0, 0.0, 3.0],
            [2.0],
            [5.0],
            [2.0]
        )
        >>> g[0, ..., 1:]
        RectGrid(
            [-1.0],
            [2.0, 4.0, 5.0],
            [5.0],
            [4.0, 7.0]
        )
        >>> g[::2, ..., ::2]
        RectGrid(
            [-1.0, 3.0],
            [2.0, 4.0, 5.0],
            [5.0],
            [2.0, 7.0]
        )

        Too few indices are filled up with an ellipsis from the right:

        >>> g[0]
        RectGrid(
            [-1.0],
            [2.0, 4.0, 5.0],
            [5.0],
            [2.0, 4.0, 7.0]
        )
        >>> g[0] == g[0, :, :, :] == g[0, ...]
        True
        """
        if isinstance(indices, list):
            if indices == []:
                new_coord_vecs = []
            else:
                new_coord_vecs = [self.coord_vectors[0][indices]]
                new_coord_vecs += self.coord_vectors[1:]
            return RectGrid(*new_coord_vecs)

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
            return RectGrid(*new_coord_vecs)

    def __array__(self, dtype=None):
        """Used with ``numpy``. Returns `points`.

        This allows usage of RectGrid with some numpy functions.

        Parameters
        ----------
        dtype : `numpy.dtype`
            The Numpy data type of the result array. ``None`` means `float`.

        Examples
        --------
        >>> g = RectGrid([0, 1], [-2, 0, 2])

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
        if self.is_uniform:
            constructor = 'uniform_grid'
            posargs = [list(self.min_pt), list(self.max_pt), self.shape]
            inner_str = signature_string(posargs, [])
            return '{}({})'.format(constructor, inner_str)
        else:
            constructor = self.__class__.__name__
            posargs = [array1d_repr(v) for v in self.coord_vectors]
            inner_str = signature_string(posargs, [], sep=[',\n', ', ', ', '],
                                         mod=['!s', ''])
            return '{}(\n{}\n)'.format(constructor, indent_rows(inner_str))

    __str__ = __repr__


def uniform_grid_fromintv(intv_prod, shape, nodes_on_bdry=True):
    """Return a grid from sampling an interval product uniformly.

    The resulting grid will by default include ``intv_prod.min_pt`` and
    ``intv_prod.max_pt`` as grid points. If you want a subdivision into
    equally sized cells with grid points in the middle, use
    `uniform_partition` instead.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Set to be sampled.
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
    sampling : `RectGrid`
        Uniform sampling grid for the interval product.

    Examples
    --------
    >>> rbox = odl.IntervalProd([-1.5, 2], [-0.5, 3])
    >>> grid = uniform_grid_fromintv(rbox, (3, 3))
    >>> grid.coord_vectors
    (array([-1.5, -1. , -0.5]), array([ 2. ,  2.5,  3. ]))

    To have the nodes in the "middle", use ``nodes_on_bdry=False``:

    >>> grid = uniform_grid_fromintv(rbox, (2, 2), nodes_on_bdry=False)
    >>> grid.coord_vectors
    (array([-1.25, -0.75]), array([ 2.25,  2.75]))

    See Also
    --------
    uniform_grid : Create a uniform grid directly.
    odl.discr.partition.uniform_partition_fromintv :
        divide interval product into equally sized subsets
    """
    if not isinstance(intv_prod, IntervalProd):
        raise TypeError('{!r} is not an `IntervalProd` instance'
                        ''.format(intv_prod))

    if (np.any(np.isinf(intv_prod.min_pt)) or
            np.any(np.isinf(intv_prod.max_pt))):
        raise ValueError('`intv_prod` must be finite, got {!r}'
                         ''.format('intv_prod'))

    shape = normalized_scalar_param_list(shape, intv_prod.ndim, safe_int_conv)

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

    # Create the grid
    coord_vecs = [np.linspace(mi, ma, num, endpoint=True, dtype=np.float64)
                  for mi, ma, num in zip(gmin, gmax, shape)]
    return RectGrid(*coord_vecs)


def uniform_grid(min_pt, max_pt, shape, nodes_on_bdry=True):
    """Return a grid from sampling an implicit interval product uniformly.

    Parameters
    ----------
    min_pt : float or sequence of float
        Vectors of lower ends of the intervals in the product.
    max_pt : float or sequence of float
        Vectors of upper ends of the intervals in the product.
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
    uniform_grid : `RectGrid`
        The resulting uniform grid.

    See Also
    --------
    uniform_grid_fromintv :
        sample a given interval product
    odl.discr.partition.uniform_partition :
        divide implicitly defined interval product into equally
        sized subsets

    Examples
    --------
    By default, the min/max points are included in the grid:

    >>> grid = odl.uniform_grid([-1.5, 2], [-0.5, 3], (3, 3))
    >>> grid.coord_vectors
    (array([-1.5, -1. , -0.5]), array([ 2. ,  2.5,  3. ]))

    If ``shape`` is supposed to refer to small subvolumes, and the grid
    should be their centers, use the option ``nodes_on_bdry=False``:

    >>> grid = odl.uniform_grid([-1.5, 2], [-0.5, 3], (2, 2),
    ...                         nodes_on_bdry=False)
    >>> grid.coord_vectors
    (array([-1.25, -0.75]), array([ 2.25,  2.75]))

    In 1D, we don't need sequences:

    >>> grid = odl.uniform_grid(0, 1, 3)
    >>> grid.coord_vectors
    (array([ 0. ,  0.5,  1. ]),)
    """
    return uniform_grid_fromintv(IntervalProd(min_pt, max_pt), shape,
                                 nodes_on_bdry=nodes_on_bdry)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
