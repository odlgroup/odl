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

+-------------+-------------------------------------------------------+
|Class name   |Description                                            |
+=============+=======================================================+
|`TensorGrid` |The points are given as the tensor product of n        |
|             |coordinate vectors, i.e. all possible n-tuples where   |
|             |the i-th coordinate is from the i-th coordinate vector.|
+-------------+-------------------------------------------------------+
|`RegularGrid`|A variant of a tensor grid where the entries of each   |
|             |coordinate vector are equispaced. This type of grid can|
|             |be represented by three n-dimensional vectors `shape`, |
|             |`center` and `stride`.                                 |
+-------------+-------------------------------------------------------+
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External module imports
import numpy as np

# ODL imports
from odl.space.set import Set
from odl.utility.utility import errfmt, array1d_repr


class TensorGrid(Set):

    """An n-dimensional tensor grid.

    This is a sparse representation of a collection of n-dimensional points
    defined by the tensor product of n coordinate vectors.

    Example: x = (x_1, x_2), y = (y_1, y_2, y_3, y_4), z = (z_1, z_2, z_3).
    The resulting grid consists of all possible combinations
    p = (x_i, y_j, z_k), hence 2 * 4 * 3 = 24 points in total.


    Attributes
    ----------

    +---------------+----------------+--------------------------------+
    |Name           |Type            |Description                     |
    +===============+================+================================+
    |`coord_vectors`|`numpy.ndarray` |Vectors containing the grid     |
    |               |tuple           |point coordinates per axis      |
    +---------------+----------------+--------------------------------+
    |`dim`          |`int`           |Number of axes                  |
    +---------------+----------------+--------------------------------+
    |`shape`        |`int` tuple     |Number of grid points per axis  |
    +---------------+----------------+--------------------------------+
    |`ntotal`       |`int`           |Total number of grid points     |
    +---------------+----------------+--------------------------------+
    |`min`          |`numpy.ndarray` |Grid point with minimal         |
    |               |                |coordinates in each axis        |
    +---------------+----------------+--------------------------------+
    |`max`          |`numpy.ndarray` |Grid point with maximal         |
    |               |                |coordinates in each axis        |
    +---------------+----------------+--------------------------------+


    Methods
    -------

    +-------------------+---------------+-----------------------------+
    |Signature          |Return type    |Description                  |
    +===================+===============+=============================+
    |`equals(other,     |boolean        |Test if `other` is a grid    |
    |tol=0.0)`          |               |with the same points up to   |
    |                   |               |a tolerance of `tol` per     |
    |                   |               |point. Equivalent to         |
    |                   |               |`self == other` for          |
    |                   |               |`tol==0.0`.                  |
    +-------------------+---------------+-----------------------------+
    |`contains(point,   |boolean        |Test if `point` is a grid    |
    |tol=0.0)`          |               |point with tolerance `tol`.  |
    |                   |               |Equivalent to `point in self`|
    |                   |               |for `tol==0`.                |
    +-------------------+---------------+-----------------------------+
    |`is_subgrid(other, |boolean        |Test if `other` is contained |
    |tol=0.0)`          |               |with tolerance `tol` per     |
    |                   |               |point.                       |
    +-------------------+---------------+-----------------------------+
    |`points(order='C')`|`numpy.ndarray`|All grid points in a single  |
    |                   |               |array                        |
    +-------------------+---------------+-----------------------------+
    |`corners           |`numpy.ndarray`|The corner points in a single|
    |(order='C')`       |               |array                        |
    +-------------------+---------------+-----------------------------+
    |`meshgrid          |`numpy.ndarray`|Efficient grid for function  |
    |(sparse=True)`     |tuple          |evaluation (see              |
    |                   |               |numpy.meshgrid)              |
    +-------------------+---------------+-----------------------------+
    |`convex_hull()`    |`IntervalProd` |The "inner" of the grid,     |
    |                   |               |a rectangular box.           |
    +-------------------+---------------+-----------------------------+
    """

    def __init__(self, *coord_vectors):
        """Initialize a TensorGrid instance.

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
                raise ValueError('Vector {} has zero length.'.format(i+1))

            if not np.all(np.isfinite(vec)):
                raise ValueError(errfmt('''
                Vector {} contains invalid entries'''.format(i+1)))

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
    def element(self):
        """An arbitrary element, the minimum coordinates."""
        return self.min

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
        >>> g1.equals(g2, tol=0.15)
        True
        """
        # pylint: disable=arguments-differ
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

        Examples
        --------

        >>> g = TensorGrid([0, 1], [-1, 0, 2])
        >>> g.contains([0, 0])
        True
        >>> [0, 0] in g  # equivalent
        True
        >>> g.contains([0.1, -0.1])
        False
        >>> g.contains([0.1, -0.1], tol=0.15)
        True
        """
        # pylint: disable=arguments-differ
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
        if not (isinstance(other, TensorGrid) and
                np.all(self.shape <= other.shape) and
                np.all(self.min >= other.min - tol) and
                np.all(self.max <= other.max + tol)):
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
        """The corner points of the grid in a single array.

        Parameters
        ----------

        order : 'C' or 'F'
            The ordering of the axes in which the corners appear in
            the output.

        Returns
        -------

        out : numpy.ndarray
            The size of the array is 2^m x dim, where m is the number of
            non-degenerate axes, i.e. the corners are stored as rows.

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
        return minmax_grid.points(order=order)

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
        """The "inner" of the grid, an IntervalProd.

        The convex hull of a set is the union of all line segments
        between points in the set. For a tensor grid, it is the
        interval product given by the extremal coordinates.

        Parameters
        ----------

        None

        Returns
        -------

        chull : IntervalProd

        Examples
        --------

        >>> g = TensorGrid([-1, 0, 3], [2, 4], [5], [2, 4, 7])
        >>> ch = g.convex_hull()
        >>> ch.begin
        array([-1.,  2.,  5.,  2.])
        >>> ch.end
        array([ 3.,  4.,  5.,  7.])
        """
        from odl.space.set import IntervalProd
        beg, end = [], []
        for axis in range(self.dim):
            beg.append(self.coord_vectors[axis][0])
            end.append(self.coord_vectors[axis][-1])

        return IntervalProd(beg, end)

    def __getitem__(self, slc):
        """self[slc] implementation.

        Parameters
        ----------

        slc : int or slice
            Negative indices and 'None' (new axis) are not supported.

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
            if len(idx) < self.dim:
                raise IndexError(errfmt('''
                Too few indices ({} < {}).'''.format(len(idx), self.dim)))
            elif len(idx) > self.dim:
                raise IndexError(errfmt('''
                Too many indices ({} > {}).'''.format(len(idx), self.dim)))

            return np.array([v[i] for i, v in zip(idx, self.coord_vectors)])

        except TypeError:
            pass

        if Ellipsis in slc_list:
            if slc_list.count(Ellipsis) > 1:
                raise IndexError("Cannot use more than one ellipsis.")
            if len(slc_list) == self.dim + 1:  # Ellipsis without effect
                ellipsis_idx = self.dim
                num_after_ellipsis = 0
            else:
                ellipsis_idx = slc_list.index(Ellipsis)
                num_after_ellipsis = len(slc_list) - ellipsis_idx - 1
            slc_list.remove(Ellipsis)

        else:
            if len(slc_list) < self.dim:
                raise IndexError(errfmt('''
                Too few axes ({} < {}).'''.format(len(slc_list), self.dim)))
            ellipsis_idx = self.dim
            num_after_ellipsis = 0

        if any(s.start == s.stop and s.start is not None
               for s in slc_list if isinstance(s, slice)):
            raise IndexError('Slices with empty axes not allowed.')

        if len(slc_list) > self.dim:
            raise IndexError(errfmt('''
            Too many axes ({} > {}).'''.format(len(slc_list), self.dim)))

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

    +---------------+----------------+--------------------------------+
    |Name           |Type            |Description                     |
    +===============+================+================================+
    |`coord_vectors`|`numpy.ndarray` |Vectors containing the grid     |
    |               |tuple           |point coordinates per axis      |
    +---------------+----------------+--------------------------------+
    |`dim`          |`int`           |Number of axes                  |
    +---------------+----------------+--------------------------------+
    |`shape`        |`int` tuple     |Number of grid points per axis  |
    +---------------+----------------+--------------------------------+
    |`center`       |`numpy.ndarray` |The grid's symmetry center (not |
    |               |                |necessarily a grid point)       |
    +---------------+----------------+--------------------------------+
    |`stride`       |`numpy.ndarray` |Vector pointing from `x[j]` to  |
    |               |                |`x[j + (1,...,1)]`              |
    +---------------+----------------+--------------------------------+
    |`ntotal`       |`int`           |Total number of grid points     |
    +---------------+----------------+--------------------------------+
    |`min`          |`numpy.ndarray` |Grid point with minimal         |
    |               |                |coordinates in each axis        |
    +---------------+----------------+--------------------------------+
    |`max`          |`numpy.ndarray` |Grid point with maximal         |
    |               |                |coordinates in each axis        |
    +---------------+----------------+--------------------------------+

    Methods
    -------

    +-------------------+---------------+-----------------------------+
    |Signature          |Return type    |Description                  |
    +===================+===============+=============================+
    |`equals(other,     |boolean        |Test if `other` is a grid    |
    |tol=0.0)`          |               |with the same points up to   |
    |                   |               |a tolerance of `tol` per     |
    |                   |               |point. Equivalent to         |
    |                   |               |`self == other` for          |
    |                   |               |`tol==0.0`.                  |
    +-------------------+---------------+-----------------------------+
    |`contains(point,   |boolean        |Test if `point` is a grid    |
    |tol=0.0)`          |               |point with tolerance `tol`.  |
    |                   |               |Equivalent to `point in self`|
    |                   |               |for `tol==0`.                |
    +-------------------+---------------+-----------------------------+
    |`is_subgrid(other, |boolean        |Test if `other` is contained |
    |tol=0.0)`          |               |with tolerance `tol` per     |
    |                   |               |point.                       |
    +-------------------+---------------+-----------------------------+
    |`points(order='C')`|`numpy.ndarray`|All grid points in a single  |
    |                   |               |array                        |
    +-------------------+---------------+-----------------------------+
    |`corners           |`numpy.ndarray`|The corner points in a single|
    |(order='C')`       |               |array                        |
    +-------------------+---------------+-----------------------------+
    |`meshgrid          |`numpy.ndarray`|Efficient grid for function  |
    |(sparse=True)`     |tuple          |evaluation (see              |
    |                   |               |numpy.meshgrid)              |
    +-------------------+---------------+-----------------------------+
    |`convex_hull()`    |`IntervalProd` |The "inner" of the grid,     |
    |                   |               |a rectangular box.           |
    +-------------------+---------------+-----------------------------+
    """

    def __init__(self, shape, center=None, stride=None):
        """Initialize a RegularGrid instance.

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

        Examples
        --------

        >>> rg = RegularGrid((2, 3), center=[-1, 1], stride=[1, 2])
        >>> rg.coord_vectors
        (array([-1.5, -0.5]), array([-1.,  1.,  3.]))
        >>> rg.dim, rg.ntotal
        (2, 6)
        >>> rg = RegularGrid((2, 3), center=[-1, 1], stride=[-1, 2])
        Traceback (most recent call last):
        ...
        ValueError: 'stride' may only have positive entries.

        Default for 'center' is [0,...,0]:

        >>> rg = RegularGrid((2, 3), stride=[1, 2])
        >>> rg.coord_vectors
        (array([-0.5,  0.5]), array([-2.,  0.,  2.]))

        Default for 'stride' is [1,...,1]:

        >>> rg = RegularGrid((2, 3), center=[-1, 1])
        >>> rg.coord_vectors
        (array([-1.5, -0.5]), array([ 0.,  1.,  2.]))
        """
        shape = np.atleast_1d(shape).astype(np.int64)
        if not np.all(shape > 0):
            raise ValueError("'shape' may only have positive entries.")

        if center is None:
            center = np.zeros_like(shape).astype(np.float64)
        else:
            center = np.atleast_1d(center).astype(np.float64)
            if len(center) != len(shape):
                raise ValueError(errfmt('''
                'center' ({}) must have the same length as 'shape' ({}).
                '''.format(len(center), len(shape))))

        if not np.all(np.isfinite(center)):
            raise ValueError("'center' has invalid entries.")

        if stride is None:
            stride = np.ones_like(shape, dtype=np.float64)
        else:
            stride = np.atleast_1d(stride).astype(np.float64)
            if len(stride) != len(shape):
                raise ValueError(errfmt('''
                'stride' ({}) must have the same length as 'shape' ({}).
                '''.format(len(stride), len(shape))))

        if not np.all(np.isfinite(stride)):
            raise ValueError("'stride' has invalid entries.")
        if not np.all(stride > 0):
            raise ValueError("'stride' may only have positive entries.")

        coord_vecs = [ct + (np.arange(shp, dtype=np.float64)-(shp-1)/2) * st
                      for ct, st, shp in zip(center, stride, shape)]
        super().__init__(*coord_vecs)

        self._center = center
        self._stride = stride

    @property
    def center(self):
        """The center of the grid. Not necessarily a grid point.

        Example
        -------

        >>> rg = RegularGrid((2, 3), [-1, 1], [1, 2])
        >>> rg.center
        array([-1.,  1.])
        """
        return self._center

    @property
    def stride(self):
        """The step per axis between two neighboring grid points.

        Example
        -------

        >>> rg = RegularGrid((2, 3), [-1, 1], [1, 2])
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

        >>> rg = RegularGrid((3, 4), [-1, 1], [1, 2])
        >>> rg.coord_vectors
        (array([-2., -1.,  0.]), array([-2.,  0.,  2.,  4.]))
        >>> rg_sub = RegularGrid((2, 2), [-1, 1], [2, 2])
        >>> rg_sub.coord_vectors
        (array([-2.,  0.]), array([ 0.,  2.]))
        >>> rg_sub.is_subgrid(rg)
        True

        Fuzzy check is also possible. Note that the tolerance still
        applies to the coordinate vectors, not 'stride' and 'center'.

        >>> rg_sub = RegularGrid((2, 4), [-1, 1], [2.01, 2.01])
        >>> # stride error accumulates towards the endpoints
        >>> rg_sub.coord_vectors[1]
        array([-2.015, -0.005,  2.005,  4.015])
        >>> rg_sub.is_subgrid(rg, tol=0.01)
        False
        >>> rg_sub.is_subgrid(rg, tol=0.02)
        True
        """
        # Optimize some common cases
        if not (isinstance(other, TensorGrid) and
                np.all(self.shape <= other.shape) and
                np.all(self.min >= other.min - tol) and
                np.all(self.max <= other.max + tol)):
            return False

        if isinstance(other, RegularGrid):
            # Do a full check for axes with less than 3 points since
            # errors can cancel out
            idcs = np.where(np.array(self.shape) <= 3)[0]
            if idcs.tolist():
                self_tg = TensorGrid(*[self.coord_vectors[i] for i in idcs])
                other_tg = TensorGrid(*[other.coord_vectors[i] for i in idcs])
                if not self_tg.is_subgrid(other_tg, tol=tol):
                    return False

            # Further checks are restricted to axes with more than 3 points
            idcs = np.where(np.array(self.shape) > 3)[0]
            if idcs.tolist():
                # Test corners, here the error is largest
                corners = self.corners()[idcs]
                if not all(other.contains(c, tol=tol) for c in corners):
                    return False

                # stride must be an integer multiple of other's stride
                stride_mult = np.around(self.stride[idcs] / other.stride[idcs])
                if not np.allclose(
                        self.stride[idcs] - stride_mult*other.stride[idcs], 0,
                        rtol=0, atol=tol):
                    return False

                # Center shift has to be multiple of other's stride
                # plus half a stride if the shape difference parity
                # is the same as the stride multiple parity (see above)
                cshift = other.center[idcs] - self.center[idcs]
                stride_mult_par = stride_mult % 2
                shape_diff_par = (np.array(other.shape)[idcs] -
                                  np.array(self.shape)[idcs]) % 2
                par_corr = np.zeros_like(idcs, dtype=np.float64)
                par_corr[stride_mult_par == shape_diff_par] = 0.5
                cshift_mult = np.around(
                    cshift / other.stride[idcs] - par_corr)
                if not np.allclose(
                        cshift - (par_corr + cshift_mult)*other.stride[idcs],
                        0, rtol=0, atol=tol):
                    return False

        elif isinstance(other, TensorGrid):
            # other is a TensorGrid, we need to fall back to full check
            # pylint: disable=unbalanced-tuple-unpacking
            for vec_o, vec_s in zip(other.coord_vectors, self.coord_vectors):
                vec_o_mg, vec_s_mg = np.meshgrid(vec_o, vec_s, sparse=True,
                                                 copy=True, indexing='ij')
                if not np.all(np.any(np.abs(vec_s_mg - vec_o_mg) <= tol,
                                     axis=0)):
                    return False
        return True

    def __getitem__(self, slc):
        """self[slc] implementation.

        Parameters
        ----------

        slc : int or slice
            Negative indices and 'None' (new axis) are not supported.

        Examples
        --------

        >>> g = RegularGrid([3, 1, 9], stride=[0.5, 2, 3])
        >>> g[0, 0, 0]
        array([ -0.5,   0. , -12. ])
        >>> g[:, 0, 0]
        RegularGrid([3, 1, 1], [0.0, 0.0, -12.0], [0.5, 2.0, 3.0])
        >>> g[:, 0, 0].coord_vectors
        (array([-0.5,  0. ,  0.5]), array([ 0.]), array([-12.]))

        Ellipsis can be used, too:

        >>> g[1:, ..., ::3]
        RegularGrid([2, 1, 3], [0.25, 0.0, -3.0], [0.5, 2.0, 9.0])
        >>> g[1:, ..., ::3].coord_vectors
        (array([ 0. ,  0.5]), array([ 0.]), array([-12.,  -3.,   6.]))
        """
        from math import ceil

        slc_list = list(np.atleast_1d(np.s_[slc]))
        if None in slc_list:
            raise IndexError('Creation of new axes not supported.')

        try:
            idx = np.array(slc_list, dtype=int)  # All single indices
            if len(idx) < self.dim:
                raise IndexError(errfmt('''
                Too few indices ({} < {}).'''.format(len(idx), self.dim)))
            elif len(idx) > self.dim:
                raise IndexError(errfmt('''
                Too many indices ({} > {}).'''.format(len(idx), self.dim)))

            return np.array([v[i] for i, v in zip(idx, self.coord_vectors)])

        except TypeError:
            pass

        if Ellipsis in slc_list:
            if slc_list.count(Ellipsis) > 1:
                raise IndexError("Cannot use more than one ellipsis.")
            if len(slc_list) == self.dim + 1:  # Ellipsis without effect
                ellipsis_idx = self.dim
                num_after_ellipsis = 0
            else:
                ellipsis_idx = slc_list.index(Ellipsis)
                num_after_ellipsis = len(slc_list) - ellipsis_idx - 1
            slc_list.remove(Ellipsis)

        else:
            if len(slc_list) < self.dim:
                raise IndexError(errfmt('''
                Too few axes ({} < {}).'''.format(len(slc_list), self.dim)))
            ellipsis_idx = self.dim
            num_after_ellipsis = 0

        if any(s.start == s.stop and s.start is not None
               for s in slc_list if isinstance(s, slice)):
            raise IndexError('Slices with empty axes not allowed.')

        if len(slc_list) > self.dim:
            raise IndexError(errfmt('''
            Too many axes ({} > {}).'''.format(len(slc_list), self.dim)))

        new_shape, new_center, new_stride = -np.ones((3, self.dim))

        # Copy axes corresponding to ellipsis
        ell_idcs = np.arange(ellipsis_idx, self.dim - num_after_ellipsis)
        new_shape[ell_idcs] = np.array(self.shape)[ell_idcs]
        new_center[ell_idcs] = self.center[ell_idcs]
        new_stride[ell_idcs] = self.stride[ell_idcs]

        # The other indices
        for i in range(ellipsis_idx):
            if isinstance(slc_list[i], slice):
                istart, istop, istep = slc_list[i].indices(self.shape[i])
            else:
                istart, istop, istep = slc_list[i], slc_list[i]+1, 1
            new_shape[i] = ceil((istop - istart) / istep)
            new_stride[i] = istep * self.stride[i]
            new_center[i] = (self.min[i] + istart * self.stride[i] +
                             (new_shape[i]-1)/2 * new_stride[i])

        for i in range(1, num_after_ellipsis + 1):
            i = -i
            if isinstance(slc_list[i], slice):
                istart, istop, istep = slc_list[i].indices(self.shape[i])
            else:
                istart, istop, istep = slc_list[i], slc_list[i]+1, 1
            new_shape[i] = ceil((istop - istart) / istep)
            new_stride[i] = istep * self.stride[i]
            new_center[i] = (self.min[i] + istart * self.stride[i] +
                             (new_shape[i]-1)/2 * new_stride[i])

        return RegularGrid(new_shape, new_center, new_stride)

    def __repr__(self):
        """repr(self) implementation."""
        return 'RegularGrid({}, {}, {})'.format(list(self.shape),
                                                list(self.center),
                                                list(self.stride))


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
