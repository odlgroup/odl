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

"""Domains for continuous functions. """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import super, zip
from future import standard_library
from future.utils import raise_from
standard_library.install_aliases()

import numpy as np

from odl.set.sets import Set
from odl.util.utility import array1d_repr
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid)


__all__ = ('IntervalProd', 'Interval', 'Rectangle', 'Cuboid')


class IntervalProd(Set):

    """An n-dimensional rectangular box.

    An interval product is a Cartesian product of n intervals, i.e. an
    n-dimensional rectangular box aligned with the coordinate axes
    as a subset of the n-dimensional Euclidean space.

    `IntervalProd` objects are immutable, hence all manipulation methods
    return a new instance.
    """

    def __init__(self, begin, end):
        """Initialize a new instance.

        Parameters
        ----------
        begin : array-like or float
            The lower ends of the intervals. A float can be used in
            one dimension.
        end : array-like or float
            The upper ends of the intervals. A float can be used in
            one dimension.

        Examples
        --------
        >>> b, e = [-1, 2.5, 70, 80], [-0.5, 10, 75, 90]
        >>> rbox = IntervalProd(b, e)
        >>> rbox
        IntervalProd([-1.0, 2.5, 70.0, 80.0], [-0.5, 10.0, 75.0, 90.0])
        """
        self.__begin = np.atleast_1d(begin).astype('float64')
        self.__end = np.atleast_1d(end).astype('float64')

        if self.begin.ndim > 1:
            raise ValueError('`begin` must be 1-dimensional, got an array '
                             'with {} axes'.format(self.begin.ndim))
        if self.end.ndim > 1:
            raise ValueError('`end` must be 1-dimensional, got an array '
                             'with {} axes'.format(self.begin.ndim))
        if len(self.begin) != len(self.end):
            raise ValueError('`begin` and `end` have different lengths '
                             '({} != {})'
                             ''.format(len(self.begin), len(self.end)))
        for i, (beg, end) in enumerate(zip(self.begin, self.end)):
            if beg > end:
                raise ValueError('in axis {}: `begin` is larger than `end` '
                                 '({} > {})'.format(i, beg, end))

        self.__ideg = np.where(self.begin == self.end)[0]
        self.__inondeg = np.where(self.begin != self.end)[0]
        super().__init__()

    @property
    def begin(self):
        """Left interval boundary/boundaries."""
        return self.__begin

    @property
    def end(self):
        """Right interval boundary/boundaries."""
        return self.__end

    @property
    def ndim(self):
        """Number of intervals in the product."""
        return len(self.begin)

    @property
    def true_ndim(self):
        """Number of non-degenerate (positive-length) intervals."""
        return len(self.inondeg)

    @property
    def volume(self):
        """`ndim`-dimensional volume of this interval product."""
        return self.measure(ndim=self.ndim)

    @property
    def length(self):
        """Length of this interval (valid for ``ndim == 1``)."""
        if self.ndim != 1:
            raise NotImplementedError('length not defined if `ndim` != 1')
        return self.volume

    @property
    def area(self):
        """Area of this rectangle (valid if ``ndim == 2``)."""
        if self.ndim != 2:
            raise NotImplementedError('area not defined if `ndim` != 2')
        return self.volume

    @property
    def midpoint(self):
        """Midpoint of this interval product."""
        midp = (self.end + self.begin) / 2.
        midp[self.ideg] = self.begin[self.ideg]
        return midp

    @property
    def ideg(self):
        """Indices of the degenerate dimensions."""
        return self.__ideg

    @property
    def inondeg(self):
        """Indices of the non-degenerate dimensions."""
        return self.__inondeg

    def min(self):
        """Return the minimum point of this interval product."""
        return self.begin

    def max(self):
        """Return the maximum point of this interval product."""
        return self.end

    def extent(self):
        """Return the vector of interval lengths per axis."""
        return self.max() - self.min()

    def element(self, inp=None):
        """Return an element of this interval product.

        Parameters
        ----------
        inp : float or array-like, optional
            Point to be cast to an element.

        Returns
        -------
        element : numpy.ndarray or float
            Array (ndim > 1) or float version of ``inp`` if provided,
            otherwise ``self.midpoint``.

        Examples
        --------
        >>> interv = IntervalProd(0, 1)
        >>> interv.element(0.5)
        0.5
        """
        if inp is None:
            return self.midpoint
        elif inp in self:
            if self.ndim == 1:
                return float(inp)
            else:
                return np.asarray(inp)
        else:
            raise TypeError('`inp` {!r} is not a valid element of {!r}'
                            ''.format(inp, self))

    def approx_equals(self, other, atol):
        """Return ``True`` if ``other`` is equal to this set up to ``atol``.

        Parameters
        ----------
        other :
            Object to be tested.
        atol : float
            Maximum allowed difference in maximum norm between the
            interval endpoints.

        Examples
        --------
        >>> from math import sqrt
        >>> rbox1 = IntervalProd(0, 0.5)
        >>> rbox2 = IntervalProd(0, sqrt(0.5)**2)
        >>> rbox1.approx_equals(rbox2, atol=0)  # Numerical error
        False
        >>> rbox1.approx_equals(rbox2, atol=1e-15)
        True
        """
        if other is self:
            return True
        elif not isinstance(other, IntervalProd):
            return False

        return (np.allclose(self.begin, other.begin, atol=atol, rtol=0.0) and
                np.allclose(self.end, other.end, atol=atol, rtol=0.0))

    def __eq__(self, other):
        """Return ``self == other``."""
        return self.approx_equals(other, atol=0.0)

    def approx_contains(self, point, atol):
        """Return ``True`` if ``point`` is "almost" contained in this set.

        Parameters
        ----------
        point : array-like or float
            Point to be tested. Its length must be equal to `ndim`.
            In the 1d case, ``point`` can be given as a float.
        atol : float
            Maximum allowed distance in maximum norm from ``point``
            to ``self``.

        Examples
        --------
        >>> from math import sqrt
        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervalProd(b, e)
        >>> # Numerical error
        >>> rbox.approx_contains([-1 + sqrt(0.5)**2, 0., 2.9], atol=0)
        False
        >>> rbox.approx_contains([-1 + sqrt(0.5)**2, 0., 2.9], atol=1e-9)
        True
        """
        try:
            # Duck-typed check of type
            point = np.array(point, dtype=np.float, copy=False, ndmin=1)
        except (ValueError, TypeError):
            return False

        if point.shape != (self.ndim,):
            return False

        return self.dist(point, exponent=np.inf) <= atol

    def __contains__(self, other):
        """Return ``other in self``.

        Examples
        --------
        >>> interv = IntervalProd(0, 1)
        >>> 0.5 in interv
        True
        >>> 2 in interv
        False
        >>> 'string' in interv
        False
        """
        try:
            # Duck-typed check of type
            point = np.array(other, dtype=np.float, copy=False, ndmin=1)
        except (ValueError, TypeError):
            return False

        if point.shape != (self.ndim,):
            return False
        return (self.begin <= point).all() and (point <= self.end).all()

    def contains_set(self, other, atol=0.0):
        """Return ``True`` if ``other`` is (almost) contained in this set.

        Parameters
        ----------
        other : `Set`
            Set to be tested.
        atol : float, optional
            Maximum allowed distance in maximum norm from ``other``
            to ``self``.

        Raises
        ------
        AttributeError
            if ``other`` does not have both ``min`` and ``max`` methods.

        Examples
        --------
        >>> b1, e1 = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox1 = IntervalProd(b1, e1)
        >>> b2, e2 = [-0.6, 0, 2.1], [-0.5, 0, 2.5]
        >>> rbox2 = IntervalProd(b2, e2)
        >>> rbox1.contains_set(rbox2)
        True
        >>> rbox2.contains_set(rbox1)
        False
        """
        if self is other:
            return True

        try:
            return (self.approx_contains(other.min(), atol) and
                    self.approx_contains(other.max(), atol))
        except AttributeError as err:
            raise_from(
                AttributeError('cannot test {!r} without `min` and `max` '
                               'methods'.format(other)), err)

    def contains_all(self, other, atol=0.0):
        """Return ``True`` if all points defined by ``other`` are contained.

        Parameters
        ----------
        other :
            Collection of points to be tested. Can be given as a single
            point, a ``(d, N)`` array-like where ``d`` is the
            number of dimensions, or a length-``d`` `meshgrid` tuple.
        atol : `float`, optional
            The maximum allowed distance in 'inf'-norm between the
            other set and this interval product.

        Returns
        -------
        contains : bool
            True if all points are contained, False otherwise.

        Examples
        --------
        >>> import odl
        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervalProd(b, e)

        Arrays are expected in (ndim, npoints) shape:

        >>> arr = np.array([[-1, 0, 2],   # defining one point at a time
        ...                 [-0.5, 0, 2]])
        >>> rbox.contains_all(arr.T)
        True

        Implicit meshgrids defined by coordinate vectors:

        >>> from odl.discr.grid import sparse_meshgrid
        >>> vec1 = (-1, -0.9, -0.7)
        >>> vec2 = (0, 0, 0)
        >>> vec3 = (2.5, 2.75, 3)
        >>> mg = sparse_meshgrid(vec1, vec2, vec3)
        >>> rbox.contains_all(mg)
        True

        Works also with an arbitrary iterable:

        >>> rbox.contains_all([[-1, -0.5], # define points by axis
        ...                    [0, 0],
        ...                    [2, 2]])
        True

        And with grids:

        >>> agrid = odl.uniform_sampling(rbox.begin, rbox.end, [3, 1, 3])
        >>> rbox.contains_all(agrid)
        True
        """
        atol = float(atol)

        # First try optimized methods
        if other in self:
            return True
        if hasattr(other, 'meshgrid'):
            return self.contains_all(other.meshgrid, atol=atol)
        elif is_valid_input_meshgrid(other, self.ndim):
            vecs = tuple(vec.squeeze() for vec in other)
            mins = np.fromiter((np.min(vec) for vec in vecs), dtype=float)
            maxs = np.fromiter((np.max(vec) for vec in vecs), dtype=float)
            return (np.all(mins >= self.begin - atol) and
                    np.all(maxs <= self.end + atol))

        # Convert to array and check each element
        other = np.asarray(other)
        if is_valid_input_array(other, self.ndim):
            if self.ndim == 1:
                mins = np.min(other)
                maxs = np.max(other)
            else:
                mins = np.min(other, axis=1)
                maxs = np.max(other, axis=1)
            return np.all(mins >= self.begin) and np.all(maxs <= self.end)
        else:
            return False

    def measure(self, ndim=None):
        """Return the Lebesgue measure of this interval product.

        Parameters
        ----------
        ndim : int, optional
            Dimension of the measure to apply. None is interpreted
            as `true_ndim`, which always results in a finite and
            positive result (unless the set is a single point).

        Examples
        --------
        >>> b, e = [-1, 2.5, 0], [-0.5, 10, 0]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.measure()
        3.75
        >>> rbox.measure(ndim=3)
        0.0
        >>> rbox.measure(ndim=3) == rbox.volume
        True
        >>> rbox.measure(ndim=1)
        inf
        >>> rbox.measure() == rbox.squeeze().volume
        True
        """
        if self.true_ndim == 0:
            return 0.0

        if ndim is None:
            return self.measure(ndim=self.true_ndim)
        elif ndim < self.true_ndim:
            return np.inf
        elif ndim > self.true_ndim:
            return 0.0
        else:
            return np.prod((self.end - self.begin)[self.inondeg])

    def dist(self, point, exponent=2.0):
        """Return the distance of ``point`` to this set.

        Parameters
        ----------
        point : array-like or float
            Point whose distance to calculate. Its length must be equal
            to the set's dimension. Can be a float in the 1d case.
        exponent : non-zero float or ``float('inf')``, optional
            Exponent of the norm used in the distance calculation.

        Returns
        -------
        dist : float
            Distance to the interior of the IntervalProd.
            Points strictly inside have distance ``0.0``, points with
            ``NaN`` have distance ``float('inf')``.

        See also
        --------
        numpy.linalg.norm : norm used to compute the distance

        Examples
        --------
        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.dist([-5, 3, 2])
        5.0
        >>> rbox.dist([-5, 3, 2], exponent=float('inf'))
        4.0
        """
        point = np.atleast_1d(point)
        if len(point) != self.ndim:
            raise ValueError('`point` must have length {}, got {}'
                             ''.format(self.ndim, len(point)))

        if np.any(np.isnan(point)):
            return float('inf')

        i_larger = np.where(point > self.end)
        i_smaller = np.where(point < self.begin)

        # Access [0] since np.where returns a tuple.
        if len(i_larger[0]) == 0 and len(i_smaller[0]) == 0:
            return 0.0
        else:
            proj = np.concatenate((point[i_larger], point[i_smaller]))
            border = np.concatenate((self.end[i_larger],
                                     self.begin[i_smaller]))
            return np.linalg.norm(proj - border, ord=exponent)

    def collapse(self, indices, values):
        """Partly collapse the interval product to single values.

        Note that no changes are made in-place.

        Parameters
        ----------
        indices : int or tuple of int
            The indices of the dimensions along which to collapse.
        values : array-like or float
            The values to which to collapse. Must have the same
            length as ``indices``. Values must lie within the interval
            boundaries.

        Returns
        -------
        collapsed : `IntervalProd`
            The collapsed set.

        Examples
        --------
        >>> b, e = [-1, 0, 2], [-0.5, 1, 3]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.collapse(1, 0)
        Cuboid([-1.0, 0.0, 2.0], [-0.5, 0.0, 3.0])
        >>> rbox.collapse([1, 2], [0, 2.5])
        Cuboid([-1.0, 0.0, 2.5], [-0.5, 0.0, 2.5])
        """
        indices = np.atleast_1d(indices).astype('int64', casting='safe')
        values = np.atleast_1d(values)
        if len(indices) != len(values):
            raise ValueError('lengths of indices {} and values {} do not '
                             'match ({} != {})'
                             ''.format(indices, values,
                                       len(indices), len(values)))

        for axis, index in enumerate(indices):
            if not 0 <= index <= self.ndim:
                raise IndexError('in axis {}: index {} out of range 0 --> {}'
                                 ''.format(axis, index, self.ndim - 1))

        if np.any(values < self.begin[indices]):
            raise ValueError('values {} not above the lower interval '
                             'boundaries {}'
                             ''.format(values, self.begin[indices]))

        if np.any(values > self.end[indices]):
            raise ValueError('values {} not below the upper interval '
                             'boundaries {}'
                             ''.format(values, self.end[indices]))

        b_new = self.begin.copy()
        b_new[indices] = values
        e_new = self.end.copy()
        e_new[indices] = values

        return IntervalProd(b_new, e_new)

    def squeeze(self):
        """Remove the degenerate dimensions.

        Note that no changes are made in-place.

        Returns
        -------
        squeezed : `IntervalProd`
            Squeezed set.

        Examples
        --------
        >>> b, e = [-1, 0, 2], [-0.5, 1, 3]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.collapse(1, 0).squeeze()
        Rectangle([-1.0, 2.0], [-0.5, 3.0])
        >>> rbox.collapse([1, 2], [0, 2.5]).squeeze()
        Interval(-1.0, -0.5)
        >>> rbox.collapse([0, 1, 2], [-1, 0, 2.5]).squeeze()
        IntervalProd([], [])
        """
        b_new = self.begin[self.inondeg]
        e_new = self.end[self.inondeg]
        return IntervalProd(b_new, e_new)

    def insert(self, index, other):
        """Insert ``other`` before ``index``.

        The given interval product (``ndim=m``) is inserted into the
        current one (``ndim=n``) before the given index, resulting in a
        new interval product with ``n+m`` dimensions.

        Note that no changes are made in-place.

        Parameters
        ----------
        index : int
            Index of the dimension before which ``other`` is to
            be inserted. Must fulfill ``-ndim <= index <= ndim``.
            Negative indices count backwards from ``self.ndim``.
        other : `IntervalProd`
            Interval product to be inserted.

        Returns
        -------
        newintvp : `IntervalProd`
            Interval product with ``other`` inserted.

        Examples
        --------
        >>> rbox = IntervalProd([-1, 2], [-0.5, 3])
        >>> rbox2 = IntervalProd([0, 0], [1, 0])
        >>> rbox.insert(1, rbox2)
        IntervalProd([-1.0, 0.0, 0.0, 2.0], [-0.5, 1.0, 0.0, 3.0])
        >>> rbox.insert(-1, rbox2)
        IntervalProd([-1.0, 0.0, 0.0, 2.0], [-0.5, 1.0, 0.0, 3.0])
        """
        index, index_in = int(index), index

        if not -self.ndim <= index <= self.ndim:
            raise IndexError('index {0} outside the valid range -{1} --> {1}'
                             ''.format(index_in, self.ndim))
        if index < 0:
            index += self.ndim

        new_beg = np.empty(self.ndim + other.ndim)
        new_end = np.empty(self.ndim + other.ndim)

        new_beg[: index] = self.begin[: index]
        new_end[: index] = self.end[: index]
        new_beg[index: index + other.ndim] = other.begin
        new_end[index: index + other.ndim] = other.end
        if index < self.ndim:  # Avoid IndexError
            new_beg[index + other.ndim:] = self.begin[index:]
            new_end[index + other.ndim:] = self.end[index:]

        return IntervalProd(new_beg, new_end)

    def append(self, other):
        """Insert at the end.

        Parameters
        ----------
        other : `IntervalProd`
            Set to be appended.

        Returns
        -------
        newintvp : `IntervalProd`
            Interval product with ``other`` appended.

        Examples
        --------
        >>> rbox = IntervalProd([-1, 2], [-0.5, 3])
        >>> rbox.append(Interval(-1.0, 0.0))
        Cuboid([-1.0, 2.0, -1.0], [-0.5, 3.0, 0.0])

        See Also
        --------
        insert
        """
        return self.insert(self.ndim, other)

    def corners(self, order='C'):
        """Return the corner points as a single array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Ordering of the axes in which the corners appear in
            the output. ``'C'`` means that the first axis varies slowest
            and the last one fastest, vice versa in ``'F'`` ordering.

        Returns
        -------
        corners : numpy.ndarray
            Array containing the corner coordinates. The size of the
            array is ``2^m x ndim``, where ``m`` is the number of
            non-degenerate axes, i.e. the corners are stored as rows.

        Examples
        --------
        >>> rbox = IntervalProd([-1, 2, 0], [-0.5, 3, 0.5])
        >>> rbox.corners()
        array([[-1. ,  2. ,  0. ],
               [-1. ,  2. ,  0.5],
               [-1. ,  3. ,  0. ],
               [-1. ,  3. ,  0.5],
               [-0.5,  2. ,  0. ],
               [-0.5,  2. ,  0.5],
               [-0.5,  3. ,  0. ],
               [-0.5,  3. ,  0.5]])
        >>> rbox.corners(order='F')
        array([[-1. ,  2. ,  0. ],
               [-0.5,  2. ,  0. ],
               [-1. ,  3. ,  0. ],
               [-0.5,  3. ,  0. ],
               [-1. ,  2. ,  0.5],
               [-0.5,  2. ,  0.5],
               [-1. ,  3. ,  0.5],
               [-0.5,  3. ,  0.5]])
        """
        from odl.discr.grid import TensorGrid

        minmax_vecs = [0] * self.ndim
        for axis in self.ideg:
            minmax_vecs[axis] = self.begin[axis]
        for axis in self.inondeg:
            minmax_vecs[axis] = (self.begin[axis], self.end[axis])

        minmax_grid = TensorGrid(*minmax_vecs)
        return minmax_grid.points(order=order)

    def __len__(self):
        """Return ``len(self)``."""
        return self.ndim

    def __getitem__(self, indices):
        """Return ``self[indices]``

        Parameters
        ----------
        indices : index expression
            Object determining which parts of the interval product
            to extract.

        Returns
        -------
        subinterval : `IntervalProd`
            Interval product corresponding to the indices.

        Examples
        --------
        >>> rbox = IntervalProd([-1, 2, 0], [-0.5, 3, 0.5])

        Indexing by integer selects single axes:

        >>> rbox[0]
        Interval(-1.0, -0.5)

        With slices, multiple axes can be selected:

        >>> rbox[:]
        Cuboid([-1.0, 2.0, 0.0], [-0.5, 3.0, 0.5])
        >>> rbox[::2]
        Rectangle([-1.0, 0.0], [-0.5, 0.5])

        A list of integers can be used for free combinations of axes:

        >>> rbox[[0, 1, 0]]
        Cuboid([-1.0, 2.0, -1.0], [-0.5, 3.0, -0.5])
        """
        return IntervalProd(self.begin[indices], self.end[indices])

    def __pos__(self):
        """Return ``+self``."""
        return self

    def __neg__(self):
        """Return ``-self``."""
        return type(self)(-self.end, -self.begin)

    def __add__(self, other):
        """Return ``self + other``."""
        if isinstance(other, IntervalProd):
            if self.ndim != other.ndim:
                raise ValueError('addition not possible for {} and {}: '
                                 'dimension mismatch ({} != {})'
                                 ''.format(self, other, self.ndim, other.ndim))
            return type(self)(self.begin + other.begin, self.end + other.end)
        elif np.isscalar(other):
            return type(self)(self.begin + other, self.end + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """Return ``self - other``."""
        return self + (-other)

    def __mul__(self, other):
        """Return ``self * other``."""
        if isinstance(other, IntervalProd):
            if self.ndim != other.ndim:
                raise ValueError('multiplication not possible for {} and'
                                 '{}: dimension mismatch ({} != {})'
                                 ''.format(self, other, self.ndim, other.ndim))

            comp_mat = np.empty([self.ndim, 4])
            comp_mat[:, 0] = self.begin * other.begin
            comp_mat[:, 1] = self.begin * other.end
            comp_mat[:, 2] = self.end * other.begin
            comp_mat[:, 3] = self.end * other.end
            new_beg = np.min(comp_mat, axis=1)
            new_end = np.max(comp_mat, axis=1)
            return type(self)(new_beg, new_end)
        elif np.isscalar(other):
            vec1 = self.begin * other
            vec2 = self.end * other
            return type(self)(np.minimum(vec1, vec2), np.maximum(vec1, vec2))
        else:
            return NotImplemented

    def __div__(self, other):
        """Return ``self / other``."""
        return self * (1.0 / other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        """Return ``other / self``."""
        if np.isscalar(other):
            for axis, (b, e) in enumerate(zip(self.begin, self.end)):
                if b <= 0 and e >= 0:
                    raise ValueError('division not possible: interval product'
                                     'contains 0 in axis {}'.format(axis))

            vec1 = other / self.begin
            vec2 = other / self.end
            return type(self)(np.minimum(vec1, vec2), np.maximum(vec1, vec2))
        else:
            return NotImplemented

    __rtruediv__ = __rdiv__

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            return 'Interval({!r}, {!r})'.format(self.begin[0], self.end[0])
        elif self.ndim == 2:
            return 'Rectangle({!r}, {!r})'.format(list(self.begin),
                                                  list(self.end))
        elif self.ndim == 3:
            return 'Cuboid({!r}, {!r})'.format(list(self.begin),
                                               list(self.end))
        else:
            return '{}({}, {})'.format(self.__class__.__name__,
                                       array1d_repr(self.begin),
                                       array1d_repr(self.end))

    def __str__(self):
        """Return ``str(self)``."""
        return ' x '.join('[{}, {}]'.format(b, e)
                          for (b, e) in zip(self.begin, self.end))


def Interval(begin, end):
    """One-dimensional interval product.

    Parameters
    ----------
    begin : array-like with shape ``(1,)`` or float
        Lower end of the interval.
    end : array-like with shape ``(1,)`` or float
        Upper end of the interval.

    """
    interval = IntervalProd(begin, end)
    if interval.ndim != 1:
        raise ValueError('cannot make an interval from `begin` {} and '
                         '`end` {}'.format(begin, end))
    return interval


def Rectangle(begin, end):
    """Two-dimensional interval product.

    Parameters
    ----------
    begin : array-like with shape ``(2,)``
        Lower ends of the intervals in the product.
    end : array-like with shape ``(2,)``
        Upper ends of the intervals in the product.
    """
    rectangle = IntervalProd(begin, end)
    if rectangle.ndim != 2:
        raise ValueError('cannot make a rectangle from `begin` {} and '
                         '`end` {}'.format(begin, end))
    return rectangle


def Cuboid(begin, end):
    """Three-dimensional interval product.

    Parameters
    ----------
    begin : array-like with shape ``(3,)``
        Lower ends of the intervals in the product.
    end : array-like with shape ``(3,)``
        Upper ends of the intervals in the product.
    """
    cuboid = IntervalProd(begin, end)
    if cuboid.ndim != 3:
        raise ValueError('cannot make a cuboid from `begin` {} and '
                         '`end` {}'.format(begin, end))
    return cuboid


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
