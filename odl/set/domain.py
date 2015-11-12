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

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import super, zip
from future import standard_library
standard_library.install_aliases()

# External imports
import numpy as np

# ODL imports
from odl.set.sets import Set, RealNumbers
from odl.util.utility import array1d_repr


__all__ = ('IntervalProd', 'Interval', 'Rectangle', 'Cuboid')


class IntervalProd(Set):

    """An n-dimensional rectangular box.

    An interval product is a Cartesian product of n intervals, i.e. an
    n-dimensional rectangular box aligned with the coordinate axes
    as a subset of :math:`R^n`.
    """

    def __init__(self, begin, end):
        """Initialize a new instance.

        Parameters
        ----------
        begin : array-like or float
            The lower ends of the intervals in the product
        end : array-like or float
            The upper ends of the intervals in the product

        Examples
        --------
        >>> b, e = [-1, 2.5, 70], [-0.5, 10, 75]
        >>> rbox = IntervalProd(b, e)
        >>> rbox
        IntervalProd([-1.0, 2.5, 70.0], [-0.5, 10.0, 75.0])
        """
        self._begin = np.atleast_1d(begin).astype(np.float64)
        self._end = np.atleast_1d(end).astype(np.float64)

        if self._begin.ndim > 1:
            raise ValueError('begin {} is {}- instead of 1-dimensional.'
                             ''.format(begin, self._begin.ndim))
        if self._end.ndim > 1:
            raise ValueError('end {} is {}- instead of 1-dimensional.'
                             ''.format(end, self._end.ndim))
        if len(self._begin) != len(self._end):
            raise ValueError('begin {} and end {} have different '
                             'lengths ({} != {}).'
                             ''.format(begin, end,
                                       len(self._begin), len(self._end)))
        if not np.all(self._begin <= self._end):
            i_wrong = np.where(self._begin > self._end)
            raise ValueError('entries at indices {} of begin exceed '
                             'those of end ({} > {}).'
                             ''.format(i_wrong, list(self._begin[i_wrong]),
                                       list(self._end[i_wrong])))

        self._ideg = np.where(self._begin == self._end)[0]
        self._inondeg = np.where(self._begin != self._end)[0]
        super().__init__()

    # Basic properties
    @property
    def begin(self):
        """The left interval boundary/boundaries."""
        return self._begin

    @property
    def end(self):
        """The right interval boundary/boundaries."""
        return self._end

    @property
    def ndim(self):
        """The number of intervals in the product."""
        return len(self.begin)

    @property
    def true_ndim(self):
        """The number of non-degenerate (zero-length) intervals."""
        return len(self._inondeg)

    @property
    def size(self):
        """The interval length per axis."""
        return self.end - self.begin

    @property
    def volume(self):
        """The 'dim'-dimensional volume of this interval product."""
        return self.measure(ndim=self.ndim)

    @property
    def midpoint(self):
        """The midpoint of the interval product."""
        midp = (self.end + self.begin) / 2.
        midp[self._ideg] = self.begin[self._ideg]
        return midp

    def min(self):
        """The minimum value in this interval product"""
        return self.begin

    def max(self):
        """The maximum value in this interval product"""
        return self.end

    def element(self):
        """An arbitrary element, the midpoint."""
        return self.midpoint

    # Overrides of the abstract base class methods
    def approx_equals(self, other, tol):
        """Test if ``other`` is equal to this set up to `tol`.

        Parameters
        ----------
        other : object
            The object to be tested
        tol : float
            The maximum allowed difference in 'inf'-norm between the
            interval endpoints.

        Examples
        --------
        >>> from math import sqrt
        >>> rbox1 = IntervalProd(0, 0.5)
        >>> rbox2 = IntervalProd(0, sqrt(0.5)**2)
        >>> rbox1.approx_equals(rbox2, tol=0)  # Num error
        False
        >>> rbox1.approx_equals(rbox2, tol=1e-15)
        True
        """
        # pylint: disable=arguments-differ
        if other is self:
            return True
        elif not isinstance(other, IntervalProd):
            return False

        return (np.allclose(self.begin, other.begin, atol=tol, rtol=0.0) and
                np.allclose(self.end, other.end, atol=tol, rtol=0.0))

    def __eq__(self, other):
        """`g.__eq__(other) <==> g == other`."""
        return self.approx_equals(other, tol=0.0)

    def approx_contains(self, point, tol):
        """Test if a point is contained.

        Parameters
        ----------
        point : array-like or float
            The point to be tested. Its length must be equal
            to the set's dimension. In the 1d case, 'point'
            can be given as a float.
        tol : float
            The maximum allowed distance in 'inf'-norm between the
            point and the set.
            Default: 0.0

        Examples
        --------

        >>> from math import sqrt
        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervalProd(b, e)
        >>> # Numerical error
        >>> rbox.approx_contains([-1 + sqrt(0.5)**2, 0., 2.9], tol=0)
        False
        >>> rbox.approx_contains([-1 + sqrt(0.5)**2, 0., 2.9], tol=1e-9)
        True
        """
        point = np.atleast_1d(point)
        if np.any(np.isnan(point)):
            return False
        if point.ndim > 1:
            return False
        if len(point) != self.ndim:
            return False
        if point[0] not in RealNumbers():
            return False
        if self.dist(point, ord=np.inf) > tol:
            return False
        return True

    def __contains__(self, other):
        """`g.__contains__(other) <==> other in g`."""
        return self.approx_contains(other, tol=0)

    def contains_set(self, other, tol=0.0):
        """Test if another set is contained.

        Parameters
        ----------
        other : :class:`Set`
            The set to be tested. It must implement a `min()` and a
            `max()` method, otherwise a `TypeError` is raised.
        tol : float, optional
            The maximum allowed distance in 'inf'-norm between the
            other set and this interval product.
            Default: 0.0
        """
        try:
            return (self.approx_contains(other.min(), tol) and
                    self.approx_contains(other.max(), tol))
        except AttributeError:
            raise AttributeError('cannot test {!r} without `min()` and `max()`'
                                 'methods.'.format(other))

    # Additional property-like methods
    def measure(self, ndim=None):
        """The (Lebesgue) measure of this interval product.

        Parameters
        ----------
        ndim : int, optional
              The dimension of the measure to apply.
              Default: `true_ndim`

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
            return np.prod((self._end - self._begin)[self._inondeg])

    def dist(self, point, ord=2.0):
        """Calculate the distance to a point.

        Parameters
        ----------
        point : array-like or float
                The point. Its length must be equal to the set's
                dimension. Can be a `float` in the 1d case.
        ord : non-zero int or float('inf'), optional
              The order of the norm (see numpy.linalg.norm).
              Default: 2.0

        Examples
        --------

        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.dist([-5, 3, 2])
        5.0
        >>> rbox.dist([-5, 3, 2], ord=float('inf'))
        4.0
        """
        point = np.atleast_1d(point)
        if len(point) != self.ndim:
            raise ValueError('length {} of point {} does not match '
                             'the dimension {} of the set {}.'
                             ''.format(len(point), point, self.ndim))

        i_larger = np.where(point > self._end)
        i_smaller = np.where(point < self._begin)

        # Access [0] since np.where returns tuple.
        if len(i_larger[0]) == 0 and len(i_smaller[0]) == 0:
            return 0.0
        else:
            proj = np.concatenate((point[i_larger], point[i_smaller]))
            border = np.concatenate((self._end[i_larger],
                                     self._begin[i_smaller]))
            return np.linalg.norm(proj - border, ord=ord)

    # Manipulation
    def collapse(self, indices, values):
        """Partly collapse the interval product to single values.

        Note that no changes are made in-place.

        Parameters
        ----------
        indices : int or tuple of ints
            The indices of the dimensions along which to collapse
        values : `float` or array-like
            The values to which to collapse. Must have the same
            lenght as 'indcs'. Values must lie within the interval
            boundaries.

        Returns
        -------
        collapsed : :class:`IntervalProd`
            The collapsed set

        Examples
        --------

        >>> b, e = [-1, 0, 2], [-0.5, 1, 3]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.collapse(1, 0)
        IntervalProd([-1.0, 0.0, 2.0], [-0.5, 0.0, 3.0])
        >>> rbox.collapse([1, 2], [0, 2.5])
        IntervalProd([-1.0, 0.0, 2.5], [-0.5, 0.0, 2.5])
        >>> rbox.collapse([1, 2], [0, 3.5])
        Traceback (most recent call last):
            ...
        ValueError: values [ 0.   3.5] not below the upper interval
        boundaries [ 1.  3.].
        """
        indices = np.atleast_1d(indices)
        values = np.atleast_1d(values)
        if len(indices) != len(values):
            raise ValueError('lengths of indices {} and values {} do not '
                             'match ({} != {}).'
                             ''.format(indices, values,
                                       len(indices), len(values)))

        if np.any(indices < 0) or np.any(indices >= self.ndim):
            raise IndexError('indices {} out of range 0 --> {}.'
                             ''.format(list(indices), self.ndim))

        if np.any(values < self.begin[indices]):
            raise ValueError('values {} not above the lower interval '
                             'boundaries {}.'
                             ''.format(values, self.begin[indices]))

        if np.any(values > self.end[indices]):
            raise ValueError('values {} not below the upper interval '
                             'boundaries {}.'
                             ''.format(values, self.end[indices]))

        b_new = self._begin.copy()
        b_new[indices] = values
        e_new = self._end.copy()
        e_new[indices] = values

        return IntervalProd(b_new, e_new)

    def squeeze(self):
        """Remove the degenerate dimensions.

        Note that no changes are made in-place.

        Returns
        -------
        squeezed : :class:`IntervalProd`
            The squeezed set

        Examples
        --------
        >>> b, e = [-1, 0, 2], [-0.5, 1, 3]
        >>> rbox = IntervalProd(b, e)
        >>> rbox.collapse(1, 0).squeeze()
        IntervalProd([-1.0, 2.0], [-0.5, 3.0])
        >>> rbox.collapse([1, 2], [0, 2.5]).squeeze()
        IntervalProd([-1.0], [-0.5])
        >>> rbox.collapse([0, 1, 2], [-1, 0, 2.5]).squeeze()
        IntervalProd([], [])
        """
        b_new = self._begin[self._inondeg]
        e_new = self._end[self._inondeg]
        return IntervalProd(b_new, e_new)

    def insert(self, other, index=None):
        """Insert another interval product before the given index.

        The given interval product (`ndim=m`) is inserted into the
        current one (`ndim=n`) before the given index, resulting in a
        new interval product with `n+m` dimensions.

        No changes are made in-place.

        Parameters
        ----------
        other : :class:`IntervalProd`, `float` or array-like
            The set to be inserted. A `float` or array a is
            treated as an `IntervalProd(a, a)`.
        index : int, Optional (default = `ndim`)
            The index of the dimension before which ``other`` is to
            be inserted. Must fulfill `0 <= index <= ndim`.

        Returns
        -------
        larger_set : :class:`IntervalProd`
            The enlarged set

        Examples
        --------

        >>> rbox = IntervalProd([-1, 2], [-0.5, 3])
        >>> rbox2 = IntervalProd([0, 0], [1, 0])
        >>> rbox.insert(rbox2, 1)
        IntervalProd([-1.0, 0.0, 0.0, 2.0], [-0.5, 1.0, 0.0, 3.0])
        >>> rbox.insert([-1.0, 0.0], 2)
        IntervalProd([-1.0, 2.0, -1.0, 0.0], [-0.5, 3.0, -1.0, 0.0])

        Without index, inserts at the end

        >>> rbox.insert([-1.0, 0.0])
        IntervalProd([-1.0, 2.0, -1.0, 0.0], [-0.5, 3.0, -1.0, 0.0])

        Can also insert by array

        >>> rbox.insert(0, 1).squeeze() == rbox
        True
        """
        if index is None:
            index = self.ndim
        elif not 0 <= index <= self.ndim:
            raise IndexError('Index ({}) out of range'.format(index))

        #TODO: do we want this?
        if not isinstance(other, IntervalProd):
            other = IntervalProd(other, other)

        new_beg = np.empty(self.ndim + other.ndim)
        new_end = np.empty(self.ndim + other.ndim)

        new_beg[: index] = self._begin[: index]
        new_end[: index] = self._end[: index]
        new_beg[index: index+other.ndim] = other.begin
        new_end[index: index+other.ndim] = other.end
        if index < self.ndim:  # Avoid IndexError
            new_beg[index+other.ndim:] = self._begin[index:]
            new_end[index+other.ndim:] = self._end[index:]

        return IntervalProd(new_beg, new_end)

    def corners(self, order='C'):
        """The corner points in a single array.

        Parameters
        ----------
        order : {'C', 'F'}
            The ordering of the axes in which the corners appear in
            the output. 'C' means that the first axis varies slowest
            and the last one fastest, vice versa in 'F' ordering.

        Returns
        -------
        out : numpy.ndarray
            The size of the array is `2^m * ndim`, where `m` is the number
            of non-degenerate axes, i.e. the corners are stored as rows.

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
        if order not in ('C', 'F'):
            raise ValueError('order {} not understood.'.format(order))

        minmax_vecs = [0] * self.ndim
        for axis in self._ideg:
            minmax_vecs[axis] = self._begin[axis]
        for axis in self._inondeg:
            minmax_vecs[axis] = (self._begin[axis], self._end[axis])

        minmax_grid = TensorGrid(*minmax_vecs)
        return minmax_grid.points(order=order)

    # Magic methods
    def __len__(self):
        """`ip.__len__() <==> len(ip)`."""
        return self.ndim

    def __pos__(self):
        """`ip.__pos__(other) <==> +ip`."""
        return self

    def __neg__(self):
        """`ip.__pos__(other) <==> +ip`."""
        return type(self)(-self.end, -self.begin)

    def __add__(self, other):
        """`ip.__add__(other) <==> ip + other`."""
        if isinstance(other, IntervalProd):
            if self.ndim != other.ndim:
                raise ValueError('Addition not possible for {} and {}: '
                                 'dimension mismatch ({} != {}).'
                                 ''.format(self, other, self.ndim, other.ndim))
            return type(self)(self.begin + other.begin, self.end + other.end)
        elif np.isscalar(other):
            return type(self)(self.begin + other, self.end + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        """`ip.__sub__(other) <==> ip - other`."""
        return self + (-other)

    def __mul__(self, other):
        """`ip.__mul__(other) <==> ip * other`."""
        if isinstance(other, IntervalProd):
            if self.ndim != other.ndim:
                raise ValueError('Multiplication not possible for {!r} and {!r}: '
                                 'dimension mismatch ({} != {}).'
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
        """`ip.__mul__(other) <==> ip / other`."""
        return self * (1.0 / other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        """`ip.__rdiv__(other) <==> other / ip`."""
        if np.isscalar(other):
            contains_zero = np.any(np.logical_and(self.begin <= 0, self.end>= 0))
            if contains_zero:
                raise ValueError('Division by other {!r} not possible:'
                                 'Interval contains 0.'
                                 ''.format(other))

            vec1 = other / self.begin
            vec2 = other / self.end
            return type(self)(np.minimum(vec1, vec2), np.maximum(vec1, vec2))
        else:
            return NotImplemented
        
    __rtruediv__ = __rdiv__

    def __repr__(self):
        """`ip.__repr__() <==> repr(ip)`."""
        return ('IntervalProd({}, {})'.format(
            array1d_repr(self.begin), array1d_repr(self._end)))

    def __str__(self):
        """`ip.__str__() <==> str(ip)`."""
        return ' x '.join('[{}, {}]'.format(b, e)
                          for (b, e) in zip(self.begin, self.end))


class Interval(IntervalProd):
    """One-dimensional interval product, i.e. just one interval."""
    def __init__(self, begin, end):
        super().__init__(begin, end)
        if self.ndim != 1:
            raise ValueError('cannot make an interval from begin {} and '
                             'end {}.'.format(begin, end))

    @property
    def length(self):
        """The length of this interval."""
        return self.end[0] - self.begin[0]

    def __repr__(self):
        return 'Interval({}, {})'.format(self.begin[0], self.end[0])


class Rectangle(IntervalProd):
    """Two-dimensional interval product."""
    def __init__(self, begin, end):
        super().__init__(begin, end)
        if self.ndim != 2:
            raise ValueError('cannot make a rectangle from begin {} and '
                             'end {}.'.format(begin, end))

    @property
    def area(self):
        """Area measure of this rectangle."""
        return self.volume

    def __repr__(self):
        return ('Rectangle({!r}, {!r})'.format(list(self.begin),
                                               list(self.end)))


class Cuboid(IntervalProd):
    """Three-dimensional interval product."""
    def __init__(self, begin, end):
        super().__init__(begin, end)
        if self.ndim != 3:
            raise ValueError('cannot make a cuboid from begin {} and '
                             'end {}.'.format(begin, end))

    def __repr__(self):
        return ('Cuboid({!r}, {!r})'.format(list(self.begin),
                                            list(self.end)))

if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
