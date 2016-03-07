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

# Imports for common Python 2/3 codebase

"""Typical domains for inverse problems. """

from __future__ import print_function, division, absolute_import

from builtins import super, zip
from future import standard_library
from future.utils import raise_from
standard_library.install_aliases()

# External imports
import numpy as np

# ODL imports
from odl.set.sets import Set
from odl.util.utility import array1d_repr, is_real_dtype
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid)


__all__ = ('IntervalProd', 'Interval', 'Rectangle', 'Cuboid')


class IntervalProd(Set):

    """An n-dimensional rectangular box.

    An interval product is a Cartesian product of n intervals, i.e. an
    n-dimensional rectangular box aligned with the coordinate axes
    as a subset of :math:`R^n`.

    `IntervalProd` objects are immutable, all methods involving them return
    a new `IntervalProd`.
    """

    def __init__(self, begin, end):
        """Initialize a new instance.

        Parameters
        ----------
        begin : `array-like` or `float`
            The lower ends of the intervals in the product
        end : `array-like` or `float`
            The upper ends of the intervals in the product

        Examples
        --------
        >>> b, e = [-1, 2.5, 70, 80], [-0.5, 10, 75, 90]
        >>> rbox = IntervalProd(b, e)
        >>> rbox
        IntervalProd([-1.0, 2.5, 70.0, 80.0], [-0.5, 10.0, 75.0, 90.0])
        """
        self._begin = np.atleast_1d(begin).astype('float64')
        self._end = np.atleast_1d(end).astype('float64')

        if self.begin.ndim > 1:
            raise ValueError('begin {} is {}- instead of 1-dimensional.'
                             ''.format(begin, self.begin.ndim))
        if self.end.ndim > 1:
            raise ValueError('end {} is {}- instead of 1-dimensional.'
                             ''.format(end, self.end.ndim))
        if len(self.begin) != len(self.end):
            raise ValueError('begin {} and end {} have different '
                             'lengths ({} != {}).'
                             ''.format(begin, end,
                                       len(self.begin), len(self.end)))
        if not np.all(self.begin <= self.end):
            i_wrong = np.where(self.begin > self.end)
            raise ValueError('entries at indices {} of begin exceed '
                             'those of end ({} > {}).'
                             ''.format(i_wrong, list(self.begin[i_wrong]),
                                       list(self.end[i_wrong])))

        self._ideg = np.where(self.begin == self.end)[0]
        self._inondeg = np.where(self.begin != self.end)[0]
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
    def volume(self):
        """The 'dim'-dimensional volume of this interval product."""
        return self.measure(ndim=self.ndim)

    @property
    def length(self):
        """The length of this interval."""
        if self.ndim != 1:
            raise NotImplementedError('length not defined if ndim != 1.')
        return self.volume

    @property
    def area(self):
        """The length of this interval."""
        if self.ndim != 2:
            raise NotImplementedError('area not defined if ndim != 2.')
        return self.volume

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

    def extent(self):
        """The interval length per axis."""
        return self.max() - self.min()

    def element(self, inp=None):
        """Create element in this set.

        Parameters
        ----------
        inp : `float` or array-like, optional
            Point to be cast to an element in self

        Returns
        -------
        element
            Returns ``inp`` if given, else ``self.midpoint``

        Raises
        ------
        TypeError
            If ``inp`` is not a valid element.

        Examples
        --------
        >>> interv = IntervalProd(0, 1)
        >>> interv.element(0.5)
        0.5
        """
        if inp is None:
            return self.midpoint
        elif inp in self:
            return inp
        else:
            raise TypeError('inp {!r} not a valid element in {!r}.'
                            ''.format(inp, self))

    def approx_equals(self, other, tol):
        """Test if ``other`` is equal to this set up to ``tol``.

        Parameters
        ----------
        other : `object`
            The object to be tested
        tol : `float`
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
        if other is self:
            return True
        elif not isinstance(other, IntervalProd):
            return False

        return (np.allclose(self.begin, other.begin, atol=tol, rtol=0.0) and
                np.allclose(self.end, other.end, atol=tol, rtol=0.0))

    def __eq__(self, other):
        """Return ``self == other``."""
        return self.approx_equals(other, tol=0.0)

    def approx_contains(self, point, tol):
        """Test if a point is contained.

        Parameters
        ----------
        point : `array-like` or `float`
            The point to be tested. Its length must be equal
            to the set's dimension. In the 1d case, 'point'
            can be given as a `float`.
        tol : `float`
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
        if point.shape != (self.ndim,):
            return False
        if not is_real_dtype(point.dtype):
            return False
        return self.dist(point, ord=np.inf) <= tol

    def __contains__(self, other):
        """Return ``other in self``.

        Parameters
        ----------
        other
            Arbitrary object to be tested.

        Returns
        -------
        containts : `bool`
            True if other is inside self.

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

    def contains_set(self, other, tol=0.0):
        """Test if another set is contained.

        Parameters
        ----------
        other : `Set`
            The set to be tested. It must implement a ``min()`` and a
            ``max()`` method, otherwise a `TypeError` is raised.
        tol : `float`, optional
            The maximum allowed distance in 'inf'-norm between the
            other set and this interval product.
            Default: 0.0

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
        try:
            return (self.approx_contains(other.min(), tol) and
                    self.approx_contains(other.max(), tol))
        except AttributeError as err:
            raise_from(
                AttributeError('cannot test {!r} without `min()` and `max()`'
                               'methods.'.format(other)), err)

    def contains_all(self, other):
        """Test if all points defined by ``other`` are contained.

        Parameters
        ----------
        other :
            Can be a single point, a ``(d, N)`` array where ``d`` is the
            number of dimensions or a length-``d`` meshgrid sequence

        Returns
        -------
        contains : `bool`
            `True` if all points are contained, `False` otherwise

        Examples
        --------
        >>> import odl
        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervalProd(b, e)

        rrays are expected in (ndim, npoints) shape

        >>> arr = np.array([[-1, 0, 2],   # defining one point at a time
        ...                 [-0.5, 0, 2]])
        >>> rbox.contains_all(arr.T)
        True

        Implicit meshgrids defined by coordinate vectors

        >>> from odl.discr.grid import sparse_meshgrid
        >>> vec1 = (-1, -0.9, -0.7)
        >>> vec2 = (0, 0, 0)
        >>> vec3 = (2.5, 2.75, 3)
        >>> mg = sparse_meshgrid(vec1, vec2, vec3)
        >>> rbox.contains_all(mg)
        True

        Also works with any iterable

        >>> rbox.contains_all([[-1, -0.5], # define points by axis
        ...                    [0, 0],
        ...                    [2, 2]])
        True

        And with grids

        >>> agrid = odl.uniform_sampling(rbox.begin, rbox.end, [3, 1, 3])
        >>> rbox.contains_all(agrid)
        True
        """
        # First try optimized methods
        if other in self:
            return True
        if hasattr(other, 'meshgrid'):
            return self.contains_all(other.meshgrid)
        elif is_valid_input_meshgrid(other, self.ndim):
            vecs = tuple(vec.squeeze() for vec in other)
            mins = np.fromiter((np.min(vec) for vec in vecs), dtype=float)
            maxs = np.fromiter((np.max(vec) for vec in vecs), dtype=float)
            return np.all(mins >= self.begin) and np.all(maxs <= self.end)

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

    # Additional property-like methods
    def measure(self, ndim=None):
        """The (Lebesgue) measure of this interval product.

        Parameters
        ----------
        ndim : `int`, optional
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
            return np.prod((self.end - self.begin)[self._inondeg])

    def dist(self, point, ord=2.0):
        """Calculate the distance to a point.

        Parameters
        ----------
        point : `array-like` or `float`
                The point. Its length must be equal to the set's
                dimension. Can be a `float` in the 1d case.
        ord : non-zero int or float('inf'), optional
              The order of the norm (see `numpy.linalg.norm`).
              Default: 2.0

        Returns
        -------
        dist : `float`
            Distance to the interior of the IntervalProd.
            Points strictly inside have distance ``0.0``, points with ``NaN``
            have distance ``infinity``.

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
                             ''.format(len(point), point, self.ndim, self))

        if np.any(np.isnan(point)):
            return np.inf

        i_larger = np.where(point > self.end)
        i_smaller = np.where(point < self.begin)

        # Access [0] since np.where returns tuple.
        if len(i_larger[0]) == 0 and len(i_smaller[0]) == 0:
            return 0.0
        else:
            proj = np.concatenate((point[i_larger], point[i_smaller]))
            border = np.concatenate((self.end[i_larger],
                                     self.begin[i_smaller]))
            return np.linalg.norm(proj - border, ord=ord)

    # Manipulation
    def collapse(self, indices, values):
        """Partly collapse the interval product to single values.

        Note that no changes are made in-place.

        Parameters
        ----------
        indices : `int` or `tuple` of `int`
            The indices of the dimensions along which to collapse
        values : `array-like` or `float`
            The values to which to collapse. Must have the same
            length as ``indices``. Values must lie within the interval
            boundaries.

        Returns
        -------
        collapsed : `IntervalProd`
            The collapsed set

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
            The squeezed set

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
        b_new = self.begin[self._inondeg]
        e_new = self.end[self._inondeg]
        return IntervalProd(b_new, e_new)

    def insert(self, index, other):
        """Return a copy with ``other`` inserted before ``index``.

        The given interval product (``ndim=m``) is inserted into the
        current one (``ndim=n``) before the given index, resulting in a
        new interval product with ``n+m`` dimensions.

        Parameters
        ----------
        index : `int`
            Index of the dimension before which ``other`` is to
            be inserted. Must fulfill ``-ndim <= index <= ndim``.
            Negative indices count backwards from ``self.ndim``.
        other : `IntervalProd`
            Interval product to be inserted

        Returns
        -------
        newintvp : `IntervalProd`
            Interval product with ``other`` inserted

        Examples
        --------

        >>> rbox = IntervalProd([-1, 2], [-0.5, 3])
        >>> rbox2 = IntervalProd([0, 0], [1, 0])
        >>> rbox.insert(1, rbox2)
        IntervalProd([-1.0, 0.0, 0.0, 2.0], [-0.5, 1.0, 0.0, 3.0])
        >>> rbox.insert(-1, rbox2)
        IntervalProd([-1.0, 0.0, 0.0, 2.0], [-0.5, 1.0, 0.0, 3.0])
        """
        if index < 0:
            index = int(index) + self.ndim
        else:
            index = int(index)

        if not 0 <= index <= self.ndim:
            raise IndexError('index {} outside the valid range 0 ... {}.'
                             ''.format(index, self.ndim))

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
        other : `IntervalProd`, `float` or array-like
            The set to be inserted. A `float` or array a is
            treated as an ``IntervalProd(a, a)``.

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
        """The corner points in a single array.

        Parameters
        ----------
        order : {'C', 'F'}
            The ordering of the axes in which the corners appear in
            the output. 'C' means that the first axis varies slowest
            and the last one fastest, vice versa in 'F' ordering.

        Returns
        -------
        corners : `numpy.ndarray`
            The size of the array is ``2^m * ndim``, where ``m``
            is the number of non-degenerate axes, i.e. the corners are
            stored as rows.

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
        for axis in self._ideg:
            minmax_vecs[axis] = self.begin[axis]
        for axis in self._inondeg:
            minmax_vecs[axis] = (self.begin[axis], self.end[axis])

        minmax_grid = TensorGrid(*minmax_vecs)
        return minmax_grid.points(order=order)

    # Magic methods
    def __len__(self):
        """Return ``len(self)``."""
        return self.ndim

    def __getitem__(self, indices):
        """Return ``self[indices]``

        Parameters
        ----------
        indices : numpy style index
            Any of: int, slice, list of ints

        Returns
        -------
        subinterval : `IntervalProd`
            Interval given by the indices

        Examples
        --------
        >>> rbox = IntervalProd([-1, 2, 0], [-0.5, 3, 0.5])

        By integer

        >>> rbox[0]
        Interval(-1.0, -0.5)

        By slice

        >>> rbox[:]
        Cuboid([-1.0, 2.0, 0.0], [-0.5, 3.0, 0.5])
        >>> rbox[::2]
        Rectangle([-1.0, 0.0], [-0.5, 0.5])

        By list of ints

        >>> rbox[[0, 1]]
        Rectangle([-1.0, 2.0], [-0.5, 3.0])
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
                raise ValueError('Addition not possible for {} and {}: '
                                 'dimension mismatch ({} != {}).'
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
                raise ValueError('Multiplication not possible for {!r} and'
                                 '{!r}: dimension mismatch ({} != {}).'
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
            contains_zero = np.any(np.logical_and(self.begin <= 0,
                                                  self.end >= 0))
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
            return 'IntervalProd({}, {})'.format(array1d_repr(self.begin),
                                                 array1d_repr(self.end))

    def __str__(self):
        """Return ``str(self)``."""
        return ' x '.join('[{}, {}]'.format(b, e)
                          for (b, e) in zip(self.begin, self.end))


def Interval(begin, end):
    """One-dimensional interval product.

    Parameters
    ----------
    begin : `array-like`, shape ``(1,)``, or `float`
        The lower ends of the intervals in the product
    end : `array-like`, shape ``(1,)``, or `float`
        The upper ends of the intervals in the product

    """
    interval = IntervalProd(begin, end)
    if interval.ndim != 1:
        raise ValueError('cannot make an interval from begin {} and '
                         'end {}.'.format(begin, end))
    return interval


def Rectangle(begin, end):
    """Two-dimensional interval product.

    Parameters
    ----------
    begin : `array-like`, shape ``(2,)``
        The lower ends of the intervals in the product
    end : `array-like`, shape ``(2,)``
        The upper ends of the intervals in the product
    """
    rectangle = IntervalProd(begin, end)
    if rectangle.ndim != 2:
        raise ValueError('cannot make a rectangle from begin {} and '
                         'end {}.'.format(begin, end))
    return rectangle


def Cuboid(begin, end):
    """Three-dimensional interval product.

    Parameters
    ----------
    begin : `array-like`, shape ``(3,)``
        The lower ends of the intervals in the product
    end : `array-like`, shape ``(3,)``
        The upper ends of the intervals in the product
    """
    cuboid = IntervalProd(begin, end)
    if cuboid.ndim != 3:
        raise ValueError('cannot make a cuboid from begin {} and '
                         'end {}.'.format(begin, end))
    return cuboid


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
