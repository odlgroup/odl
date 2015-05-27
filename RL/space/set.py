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


# Imports for common Python 2/3 codebase
from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
try:
    from builtins import object, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import object, super
from future import standard_library

# External module imports
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real, Complex
import numpy as np
from numpy import float64

# RL imports
from RL.utility.utility import errfmt

standard_library.install_aliases()


class Set(object):
    """ An arbitrary set
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def equals(self, other):
        """ Test two sets for equality
        """

    @abstractmethod
    def contains(self, other):
        """ Test if other is a member of self
        """

    def element(self):
        """ Return some (arbitrary) element
        """
        raise NotImplementedError("'element' method not implemented")

    def __eq__(self, other):
        return self.equals(other)

    def __ne__(self, other):
        return not self.equals(other)

    def __contains__(self, other):
        return self.contains(other)


class EmptySet(Set):
    """ The empty set has no members (None is considered "no element")
    """
    def equals(self, other):
        return isinstance(other, EmptySet)

    def contains(self, other):
        return other is None

    def element(self):
        return None

    def __str__(self):
        return "EmptySet"

    def __repr__(self):
        return "EmptySet()"


class ComplexNumbers(Set):
    """ The set of complex numbers
    """

    def equals(self, other):
        return isinstance(other, ComplexNumbers)

    def contains(self, other):
        return isinstance(other, Complex)

    def element(self):
        return complex(0.0, 0.0)

    def __str__(self):
        return "ComplexNumbers"

    def __repr__(self):
        return "ComplexNumbers()"


class RealNumbers(Set):
    """ The set of real numbers
    """

    def equals(self, other):
        return isinstance(other, RealNumbers)

    def contains(self, other):
        return isinstance(other, Real)

    def element(self):
        return 0.0

    def __str__(self):
        return "RealNumbers"

    def __repr__(self):
        return "RealNumbers()"


class Integers(Set):
    """ The set of all integers
    """

    def equals(self, other):
        return isinstance(other, Integers)

    def contains(self, other):
        return isinstance(other, Integral)

    def element(self):
        return 0

    def __str__(self):
        return "Integers"

    def __repr__(self):
        return "Integers()"


class IntervProd(Set):
    """ An N-dimensional rectangular box

    An IntervProd is a Cartesian product of N intervals, i.e. an
    N-dimensional rectangular box aligned with the coordinate axes
    as a subset of R^N.
    """

    def __init__(self, begin, end):
        """
        Parameters
        ----------
        begin : array-like or float
                The lower ends of the intervals in the product
        end : array-like or float
              The upper ends of the intervals in the product

        Examples
        --------

        >>> b, e = [-1, 2.5, 70], [-0.5, 10, 75]
        >>> rbox = IntervProd(b, e)
        >>> rbox
        IntervProd([-1.0, 2.5, 70.0], [-0.5, 10.0, 75.0])
        """

        begin = np.atleast_1d(begin).astype(float64)
        end = np.atleast_1d(end).astype(float64)

        if len(begin) != len(end):
            raise ValueError(errfmt('''
            Lengths of 'begin' ({}) and 'end' ({}) do not match.
            '''.format(len(begin), len(end))))

        if not np.all(begin <= end):
            i_wrong = np.where(begin > end)
            raise ValueError(errfmt('''
            Entries of 'begin' exceed those of 'end' ({!r} > {!r})
            '''.format(list(begin[i_wrong]), list(end[i_wrong]))))

        self._begin = begin
        self._end = end
        self._ideg = np.where(self._begin == self._end)[0]
        self._inondeg = np.where(self._begin != self._end)[0]

    # Basic properties
    @property
    def begin(self):
        return self._begin[0] if self.dim == 1 else self._begin

    @property
    def end(self):
        return self._end[0] if self.dim == 1 else self._end

    @property
    def dim(self):
        return len(self._begin)

    @property
    def truedim(self):
        return len(self._inondeg)

    @property
    def volume(self):
        return self.measure(dim=self.dim)

    def midpoint(self):
        """ The midpoint of the interval product

        TODO: doc
        """
        midp = (self._end - self._begin) / 2.
        midp[self._ideg] = self._begin[self._ideg]
        return midp[0] if self.dim == 1 else midp

    def element(self):
        return self.midpoint()

    # Overrides of the abstract base class methods
    def equals(self, other, tol=0.0):
        """
        Test if another set is equal to the current one

        Parameters
        ----------
        other : object
                The object to be tested.
        tol : float, optional
              The maximum allowed absolute difference between the
              interval endpoints.
              Default: 0.0

        Examples
        --------

        >>> b1, e1 = [-1, 0, 2], [-0.5, 0, 3]
        >>> b2, e2 = [np.sin(-np.pi/2), 0, 2], [-0.5, 0, np.sqrt(3)**2]
        >>> rbox1 = IntervProd(b1, e1)
        >>> rbox2 = IntervProd(b2, e2)
        >>> rbox1.equals(rbox2)  # Num error
        False
        >>> rbox1 == rbox2  # Equivalent to rbox1.equals(rbox2)
        False
        >>> rbox1.equals(rbox2, tol=1e-15)
        True
        """

        if not isinstance(other, IntervProd):
            return False

        return (np.all(np.abs(self.begin - other.begin) <= tol) and
                np.all(np.abs(self.end - other.end) <= tol))

    def contains(self, point, tol=0.0):
        """
        Test if a point is contained

        Parameters
        ----------
        point : array-like or float
                The point to be tested. Its length must be equal
                to the set's dimension. In the 1d case, 'point'
                can be given as a float.
        tol : float, optional
              The maximum allowed distance (in 'inf'-norm) of a point
              to the set.
              Default: 0.0

        Examples
        --------

        >>> from math import sqrt
        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervProd(b, e)
        >>> rbox.contains([-1 + sqrt(0.5)**2, 0., 2.9])  # Num error
        False
        >>> rbox.contains([-1 + sqrt(0.5)**2, 0., 2.9], tol=1e-15)
        True
        """

        point = np.atleast_1d(point)

        if len(point) != self.dim:
            return False

        if not RealNumbers().contains(point[0]):
            return False
        if self.dist(point, ord=np.inf) > tol:
            return False
        return True

    # Additional property-like methods
    def measure(self, dim=None):
        """
        The (Lebesgue) measure of the IntervProd instance

        Parameters
        ----------
        dim : int, optional
              The dimension of the measure to apply.
              Default: truedim

        Examples
        --------

        >>> b, e = [-1, 2.5, 0], [-0.5, 10, 0]
        >>> rbox = IntervProd(b, e)
        >>> rbox.measure()
        3.75
        >>> rbox.measure(dim=3)
        0.0
        >>> rbox.measure(dim=3) == rbox.volume
        True
        >>> rbox.measure(dim=1)
        inf
        >>> rbox.measure() == rbox.squeeze().volume
        True
        """

        if self.truedim == 0:
            return 0.0

        if dim is None:
            return self.measure(dim=self.truedim)
        elif dim < self.truedim:
            return np.inf
        elif dim > self.truedim:
            return 0.0
        else:
            return np.prod((self._end - self._begin)[self._inondeg])

    def dist(self, point, ord=None):
        """
        Calculate the distance to a point

        Parameters
        ----------
        point : array-like or float
                The point. Its length must be equal to the set's
                dimension. In the 1d case, 'point' can be given as a
                float.
        ord : non-zero int or float('inf'), optional
              The order of the norm (see numpy.linalg.norm).
              Default: 2

        Examples
        --------

        >>> b, e = [-1, 0, 2], [-0.5, 0, 3]
        >>> rbox = IntervProd(b, e)
        >>> rbox.dist([-5, 3, 2])
        5.0
        >>> rbox.dist([-5, 3, 2], ord=float('inf'))
        4
        """

        # TODO: Apply same principle as in MetricProductSpace?
        point = np.atleast_1d(point)
        if len(point) != self.dim:
            raise ValueError(errfmt('''
            'point' dimension ({}) different from set dimension ({}).
            '''.format(len(point), self.dim)))

        i_larger = np.where(point > self._end)
        i_smaller = np.where(point < self._begin)
        proj = point.copy()
        proj[i_larger] = self._end[i_larger]
        proj[i_smaller] = self._begin[i_smaller]
        return np.linalg.norm(point - proj, ord=ord)

    # Manipulation
    def collapse(self, index, value):
        """
        Partly collapse the interval product to single values

        Note that no changes are made in-place.

        Parameters
        ----------
        index : int or tuple of ints
            The indices of the dimensions along which to collapse
        value : float or array-like
            The values to which to collapse. Must have the same
            lenght as 'index'. Values must lie within the interval
            boundaries.

        Returns
        -------
        The collapsed IntervProd

        Examples
        --------

        >>> b, e = [-1, 0, 2], [-0.5, 1, 3]
        >>> rbox = IntervProd(b, e)
        >>> rbox.collapse(1, 0)
        IntervProd([-1.0, 0.0, 2.0], [-0.5, 0.0, 3.0])
        >>> rbox.collapse([1, 2], [0, 2.5])
        IntervProd([-1.0, 0.0, 2.5], [-0.5, 0.0, 2.5])
        >>> rbox.collapse([1, 2], [0, 3.5])
        Traceback (most recent call last):
            ...
        ValueError: 'value' not within interval boundaries ([3.5] > [3.0])
        """

        index = np.atleast_1d(index)
        value = np.atleast_1d(value)
        if len(index) != len(value):
            raise ValueError(errfmt('''
            lengths of 'index' ({}) and 'value' ({}) do not match
            '''.format(len(index), len(value))))

        if np.any(index < 0) or np.any(index >= self.dim):
            raise IndexError(errfmt('''
            'index'({!r}) out of range (max {})
            '''.format(list(index), self.dim)))

        if np.any(value < self._begin[index]):
            i_smaller = np.where(value < self._begin[index])
            raise ValueError(errfmt('''
            'value' not within interval boundaries ({!r} < {!r})
            '''.format(list(value[i_smaller]),
                       list((self._begin[index])[i_smaller]))))

        if np.any(value > self._end[index]):
            i_larger = np.where(value > self._end[index])
            raise ValueError(errfmt('''
            'value' not within interval boundaries ({!r} > {!r})
            '''.format(list(value[i_larger]),
                       list((self._end[index])[i_larger]))))

        b_new = self._begin.copy()
        b_new[index] = value
        e_new = self._end.copy()
        e_new[index] = value

        return IntervProd(b_new, e_new)

    def squeeze(self):
        """
        Remove the collapsed dimensions

        Note that no changes are made in-place.

        Returns
        -------
        The squeezed IntervProd

        Examples
        --------

        >>> b, e = [-1, 0, 2], [-0.5, 1, 3]
        >>> rbox = IntervProd(b, e)
        >>> rbox.collapse(1, 0).squeeze()
        IntervProd([-1.0, 2.0], [-0.5, 3.0])
        >>> rbox.collapse([1, 2], [0, 2.5]).squeeze()
        IntervProd([-1.0], [-0.5])
        >>> rbox.collapse([0, 1, 2], [-1, 0, 2.5]).squeeze()
        IntervProd([], [])
        """

        b_new = self._begin[self._inondeg]
        e_new = self._end[self._inondeg]
        return IntervProd(b_new, e_new)

    def insert(self, other, index):
        """
        Insert another IntervProd before the given index

        The given IntervProd (dim=m) is inserted into the current
        one (dim=n) before the given index, resulting in a new
        IntervProd of dimension n+m.
        Note that no changes are made in-place.

        Parameters
        ----------
        other : IntervProd, float or array-like
                The IntervProd to be inserted. A float or array a is
                treated as an IntervProd(a, a).
        index : int
                The index of the dimension before which 'other' is to
                be inserted. Must fulfill 0 <= index <= dim.

        Returns
        -------
        The enlarged IntervProd

        Examples
        --------

        >>> rbox = IntervProd([-1, 2], [-0.5, 3])
        >>> rbox2 = IntervProd([0, 0], [1, 0])
        >>> rbox.insert(rbox2, 1)
        IntervProd([-1.0, 0.0, 0.0, 2.0], [-0.5, 1.0, 0.0, 3.0])
        >>> rbox.insert([-1.0, 0.0], 2)
        IntervProd([-1.0, 2.0, -1.0, 0.0], [-0.5, 3.0, -1.0, 0.0])
        >>> rbox.insert(0, 1).squeeze().equals(rbox)
        True
        """

        if not 0 <= index <= self.dim:
            raise IndexError('Index ({}) out of range'.format(index))

        if not isinstance(other, IntervProd):
            other = IntervProd(other, other)

        new_beg = np.empty(self.dim + other.dim)
        new_end = np.empty(self.dim + other.dim)

        new_beg[: index] = self._begin[: index]
        new_end[: index] = self._end[: index]
        new_beg[index: index+other.dim] = other.begin
        new_end[index: index+other.dim] = other.end
        if index < self.dim:  # Avoid IndexError
            new_beg[index+other.dim:] = self._begin[index:]
            new_end[index+other.dim:] = self._end[index:]

        return IntervProd(new_beg, new_end)

    # Magic methods
    def __repr__(self):
        return ('IntervProd({b!r}, {e!r})'.format(b=list(self._begin),
                                                  e=list(self._end)))

    __str__ = __repr__


class Interval(IntervProd):
    """ The set of real numbers in the interval [begin, end]
    """
    def __init__(self, begin, end):
        super().__init__(begin, end)
        if self.dim != 1:
            raise ValueError(errfmt('''
            'begin' and 'end' must scalar or have length 1 (got {}).
            '''.format(self.dim)))

    @property
    def length(self):
        return self.end - self.begin

    def __repr__(self):
        return 'Interval({b}, {e})'.format(b=self.begin, e=self.end)


class Rectangle(IntervProd):
    def __init__(self, begin, end):
        super().__init__(begin, end)
        if self.dim != 2:
            raise ValueError(errfmt('''
            Lengths of 'begin' and 'end' must be equal to 2 (got {}).
            '''.format(self.dim)))

    @property
    def area(self):
        return self.volume

    def __repr__(self):
        return ('Rectangle({b!r}, {e!r})'.format(b=list(self._begin),
                                                 e=list(self._end)))


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
