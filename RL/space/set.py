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
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
try:
    from builtins import object, super, zip
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import object, super, zip
from future.utils import with_metaclass
from future import standard_library

# External module imports
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real, Complex
import numpy as np

# RL imports
from RL.utility.utility import errfmt

standard_library.install_aliases()


class AbstractSet(with_metaclass(ABCMeta, object)):
    """ An arbitrary set
    """

    @abstractmethod
    def equals(self, other):
        """ Test two sets for equality
        """

    @abstractmethod
    def contains(self, other):
        """ Test if other is a member of self
        """

    # Implicitly default implemented methods
    def __eq__(self, other):
        return self.equals(other)

    def __ne__(self, other):
        return not self.equals(other)

    def __contains__(self, other):
        return self.contains(other)


class EmptySet(AbstractSet):
    """ The empty set has no members (None is considered "no element")
    """
    def equals(self, other):
        return isinstance(other, EmptySet)

    def contains(self, other):
        return other is None


class ComplexNumbers(AbstractSet):
    """ The set of complex numbers
    """

    def equals(self, other):
        return isinstance(other, ComplexNumbers)

    def contains(self, other):
        return isinstance(other, Complex)


class RealNumbers(ComplexNumbers):
    """ The set of real numbers
    """

    def equals(self, other):
        return isinstance(other, RealNumbers)

    def contains(self, other):
        return isinstance(other, Real)


class Integers(RealNumbers):
    """ The set of all non-negative integers
    """

    def equals(self, other):
        return isinstance(other, Integers)

    def contains(self, other):
        return isinstance(other, Integral)


class IntervalProd(AbstractSet):
    """The product of N intervals, i.e. an N-dimensional rectangular
    box (aligned with the coordinate axes) as a subset of R^N.
    """

    def __init__(self, begin, end):
        begin = np.atleast_1d(begin)
        end = np.atleast_1d(end)

        if len(begin) != len(end):
            raise ValueError(errfmt('''
            Lengths of 'begin' ({}) and 'end' ({}) do not match.
            '''.format(len(begin), len(end))))

        if not np.all(begin <= end):
            raise ValueError(errfmt('''
            Entries of 'begin' may not exceed those of 'end'.'''))

        self._begin = begin
        self._end = end

    # TODO: setters?
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
    def volume(self):
        return np.prod(self._end - self._begin)

    def equals(self, other):
        return (isinstance(other, IntervalProd) and
                np.all(self.begin == other.begin) and
                np.all(self.end == other.end))

    def contains(self, other):
        other = np.atleast_1d(other)
        if len(other) != self.dim:
            return False

        reals = RealNumbers()
        for i, (begin_i, end_i) in enumerate(zip(self._begin, self._end)):
            if other[i] not in reals:
                return False
            if not begin_i <= other[i] <= end_i:
                return False
        return True

    def __repr__(self):
        return ('IntervalProd({b}, {e})'.format(b=self.begin, e=self.end))

    def __str__(self):
        return self.__repr__()


class Interval(IntervalProd):
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
        return ('Interval({b}, {e})'.format(b=self.begin, e=self.end))


class Rectangle(IntervalProd):
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
        return ('Rectangle({b}, {e})'.format(b=self.begin, e=self.end))
