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
    from builtins import str, object
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, object
from future.utils import with_metaclass
from future import standard_library

# External module imports
from numbers import Number
from abc import ABCMeta, abstractmethod, abstractproperty

# RL imports
from RL.utility.utility import errfmt

standard_library.install_aliases()


class Functional(with_metaclass(ABCMeta, object)):
    """Abstract functional
    """

    # Abstract methods
    @abstractmethod
    def _apply(self, rhs):
        """Apply the functional,
        abstract, pseudocode:

        return function(rhs)
        """
        pass

    @abstractproperty
    def domain(self):
        """Get the domain of the functional
        """

    @abstractproperty
    def range(self):
        """Get the range of the functional
        """

    # Implicitly defined methods
    def apply(self, rhs):
        """Apply the operator with error checking
        """
        if not self.domain.contains(rhs):
            raise TypeError(errfmt('''
            rhs ({}) is not in the domain ({}) of this functional
            '''.format(rhs, self.domain)))

        result = self._apply(rhs)

        if not self.range.contains(result):
            raise TypeError(errfmt('''
            functional returned: ({}), is in wrong set
            '''.format(result)))

        return result

    def __call__(self, rhs):
        """Shorthand for self.apply(rhs)
        """
        return self.apply(rhs)

    def __add__(self, other):
        """Operator addition (pointwise)
        """

        if isinstance(other, Functional):  # Calculate sum
            return FunctionalSum(self, other)
        else:
            raise TypeError('Expected a Functional')

    def __mul__(self, other):
        """Pointwise multiplication of operators (A*B)(x) == A(x)*B(x)
        or scalar multiplication
        """

        if isinstance(other, Functional):
            return FunctionalPointwiseProduct(self, other)
        elif isinstance(other, Number):
            return FunctionalScalarMultiplication(self, other)
        else:
            raise TypeError('Expected a Functional or a scalar')

    __rmul__ = __mul__

    def __str__(self):
        return ('Functional ' + self.__class__.__name__ + ': ' +
                str(self.domain) + '->' + str(self.range))


class FunctionalSum(Functional):
    """Expression type for the sum of functionals
    """
    def __init__(self, op1, op2):
        if op1.range != op2.range or op1.domain != op2.domain:
            raise TypeError('Range and domain of functionals do not fit')

        self._op1 = op1
        self._op2 = op2

    def _apply(self, rhs):
        return self._op1._apply(rhs) + self._op2._apply(rhs)

    @property
    def domain(self):
        return self._op1.domain

    @property
    def range(self):
        return self._op1.range


class FunctionalPointwiseProduct(Functional):
    """Pointwise multiplication of functionals
    """

    def __init__(self, op1, op2):
        if op1.range != op2.range or op1.domain != op2.domain:
            raise TypeError('Range and domain of functionals do not fit')

        self._op1 = op1
        self._op2 = op2

    def _apply(self, rhs):
        return self._op1._apply(rhs) * self._op2._apply(rhs)

    @property
    def domain(self):
        return self._op1.domain

    @property
    def range(self):
        return self._op1.range


class FunctionalScalarMultiplication(Functional):
    """Expression type for the multiplication of functionals with scalars
    """

    def __init__(self, op, scalar):
        if not op.range.contains(scalar):
            raise TypeError('Scalar is not compatible with this functional')

        self._op = op
        self._scal = scalar

    def _apply(self, rhs):
        return self._scal * self._op._apply(rhs)

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range
