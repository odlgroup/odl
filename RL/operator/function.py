# Copyright 2014, 2015 Jonas Adler
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
    from builtins import str, range, object
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, range, object
from future.utils import with_metaclass
from future import standard_library

# External module imports
# from numbers import Number
from abc import ABCMeta, abstractmethod  # , abstractproperty

# RL imports
from RL.space.set import EmptySet,  # , AbstractSet
from RL.utility.utility import errfmt

standard_library.install_aliases()


class Function(with_metaclass(ABCMeta, object)):
    """Abstract function on some sets
    """

    def __init__(self, input, returns=EmptySet()):
        self._sets = input
        self._returns = returns

    @abstractmethod
    def applyImpl(self, *args):
        """Apply the function, abstract
        """

    # Implicitly defined operators
    def domain(self, index):
        """Get the set the index:th argument of the function should belong to
        """
        return self._sets[index]

    @property
    def range(self):
        """The return type of this function
        """
        return self._returns

    @property
    def nargs(self):
        """Get the number of arguments of this function
        """
        return len(self._sets)

    def apply(self, *args):
        if len(args) != self.nargs:
            raise TypeError(errfmt('''
            Number of arguments provided ({}) does not match the expected
            number ({}) of this function ({})
            '''.format(len(args), self.nargs, self)))

        for i in range(self.nargs):
            if not self.domain(i).isMember(args[i]):
                raise TypeError(errfmt('''
                The {}:th argument ({}) is not in the domain of this function
                ({})'''.format(i, args[i], self)))

        returnValue = self.applyImpl(*args)

        if not self.range.isMember(returnValue):
            raise TypeError(errfmt('''
            The return value ({}) is not in the range ({}) of this function
            ({})'''.format(returnValue, self.range, self)))

        return returnValue

    def __call__(self, *args):
        """Shorthand for self.apply(rhs)
        """
        self.apply(*args)

    def __str__(self):
        return ('Function ' + self.__class__.__name__ + '(' +
                ', '.join(str(self.domain(i)) for i in range(self.nargs)) +
                ')')


class LambdaFunction(Function):
    """Shorthand for defining a function with a lambda
    """

    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        Function.__init__(self, *args, **kwargs)

    def applyImpl(self, *args, **kwargs):
        self.fun(*args, **kwargs)
