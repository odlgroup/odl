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
from future import standard_library
standard_library.install_aliases()

# External
from abc import ABCMeta, abstractmethod

# Internal
from odl.util.utility import with_metaclass

__all__ = ('Partial', 'StorePartial', 'ForEachPartial',
           'PrintIterationPartial', 'PrintStatusPartial')


class Partial(with_metaclass(ABCMeta, object)):

    """Abstract base class for sending partial results of iterations."""

    @abstractmethod
    def send(self, result):
        """Send the result to the partial object."""


class StorePartial(Partial):

    """Simple object for storing all partial results of the solvers."""

    def __init__(self):
        self.results = []

    def send(self, result):
        """Append result to results list."""
        self.results.append(result.copy())

    def __iter__(self):
        return self.results.__iter__()


class ForEachPartial(Partial):

    """Simple object for applying a function to each iterate."""

    def __init__(self, function):
        self.function = function

    def send(self, result):
        """Applies function to result."""
        self.function(result)


class PrintIterationPartial(Partial):

    """Print the interation count."""

    def __init__(self):
        self.iter = 0

    def send(self, _):
        """Print the current iteration."""
        print("iter = {}".format(self.iter))
        self.iter += 1


class PrintStatusPartial(Partial):

    """Print the interation count and current norm of each iterate."""

    def __init__(self):
        self.iter = 0

    def send(self, result):
        """Print the current iteration and norm."""
        print("iter = {}, norm = {}".format(self.iter, result.norm()))
        self.iter += 1


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
