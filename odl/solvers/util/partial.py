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

"""Partial objects for per-iterate actions in iterative methods."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
from abc import ABCMeta, abstractmethod

# Internal
from odl.util.utility import with_metaclass

__all__ = ('Partial', 'StorePartial', 'ForEachPartial',
           'PrintIterationPartial', 'PrintNormPartial', 'ShowPartial')


class Partial(with_metaclass(ABCMeta, object)):

    """Abstract base class for sending partial results of iterations."""

    @abstractmethod
    def send(self, result):
        """Send the result to the partial object."""

    def __and__(self, other):
        """ Return ``self & other``

        Compose partials, calls both in sequence.

        Parameters
        ----------
        other : `Partial`
            The other partial to compose with

        Returns
        -------
        result : `Partial`
            A partial which's `send` method calls both constituends partials.

        Examples
        --------
        >>> store = StorePartial()
        >>> iter = PrintIterationPartial()
        >>> both = store & iter
        >>> both
        StorePartial() & PrintIterationPartial()
        """
        class AndPartial(Partial):
            def send(_, result):
                self.send(result)
                other.send(result)

            def __repr__(_):
                return '{} & {}'.format(self, other)

        return AndPartial()

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


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


class PrintNormPartial(Partial):

    """Print the current norm."""

    def __init__(self):
        self.iter = 0

    def send(self, result):
        """Print the current iteration and norm."""
        print("norm = {}".format(result.norm()))
        self.iter += 1


class ShowPartial(Partial):

    """Show the partial result."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.fig = None
        self.plot_every_nth = kwargs.pop('plot_every_nth', 1)
        self.niter = 0

    def send(self, x):
        """Show the current iteration."""
        self.niter += 1
        if self.niter % self.plot_every_nth == 0:
            self.fig = x.show(fig=self.fig, show=True,
                              *self.args, **self.kwargs)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
