# Copyright 2014, 2015 Jonas Adler
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

"""Operators to cast between spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.operator.operator import Operator

__all__ = ('Embedding', 'EmbeddingFnInFn', 'EmbeddingPowerSpaceInFn',
           'EmbeddingFnInPowerSpace')


class Embedding(Operator):
    """An operators to cast between spaces."""
    def __init__(self, target, origin):
        super().__init__(origin, target)


class EmbeddingFnInFn(Embedding):
    """ Embed a `Fn` space into another """

    def __init__(self, target, origin):
        # TODO: tests goes here

        super().__init__(target, origin)

    def _apply(self, x, out):
        out[:] = x.asflatarray()

    def _call(self, x):
        return self.range.element(x.asflatarray())

    @property
    def adjoint(self):
        return EmbeddingFnInFn(self.domain, self.range)


class EmbeddingPowerSpaceInFn(Embedding):
    """ Embed a `PowerSpace` of `Fn` space into a `Fn` """

    def __init__(self, target, origin):
        # TODO: tests goes here

        super().__init__(target, origin)

    def _apply(self, x, out):
        out[:] = x.asflatarray()

    def _call(self, x):
        return self.range.element(x.asflatarray())

    @property
    def adjoint(self):
        return EmbeddingFnInPowerSpace(self.domain, self.range)


class EmbeddingFnInPowerSpace(Embedding):
    """ Embed a `Fn` into `PowerSpace` of `Fn`'s"""

    def __init__(self, target, origin):
        # TODO: tests goes here

        super().__init__(target, origin)

    def _apply(self, x, out):
        index = 0
        for sub_vec in out:
            sub_vec[:] = x.asflatarray(index, index + sub_vec.size)
            index += sub_vec.size

    def _call(self, x):
        out = self.range.element()
        index = 0
        for sub_vec in out:
            sub_vec[:] = x.asflatarray(index, index + sub_vec.size)
            index += sub_vec.size
        return out

    @property
    def adjoint(self):
        return EmbeddingPowerSpaceInFn(self.domain, self.range)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
