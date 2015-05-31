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

""" Examples of sequence spaces, function spaces defined on the integers.
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
from builtins import super

# External module imports
import numpy as np

# RL imports
from RL.space.euclidean import EuclidRn
from RL.space.function import FunctionSpace
from RL.space.set import Integers

standard_library.install_aliases()


class SequenceSpace(FunctionSpace):
    """The space of sequences
    """

    def __init__(self):
        FunctionSpace.__init__(self, Integers())

    def equals(self, other):
        return (isinstance(other, SequenceSpace) and
                super().equals(other))


class TruncationDiscretization(EuclidRn):
    """ Truncation discretization of the integers
    Represents vectors by R^n elements
    """

    def __init__(self, parent, n):
        if not isinstance(parent.domain, Integers):
            raise NotImplementedError("Can only discretize the integers")

        self.parent = parent
        super().__init__(n)

    def _inner(self, v1, v2):
        return super()._inner(v1, v2)

    def zero(self):
        return self.element(np.zeros(self.dim), copy=False)

    def element(self):
        # FIXME: Remove this function
        return self.element(np.empty(self.dim), copy=False)

    def equals(self, other):
        return (isinstance(other, TruncationDiscretization) and
                super().equals(other))

    def element(self, *args, **kwargs):
        # FIXME: This is incomplete and does not fully implement the new
        # element() behavior
        return self.__class__.Vector(self, *args, **kwargs)

    def integrate(self, vector):
        return vector.data.sum()

    def points(self):
        return np.arange(self.dim)

    class Vector(EuclidRn.Vector):
        def __init__(self, space, *args, **kwargs):
            if ((len(args) == 1 and
                 isinstance(args[0], SequenceSpace.Vector) and
                 args[0].space == space.parent)):

                super().__init__(space, args[0](space.points()),
                                 copy=False)
            else:
                super().__init__(space, *args, **kwargs)
