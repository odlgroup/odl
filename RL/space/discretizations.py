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


from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from future.builtins import object, zip
from future import standard_library
standard_library.install_aliases()

import RL.space.set as sets
import RL.space.space as space
from RL.space.function import L2
import numpy as np


def makeUniformDiscretization(parent, rnimpl):
    RNType = type(rnimpl)
    RNVectortype = RNType.Vector

    class UniformDiscretization(RNType):
        """ Uniform discretization of an interval
            Represents vectors by RN elements
            Uses trapezoid method for integration
        """

        def __init__(self, parent, rn):
            if not isinstance(parent.domain, sets.Interval):
                raise NotImplementedError("Can only discretize intervals")

            if not isinstance(rn, space.HilbertSpace):
                raise NotImplementedError("RN has to be a hilbert space")

            if not isinstance(rn, space.Algebra):
                raise NotImplementedError("RN has to be an algebra")

            self.parent = parent
            self._rn = rn
            self.scale = (self.parent.domain.end-self.parent.domain.begin)/(self.n-1)

        def innerImpl(self, v1, v2):
            return self._rn.innerImpl(v1, v2)*self.scale

        def normSqImpl(self, vector):
            return self._rn.normSqImpl(vector)*self.scale

        def __eq__(self, other):
            return isinstance(other, UniformDiscretization) and self.parent.equals(other.parent) and self._rn.equals(other._rn)

        def makeVector(self, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], L2.Vector):
                return self.makeVector(np.array([args[0](point) for point in self.points()], dtype=np.float))
            else:
                return RNType.makeVector(self, *args, **kwargs)

        def integrate(self, vector):
            return float(self._rn.sum(vector) * self.scale)

        def points(self):
            return np.linspace(self.parent.domain.begin, self.parent.domain.end, self.n)

        def __getattr__(self, name):
            return getattr(self._rn, name)

        def __str__(self):
            return "UniformDiscretization(" + str(self._rn) + ")"

        class Vector(RNVectortype):
            pass

    return UniformDiscretization(parent, rnimpl)


def makePixelDiscretization(parent, rnimpl, cols, rows, order='C'):
    """ Creates an pixel discretization of space parent using rn as the underlying representation.

    order indicates the order data is stored in, 'C'-order is the default numpy order, also called row major.
    """
    RNType = type(rnimpl)
    RNVectortype = RNType.Vector

    class PixelDiscretization(RNType):
        """ Uniform discretization of an square
            Represents vectors by RN elements
            Uses sum method for integration
        """

        def __init__(self, parent, rn, cols, rows, order):
            if not isinstance(parent.domain, sets.Square):
                raise NotImplementedError("Can only discretize Squares")

            if not isinstance(rn, space.HilbertSpace):
                raise NotImplementedError("RN has to be a hilbert space")

            if not isinstance(rn, space.Algebra):
                raise NotImplementedError("RN has to be an algebra")

            if not rn.dimension == cols*rows:
                raise NotImplementedError("Dimensions do not match, expected {}x{} = {}, got {}".format(cols, rows, cols*rows, rn.dimension))

            self.parent = parent
            self.cols = cols
            self.rows = rows
            self.order = order
            self._rn = rn
            dx = (self.parent.domain.end[0]-self.parent.domain.begin[0])/(self.cols-1)
            dy = (self.parent.domain.end[1]-self.parent.domain.begin[1])/(self.rows-1)
            self.scale = dx * dy

        def innerImpl(self, v1, v2):
            return self._rn.innerImpl(v1, v2)*self.scale

        def normSqImpl(self, vector):
            return self._rn.normSqImpl(vector)*self.scale

        def equals(self, other):
            return isinstance(other, PixelDiscretization) and self.cols == other.cols and self.rows == other.rows and self._rn.equals(other._rn)

        def makeVector(self, *args, **kwargs):
            if len(args) == 1 and isinstance(args[0], L2.Vector):
                return self.makeVector(np.array([args[0]([x, y]) for x, y in zip(*self.points())], dtype=np.float))
            elif len(args) == 1 and isinstance(args[0], np.ndarray):
                if args[0].shape == (self.cols, self.rows):
                    return self.makeVector(args[0].flatten(self.order))
                elif args[0].shape == (self.dimension,):
                    return RNType.makeVector(self, args[0])
                else:
                    raise ValueError("Input numpy array ({}) is of shape {}, expected shape shape {} or {}".format(args[0], args[0].shape, (self.n,), (self.cols, self.rows)))
            else:
                return RNType.makeVector(self, *args, **kwargs)

        def integrate(self, vector):
            return float(self._rn.sum(vector) * self.scale)

        def points(self):
            x, y = np.meshgrid(np.linspace(self.parent.domain.begin[0], self.parent.domain.end[0], self.cols),
                               np.linspace(self.parent.domain.begin[1], self.parent.domain.end[1], self.rows))
            return x.flatten(self.order), y.flatten(self.order)

        def __getattr__(self, name):
            return getattr(self._rn, name)

        def __str__(self):
            return "PixelDiscretization(" + str(self._rn) + ", " + str(self.cols) + "x" + str(self.rows) + ")"

        class Vector(RNVectortype):
            pass

    return PixelDiscretization(parent, rnimpl, cols, rows, order)