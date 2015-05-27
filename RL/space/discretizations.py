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

"""
Default implementations of discretizations of sets using an underlying
RN representation
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
try:
    from builtins import str, zip, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, zip, super
from future import standard_library

from math import sqrt

# External module imports
import numpy as np

# RL imports
import RL.space.set as sets
import RL.space.space as space
from RL.space.function import FunctionSpace
from RL.utility.utility import errfmt

standard_library.install_aliases()


def uniform_discretization(parent, rnimpl):
    """ Creates an UniformDiscretization of space parent using rn as the
    underlying representation.
    """
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
                raise NotImplementedError("RN has to be a Hilbert space")

            if not isinstance(rn, space.Algebra):
                raise NotImplementedError("RN has to be an algebra")

            self.parent = parent
            self._rn = rn
            self.scale = (self.parent.domain.length / (self.n - 1))

        def _inner(self, v1, v2):
            return self._rn._inner(v1, v2) * self.scale

        def _norm(self, vector):
            return self._rn._norm(vector) * sqrt(self.scale)

        def __eq__(self, other):
            return (isinstance(other, UniformDiscretization) and
                    self.parent.equals(other.parent) and
                    self._rn.equals(other._rn))

        def element(self, data=None, **kwargs):
            # TODO: 'data' is not a good name here
            if isinstance(data, FunctionSpace.Vector):
                tmp = np.array([data(point) for point in self.points()],
                               **kwargs)
                return self.element(tmp)
            else:
                return super().element(data, **kwargs)

        def integrate(self, vector):
            return float(self._rn.sum(vector) * self.scale)

        def points(self):
            return np.linspace(self.parent.domain.begin,
                               self.parent.domain.end, self.n)

        def __getattr__(self, name):
            return getattr(self._rn, name)

        def __str__(self):
            return "UniformDiscretization(" + str(self._rn) + ")"

        def __repr__(self):
            return ("UniformDiscretization(" + repr(self.parent) + "," +
                    repr(self._rn) + ")")

        class Vector(RNVectortype):
            pass

    return UniformDiscretization(parent, rnimpl)


def pixel_discretization(parent, rnimpl, cols, rows, order='C'):
    """ Creates an pixel discretization of space parent using rn as the
    underlying representation.

    order indicates the order data is stored in, 'C'-order is the default
    numpy order, also called row major.
    """
    RNType = type(rnimpl)
    RNVectortype = RNType.Vector

    class PixelDiscretization(RNType):
        """ Uniform discretization of an square
            Represents vectors by RN elements
            Uses sum method for integration
        """

        def __init__(self, parent, rn, cols, rows, order):
            if not isinstance(parent.domain, sets.Rectangle):
                raise NotImplementedError('Can only discretize Squares')

            if not isinstance(rn, space.HilbertSpace):
                raise NotImplementedError('RN has to be a Hilbert space')

            if not isinstance(rn, space.Algebra):
                raise NotImplementedError('RN has to be an algebra')

            if not rn.n == cols*rows:
                raise NotImplementedError(errfmt('''
                Dimensions do not match, expected {}x{} = {}, got {}
                '''.format(cols, rows, cols*rows, rn.n)))

            self.parent = parent
            self.cols = cols
            self.rows = rows
            self.order = order
            self._rn = rn
            dx = ((self.parent.domain.end[0] - self.parent.domain.begin[0]) /
                  (self.cols - 1))
            dy = ((self.parent.domain.end[1] - self.parent.domain.begin[1]) /
                  (self.rows - 1))
            self.scale = dx * dy

        def _inner(self, v1, v2):
            return self._rn._inner(v1, v2) * self.scale

        def _norm(self, vector):
            return self._rn._norm(vector) * sqrt(self.scale)

        def equals(self, other):
            return (isinstance(other, PixelDiscretization) and
                    self.cols == other.cols and self.rows == other.rows and
                    self._rn.equals(other._rn))

        def element(self, data=None, **kwargs):
            if isinstance(data, FunctionSpace.Vector):
                tmp = np.array([data([x, y])
                                for x, y in zip(*self.points())],
                               **kwargs)
                return self.element(tmp)

            elif isinstance(data, np.ndarray):
                if data.shape == (self.cols, self.rows):
                    return self.element(data.flatten(self.order))
                elif data.shape == (self.n,):
                    return super().element(data)
                else:
                    raise ValueError(errfmt('''
                    Input numpy array is of shape {}, expected shape
                    {} or {}'''.format(data.shape, (self.n,),
                                       (self.cols, self.rows))))
            else:
                return super().element(data, **kwargs)

        def integrate(self, vector):
            return float(self._rn.sum(vector) * self.scale)

        def points(self):
            x, y = np.meshgrid(np.linspace(self.parent.domain.begin[0],
                                           self.parent.domain.end[0],
                                           self.cols),
                               np.linspace(self.parent.domain.begin[1],
                                           self.parent.domain.end[1],
                                           self.rows))
            return x.flatten(self.order), y.flatten(self.order)

        def __getattr__(self, name):
            return getattr(self._rn, name)

        def __str__(self):
            return ('PixelDiscretization(' + str(self._rn) + ', ' +
                    str(self.cols) + 'x' + str(self.rows) + ')')


        def __repr__(self):
            return ("PixelDiscretization(" + repr(self.parent) + ", " +
                    repr(self._rn) + ", " +
                    str(self.cols) + ', ' +
                    str(self.rows) + ")")

        class Vector(RNVectortype):
            pass

    return PixelDiscretization(parent, rnimpl, cols, rows, order)
