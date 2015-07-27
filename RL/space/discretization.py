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
from builtins import str, zip, super
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


def uniform_discretization(parent, rnimpl, shape=None, order='C'):
    """ Creates an discretization of space parent using rn as the
    underlying representation.

    order indicates the order data is stored in, 'C'-order is the default
    numpy order, also called row major.
    """

    rn_type = type(rnimpl)
    rn_vector_type = rn_type.Vector

    if shape is None:
        shape = (rnimpl.dim,)

    class UniformDiscretization(rn_type):
        """ Uniform discretization of an square
            Represents vectors by R^n elements
            Uses sum method for integration
        """

        def __init__(self, parent, rn, shape, order):
            if not isinstance(parent.domain, sets.IntervalProd):
                raise NotImplementedError('Can only discretize IntervalProds')

            if not isinstance(rn, space.HilbertSpace):
                pass
                # raise NotImplementedError('RN has to be a Hilbert space')

            if not isinstance(rn, space.Algebra):
                pass
                # raise NotImplementedError('RN has to be an algebra')

            if rn.dim != np.prod(shape):
                raise NotImplementedError(errfmt('''
                Dimensions do not match, expected {}, got {}
                '''.format(np.prod(rn.dim), np.prod(shape))))

            self.parent = parent
            self.shape = tuple(shape)
            self.order = order
            self._rn = rn
            dx = np.array(
                [((self.parent.domain.end[i] - self.parent.domain.begin[i]) /
                 (self.shape[i] - 1)) for i in range(self.parent.domain.dim)])
            self.scale = float(np.prod(dx))

        def _inner(self, v1, v2):
            return self._rn._inner(v1, v2) * self.scale

        def _norm(self, vector):
            return self._rn._norm(vector) * sqrt(self.scale)

        def equals(self, other):
            return (isinstance(other, UniformDiscretization) and
                    self.shape == other.shape and
                    self._rn.equals(other._rn))

        def element(self, data=None, **kwargs):
            if isinstance(data, FunctionSpace.Vector):
                if self.parent.domain.dim == 1:
                    tmp = np.array([data(point)
                                    for point in self.points()],
                                   **kwargs)
                else:
                    tmp = np.array([data(point)
                                    for point in zip(*self.points())],
                                   **kwargs)
                return self.element(tmp)
            elif data is not None:
                data = np.asarray(data)
                if data.shape == (self.dim,):
                    return super().element(data)
                elif data.shape == self.shape:
                    return self.element(data.flatten(self.order))
                else:
                    raise ValueError(errfmt('''
                    Input numpy array is of shape {}, expected shape
                    {} or {}'''.format(data.shape, (self.dim,), self.shape)))
            else:
                return super().element(data, **kwargs)

        def integrate(self, vector):
            return float(self._rn.sum(vector) * self.scale)

        def points(self):
            if self.parent.domain.dim == 1:
                return np.linspace(self.parent.domain.begin[0],
                                   self.parent.domain.end[0],
                                   self.shape[0])
            else:
                oneD = [np.linspace(self.parent.domain.begin[i],
                        self.parent.domain.end[i],
                        self.shape[i]) for i in range(self.parent.domain.dim)]

                points = np.meshgrid(*oneD)

                return tuple(point.flatten(self.order) for point in points)

        def __getattr__(self, name):
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                return getattr(self._rn, name)

        def __str__(self):
            if len(self.shape) > 1:
                return ('[' + repr(self.parent) + ', ' + str(self._rn) + ', ' +
                        'x'.join(str(d) for d in self.shape) + ']')
            else:
                return '[' + repr(self.parent) + ', ' + str(self._rn) + ']'

        def __repr__(self):
            shapestr = (', ' + repr(self.shape)
                        if self.shape != (self._rn.dim,) else '')
            orderstr = ', ' + repr(self.order) if self.order != 'C' else ''

            return ("uniform_discretization(" + repr(self.parent) + ", " +
                    repr(self._rn) + shapestr + orderstr + ")")

        class Vector(rn_vector_type):
            def asarray(self):
                return np.reshape(self[:], self.space.shape, self.space.order)

    return UniformDiscretization(parent, rnimpl, shape, order)
