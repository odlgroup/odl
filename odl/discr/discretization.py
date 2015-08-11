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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, zip, super
from future import standard_library
standard_library.install_aliases()

from math import sqrt

# External module imports
import numpy as np

# ODL imports
from odl.space.set import Set, IntervalProd
from odl.space.space import HilbertSpace, Algebra
from odl.space.function import FunctionSpace
from odl.space.cartesian import Ntuples
from odl.operator.operator import Operator
from odl.utility.utility import errfmt


class Discretization(Set):

    """General discretization class.

    A discretization in ODL is a way to encode the transition from
    an arbitrary set to a set of `n`-tuples explicitly representable
    in a computer. The most common use case is the discretization of
    an infinite-dimensional vector space of functions by means of
    storing coefficients in a finite basis.

    The minimal information required to create a discretization is
    the set to be discretized and a backend for storage and processing
    of the `n`-tuples.

    As additional information, two mappings can be provided.
    The first one is an explicit way to map an (abstract) element from
    the source set to an `n`-tuple. This mapping is called
    **restriction** in ODL.
    The second one encodes the converse way of mapping an `n`-tuple to
    an element of the original set. This mapping is called
    **extension**.
    """

    def __init__(self, set_, ntuples, restr=None, ext=None):
        """Initialize a new `Discretization` instance.

        Parameters
        ----------
        `set_` : `Set`
            The (abstract) set to be discretized
        `ntuples` : `Ntuples`
            Data structure holding the values of a discretized object
        `restr` : `Operator`, optional
            Mapping from a element of `set_` to an element of
            `ntuples`. Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        `ext` : `Operator`, optional
            Mapping from a element of `ntuples` to an element of
            `set_`. Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(set_, Set):
            raise TypeError(errfmt('''
            `set_` {} not a `Set` instance.'''.format(set_)))

        if not isinstance(ntuples, Ntuples):
            raise TypeError(errfmt('''
            `ntuples` {} not a `Ntuples` instance.'''.format(ntuples)))

        if restr is not None:
            if not isinstance(restr, Operator):
                raise TypeError(errfmt('''
                `restr` {} not an `Operator` instance.'''.format(restr)))

            if restr.domain != set_:
                raise ValueError(errfmt('''
                `domain` attribute {} of `restr` not equal to `set_`.
                '''.format(restr.domain)))

            if restr.range != ntuples:
                raise ValueError(errfmt('''
                `range` attribute {} of `restr` not equal to `ntuples`.
                '''.format(restr.range)))

        if ext is not None:
            if not isinstance(ext, Operator):
                raise TypeError(errfmt('''
                `ext` {} not an `Operator` instance.'''.format(ext)))

            if restr.domain != ntuples:
                raise ValueError(errfmt('''
                `domain` attribute {} of `restr` not equal to `ntuples`.
                '''.format(restr.domain)))

            if restr.range != set_:
                raise ValueError(errfmt('''
                `range` attribute {} of `restr` not equal to `set_`.
                '''.format(restr.range)))

        self._set = set_
        self._ntuples = ntuples
        self._restriction = restr
        self._extension = ext

    @property
    def set(self):
        """Return `set` attribute."""
        return self._set

    @property
    def ntuples(self):
        """Return `ntuples` attribute."""
        return self._ntuples

    @property
    def restriction(self):
        """Return `restriction` attribute."""
        return self._restriction

    @property
    def extension(self):
        """Return `extension` attribute."""
        return self._extension


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
            if not isinstance(parent.domain, IntervalProd):
                raise NotImplementedError('Can only discretize IntervalProds')

            if not isinstance(rn, HilbertSpace):
                pass
                # raise NotImplementedError('Rn has to be a Hilbert space')

            if not isinstance(rn, Algebra):
                pass
                # raise NotImplementedError('Rn has to be an algebra')

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

        def _inner(self, vec1, vec2):
            return self._rn._inner(vec1, vec2) * self.scale

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
                oned = [np.linspace(self.parent.domain.begin[i],
                                    self.parent.domain.end[i],
                                    self.shape[i])
                        for i in range(self.parent.domain.dim)]

                points = np.meshgrid(*oned)

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
