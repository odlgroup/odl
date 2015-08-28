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
from builtins import super
from future import standard_library
standard_library.install_aliases()
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta

# ODL imports
from odl.operator.operator import Operator, LinearOperator
from odl.space.cartesian import (Ntuples, Rn, Cn, MetricRn, MetricCn,
                                 NormedRn, NormedCn, HilbertRn, HilbertCn)
from odl.space.set import Set
from odl.space.space import (LinearSpace, NormedSpace, MetricSpace,
                             HilbertSpace, Algebra)
from odl.utility.utility import errfmt


class Discretization(with_metaclass(ABCMeta, Set)):

    """Abstract discretization class.

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

    Attributes
    ----------

    +-------------+----------------+----------------------------------+
    |Name         |Type            |Description                       |
    +=============+================+==================================+
    |`set`        |`Set`           |The set to be discretized         |
    +-------------+----------------+----------------------------------+
    |`ntuples`    |`Ntuples`       |Data structure holding the values |
    |             |                |of a discretized object           |
    +-------------+----------------+----------------------------------+
    |`restriction`|`Operator`      |Operator mapping a `set` element  |
    |             |                |to an `ntuples` element. Raises   |
    |             |                |`NotImplementedError` by default. |
    +-------------+----------------+----------------------------------+
    |`extension`  |`Operator`      |Operator mapping an `ntuples`     |
    |             |                |element to a `set` element. Raises|
    |             |                |`NotImplementedError` by default. |
    +-------------+----------------+----------------------------------+

    Methods
    -------

    +-----------+----------------+------------------------------------+
    |Signature  |Return type     |Description                         |
    +===========+================+====================================+
    |`element   |`Discretization.|Create an element either from       |
    |(inp=None)`|Vector`         |scratch using `ntuples.element()` or|
    |           |                |from `inp` by calling               |
    |           |                |`ntuples.element(inp)` or applying  |
    |           |                |`restriction` to `set.element(inp)`,|
    |           |                |in this order.                      |
    +-----------+----------------+------------------------------------+
    """

    def __init__(self, set_, ntuples, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        set_ : `Set`
            The (abstract) set to be discretized
        ntuples : `Ntuples`
            Data structure holding the values of a discretized object
        restr : `Operator`, optional
            Operator mapping a `set` element to an `ntuples` element.
            Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        ext : `Operator`, optional
            Operator mapping an `ntuples` element to a `set` element.
            Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(set_, Set):
            raise TypeError(errfmt('''
            `set_` {} not a `Set` instance.'''.format(set_)))

        if not isinstance(ntuples, Ntuples):
            raise TypeError(errfmt('''
            `ntuples` {} not an `Ntuples` instance.'''.format(ntuples)))

        if restr is not None:
            if not isinstance(restr, Operator):
                raise TypeError(errfmt('''
                `restr` {} not an `Operator` instance.'''.format(restr)))

            if restr.domain != set_:
                raise ValueError(errfmt('''
                `restr.domain` {} not equal to `set_` {}.
                '''.format(restr.domain, set_)))

            if restr.range != ntuples:
                raise ValueError(errfmt('''
                `restr.range` {} not equal to `ntuples` {}.
                '''.format(restr.range, ntuples)))

        if ext is not None:
            if not isinstance(ext, Operator):
                raise TypeError(errfmt('''
                `ext` {} not an `Operator` instance.'''.format(ext)))

            if ext.domain != ntuples:
                raise ValueError(errfmt('''
                `ext.domain` {} not equal to `ntuples {}`.
                '''.format(ext.domain, ntuples)))

            if ext.range != set_:
                raise ValueError(errfmt('''
                `ext.range` {} not equal to `set_` {}.
                '''.format(ext.range, set_)))

        self._set = set_
        self._ntuples = ntuples
        self._restriction = restr
        self._extension = ext

    @property
    def set(self):
        """Return the `set` attribute."""
        return self._set

    @property
    def ntuples(self):
        """Return the `ntuples` attribute."""
        return self._ntuples

    @property
    def restriction(self):
        """The operator mapping a `set` element to an n-tuple."""
        if self._restriction is not None:
            return self._restriction
        else:
            raise NotImplementedError('no `restriction` provided.')

    @property
    def extension(self):
        """The operator mapping an n-tuple to a `set` element."""
        if self._extension is not None:
            return self._extension
        else:
            raise NotImplementedError('no `extension` provided.')

    def element(self, inp=None):
        """Create an element from `inp` or from scratch.

        Parameters
        ----------
        inp : `object`, optional
            The input data to create an element from. Must be
            recognizable by the `element()` method of either `ntuples`
            or `set`.

        Returns
        -------
        element : `Discretization.Vector`
            The discretized element, calculated as
            `ntuples.element(inp)` or
            `restriction(set.element(inp))`, tried in this order.
        """
        try:
            elem = self.ntuples.element(inp)
        except TypeError:
            elem = self.restriction(self.set.element(inp))
        return self.Vector(self, elem.data)

    def contains(self, other):
        """Test if `other` is a member of this discretization."""
        return (isinstance(other, Discretization.Vector) and
                other.space == self)

    def equals(self, other):
        """Test if `other` is equal to this discretization.

        Returns
        -------
        equals : `bool`
            `True` if `other` is a `Discretization` instance and
            all attributes `set`, `ntuples`, `restriction` and
            `extension` of `other` and this discretization are equal,
            `False` otherwise.
        """
        return (isinstance(other, Discretization) and
                other.set == self.set and
                other.ntuples == self.ntuples and
                other.restriction == self.restriction and
                other.extension == self.extension)

    # Pass-through attributes of the wrapped `ntuples`
    @property
    def dim(self):
        """The dimension of this discretization.

        Equals the dimension of the `ntuples` attribute, i.e. the
        number of values representing a discretized element.
        """
        return self.ntuples.dim

    @property
    def dtype(self):
        """The data type of this discretization.

        Equals the data type of the `ntuples` attribute.
        """
        return self.ntuples.dtype

    class Vector(Ntuples.Vector):

        """Representation of a `Discretization` element."""

        def __init__(self, space, data):
            """Initialize a new instance.

            Since `Discretization` does not subclass `Ntuples`,
            we must work around the error raised by the
            `Ntuples.Vector` initializer.
            """
            super().__init__(space.ntuples, data)

        def equals(self, other):
            """Test if `other` is equal to this vector.

            Returns
            -------
            equals: `bool`
                `True` if `other` belongs to this vector's space
                and their values are all equal.
            """
            return other in self.space and other.data == self.data


class LinearSpaceDiscretization(with_metaclass(ABCMeta, Discretization,
                                               LinearSpace)):

    """Abstract class for discretizations of linear vector spaces.

    This variant of `Discretization` adds linear structure to all
    its members. The `set` is a linear space, the `ntuples`
    for the data representation is an implementation of either
    :math:`R^n` or :math:`C^n`, and both `restriction` and
    `extension` are linear operators.
    """

    def __init__(self, space, ntuples, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        space : `LinearSpace`
            The (abstract) space to be discretized
        ntuples : `Rn` or `Cn`
            Data structure holding the values of a discretized object.
            Its `field` attribute must be the same as `space.field`.
        restr : `LinearOperator`, optional
            Operator mapping a `set` element to an `ntuples` element.
            Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        ext : `LinearOperator`, optional
            Operator mapping an `ntuples` element to a `set` element.
            Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(space, LinearSpace):
            raise TypeError(errfmt('''
            `space` {} not a `LinearSpace` instance.'''.format(space)))

        if not isinstance(ntuples, (Rn, Cn)):
            raise TypeError(errfmt('''
            `ntuples` {} not an `Rn` or `Cn` instance.'''.format(ntuples)))

        if space.field != ntuples.field:
            raise ValueError(errfmt('''
            `space.field` {} not equal to `ntuples.field` {}.
            '''.format(space.field, ntuples.field)))

        if restr is not None:
            if not isinstance(restr, LinearOperator):
                raise TypeError(errfmt('''
                `restr` {} not a `LinearOperator` instance.'''.format(restr)))

            if restr.domain != space:
                raise ValueError(errfmt('''
                `restr.domain` {} not equal to `space` {}.
                '''.format(restr.domain, space)))

            if restr.range != ntuples:
                raise ValueError(errfmt('''
                `restr.range` {} not equal to `ntuples` {}.
                '''.format(restr.range, ntuples)))

        if ext is not None:
            if not isinstance(ext, LinearOperator):
                raise TypeError(errfmt('''
                `ext` {} not a `LinearOperator` instance.'''.format(ext)))

            if ext.domain != ntuples:
                raise ValueError(errfmt('''
                `ext.domain` {} not equal to `ntuples {}`.
                '''.format(ext.domain, ntuples)))

            if ext.range != space:
                raise ValueError(errfmt('''
                `ext.range` {} not equal to `space` {}.
                '''.format(ext.range, space)))

        super().__init__(space, ntuples, restr, ext)

    @property
    def space(self):
        """Another name for `set`"""
        return self._set

    # Pass-through attributes of the wrapped `ntuples`
    def _lincomb(self, z, a, x, b, y):
        """Raw linear combination."""
        return self.ntuples._lincomb(z, a, x, b, y)

    @property
    def field(self):
        """The field of this discretization."""
        return self.ntuples.field

    class Vector(Discretization.Vector, Cn.Vector):

        """Representation of a `LinearSpaceDiscretization` element."""


class MetricSpaceDiscretization(with_metaclass(ABCMeta,
                                               LinearSpaceDiscretization,
                                               MetricSpace)):

    """Abstract class for discretizations of metric spaces."""

    def __init__(self, space, ntuples, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        space : `MetricSpace`
            The (abstract) space to be discretized
        ntuples : `MetricRn` or `MetricCn`
            Data structure holding the values of a discretized object.
            Its `field` attribute must be the same as `space.field`.
        restr : `LinearOperator`, optional
            Operator mapping a `set` element to an `ntuples` element.
            Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        ext : `LinearOperator`, optional
            Operator mapping an `ntuples` element to a `set` element.
            Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(space, MetricSpace):
            raise TypeError(errfmt('''
            `space` {} not a `MetricSpace` instance.'''.format(space)))

        if not isinstance(ntuples, (MetricRn, MetricCn)):
            raise TypeError(errfmt('''
            `ntuples` {} not a `MetricRn` or `MetricCn` instance.
            '''.format(ntuples)))

        super().__init__(space, ntuples, restr, ext)

    def _dist(self, x, y):
        """Raw distance implementation."""
        return self.ntuples._dist(x, y)


class NormedSpaceDiscretization(with_metaclass(ABCMeta,
                                               MetricSpaceDiscretization,
                                               NormedSpace)):

    """Abstract class for discretizations of normed spaces."""

    def __init__(self, space, ntuples, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        `space` : `NormedSpace`
            The (abstract) space to be discretized
        `ntuples` : `NormedRn` or `NormedCn`
            Data structure holding the values of a discretized object.
            Its `field` attribute must be the same as `space.field`.
        `restr` : `LinearOperator`, optional
            Operator mapping a `set` element to an `ntuples` element.
            Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        `ext` : `LinearOperator`, optional
            Operator mapping an `ntuples` element to a `set` element.
            Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(space, NormedSpace):
            raise TypeError(errfmt('''
            `space` {} not a `NormedSpace` instance.'''.format(space)))

        if not isinstance(ntuples, (NormedRn, NormedCn)):
            raise TypeError(errfmt('''
            `ntuples` {} not a `NormedRn` or `NormedCn` instance.
            '''.format(ntuples)))

        super().__init__(space, ntuples, restr, ext)

    def _norm(self, x):
        """Raw norm implementation."""
        return self.ntuples._norm(x)


class HilbertSpaceDiscretization(with_metaclass(ABCMeta,
                                                NormedSpaceDiscretization,
                                                HilbertSpace)):

    """Abstract class for discretizations of Hilbert spaces."""

    def __init__(self, space, ntuples, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        `space` : `HilbertSpace`
            The (abstract) space to be discretized
        `ntuples` : `HilbertRn` or `HilbertCn`
            Data structure holding the values of a discretized object.
            Its `field` attribute must be the same as `space.field`.
        `restr` : `LinearOperator`, optional
            Operator mapping a `set` element to an `ntuples` element.
            Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        `ext` : `LinearOperator`, optional
            Operator mapping an `ntuples` element to a `set` element.
            Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(space, HilbertSpace):
            raise TypeError(errfmt('''
            `space` {} not a `HilbertSpace` instance.'''.format(space)))

        if not isinstance(ntuples, (HilbertRn, HilbertCn)):
            raise TypeError(errfmt('''
            `ntuples` {} not a `HilbertRn` or `HilbertCn` instance.
            '''.format(ntuples)))

        super().__init__(space, ntuples, restr, ext)

    def _inner(self, x, y):
        """Raw inner product implementation."""
        return self.ntuples._inner(x, y)


class AlgebraDiscretization(with_metaclass(ABCMeta,
                                           LinearSpaceDiscretization,
                                           Algebra)):

    """Abstract class for discretizations of algebras."""

    def __init__(self, space, ntuples, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        `space` : `Algebra`
            The (abstract) space to be discretized
        `ntuples` : `Rn` or `Cn`
            Data structure holding the values of a discretized object.
            Its `field` attribute must be the same as `space.field`.
        `restr` : `LinearOperator`, optional
            Operator mapping a `set` element to an `ntuples` element.
            Must satisfy `restr.domain == set_` and
            `restr.range == ntuples`.
        `ext` : `LinearOperator`, optional
            Operator mapping an `ntuples` element to a `set` element.
            Must satisfy `ext.domain == ntuples` and
            `ext.range == set_`.
        """
        if not isinstance(space, Algebra):
            raise TypeError(errfmt('''
            `space` {} not a `HilbertSpace` instance.'''.format(space)))

        if not isinstance(ntuples, (Rn, Cn)):
            raise TypeError(errfmt('''
            `ntuples` {} not an `Rn` or `Cn` instance.
            '''.format(ntuples)))

        super().__init__(space, ntuples, restr, ext)

    def _multiply(self, x, y):
        """Raw multiply implementation."""
        return self.ntuples._multiply(x, y)


#def uniform_discretization(parent, rnimpl, shape=None, order='C'):
#    """ Creates an discretization of space parent using rn as the
#    underlying representation.
#
#    order indicates the order data is stored in, 'C'-order is the default
#    numpy order, also called row major.
#    """
#
#    rn_type = type(rnimpl)
#    rn_vector_type = rn_type.Vector
#
#    if shape is None:
#        shape = (rnimpl.dim,)
#
#    class UniformDiscretization(rn_type):
#        """ Uniform discretization of an square
#            Represents vectors by R^n elements
#            Uses sum method for integration
#        """
#
#        def __init__(self, parent, rn, shape, order):
#            if not isinstance(parent.domain, IntervalProd):
#                raise NotImplementedError('Can only discretize IntervalProds')
#
#            if not isinstance(rn, HilbertSpace):
#                pass
#                # raise NotImplementedError('Rn has to be a Hilbert space')
#
#            if not isinstance(rn, Algebra):
#                pass
#                # raise NotImplementedError('Rn has to be an algebra')
#
#            if rn.dim != np.prod(shape):
#                raise NotImplementedError(errfmt('''
#                Dimensions do not match, expected {}, got {}
#                '''.format(np.prod(rn.dim), np.prod(shape))))
#
#            self.parent = parent
#            self.shape = tuple(shape)
#            self.order = order
#            self._rn = rn
#            dx = np.array(
#                [((self.parent.domain.end[i] - self.parent.domain.begin[i]) /
#                  (self.shape[i] - 1)) for i in range(self.parent.domain.dim)])
#            self.scale = float(np.prod(dx))
#
#        def _inner(self, vec1, vec2):
#            return self._rn._inner(vec1, vec2) * self.scale
#
#        def _norm(self, vector):
#            return self._rn._norm(vector) * sqrt(self.scale)
#
#        def equals(self, other):
#            return (isinstance(other, UniformDiscretization) and
#                    self.shape == other.shape and
#                    self._rn.equals(other._rn))
#
#        def element(self, data=None, **kwargs):
#            if isinstance(data, FunctionSpace.Vector):
#                if self.parent.domain.dim == 1:
#                    tmp = np.array([data(point)
#                                    for point in self.points()],
#                                   **kwargs)
#                else:
#                    tmp = np.array([data(point)
#                                    for point in zip(*self.points())],
#                                   **kwargs)
#                return self.element(tmp)
#            elif data is not None:
#                data = np.asarray(data)
#                if data.shape == (self.dim,):
#                    return super().element(data)
#                elif data.shape == self.shape:
#                    return self.element(data.flatten(self.order))
#                else:
#                    raise ValueError(errfmt('''
#                    Input numpy array is of shape {}, expected shape
#                    {} or {}'''.format(data.shape, (self.dim,), self.shape)))
#            else:
#                return super().element(data, **kwargs)
#
#        def integrate(self, vector):
#            return float(self._rn.sum(vector) * self.scale)
#
#        def points(self):
#            if self.parent.domain.dim == 1:
#                return np.linspace(self.parent.domain.begin[0],
#                                   self.parent.domain.end[0],
#                                   self.shape[0])
#            else:
#                oned = [np.linspace(self.parent.domain.begin[i],
#                                    self.parent.domain.end[i],
#                                    self.shape[i])
#                        for i in range(self.parent.domain.dim)]
#
#                points = np.meshgrid(*oned)
#
#                return tuple(point.flatten(self.order) for point in points)
#
#        def __getattr__(self, name):
#            if name in self.__dict__:
#                return self.__dict__[name]
#            else:
#                return getattr(self._rn, name)
#
#        def __str__(self):
#            if len(self.shape) > 1:
#                return ('[' + repr(self.parent) + ', ' + str(self._rn) + ', ' +
#                        'x'.join(str(d) for d in self.shape) + ']')
#            else:
#                return '[' + repr(self.parent) + ', ' + str(self._rn) + ']'
#
#        def __repr__(self):
#            shapestr = (', ' + repr(self.shape)
#                        if self.shape != (self._rn.dim,) else '')
#            orderstr = ', ' + repr(self.order) if self.order != 'C' else ''
#
#            return ("uniform_discretization(" + repr(self.parent) + ", " +
#                    repr(self._rn) + shapestr + orderstr + ")")
#
#        class Vector(rn_vector_type):
#            def asarray(self):
#                return np.reshape(self[:], self.space.shape, self.space.order)
#
#    return UniformDiscretization(parent, rnimpl, shape, order)
