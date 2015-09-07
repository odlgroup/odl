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
from __future__ import print_function, division, absolute_import
from __future__ import unicode_literals
from builtins import super
from future import standard_library
standard_library.install_aliases()
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta

# ODL imports
from odl.operator.operator import Operator, LinearOperator
from odl.space.cartesian import NtuplesBase, FnBase
from odl.space.set import Set
from odl.space.space import LinearSpace


class Discretization(with_metaclass(ABCMeta, Set)):

    """Abstract discretization class.

    A discretization in ODL is a way to encode the transition from
    an arbitrary set to a set of n-tuples explicitly representable
    in a computer. The most common use case is the discretization of
    an infinite-dimensional vector space of functions by means of
    storing coefficients in a finite basis.

    The minimal information required to create a discretization is
    the set to be discretized ("undiscretized space") and a backend
    for storage and processing of the n-tuples ("data space" or
    "discretized space").

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
    |`uspace`     |`Set`           |The set to be discretized         |
    +-------------+----------------+----------------------------------+
    |`dspace`     |`NtuplesBase`   |Data space providing structures to|
    |             |                |hold the values of a discretized  |
    |             |                |object                            |
    +-------------+----------------+----------------------------------+
    |`restriction`|`Operator`      |Operator mapping a `uspace`       |
    |             |                |element to a `dspace` element.    |
    |             |                |Raises `NotImplementedError` by   |
    |             |                |default.                          |
    +-------------+----------------+----------------------------------+
    |`extension`  |`Operator`      |Operator mapping a `dspace`       |
    |             |                |element to a `uspace` element.    |
    |             |                |Raises `NotImplementedError` by   |
    |             |                |default.                          |
    +-------------+----------------+----------------------------------+

    Methods
    -------

    +-----------+----------------+------------------------------------+
    |Signature  |Return type     |Description                         |
    +===========+================+====================================+
    |`element   |`Discretization.|Create an element either from       |
    |(inp=None)`|Vector`         |scratch using `dspace.element()` or |
    |           |                |from `inp` by calling               |
    |           |                |`dspace.element(inp)` or applying   |
    |           |                |`restriction` to                    |
    |           |                |`uspace.element(inp)`, in this      |
    |           |                |order.                              |
    +-----------+----------------+------------------------------------+
    """

    def __init__(self, uspace, dspace, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        uspace : `Set`
            The undiscretized (abstract) set to be discretized
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object
        restr : `Operator`, optional
            Operator mapping a `uspace` element to a `dspace` element.
            Must satisfy `restr.domain == uspace` and
            `restr.range == dspace`.
        ext : `Operator`, optional
            Operator mapping a `dspace` element to a `uspace` element.
            Must satisfy `ext.domain == dspace` and
            `ext.range == uspace`.
        """
        if not isinstance(uspace, Set):
            raise TypeError('undiscretized space {} not a `Set` instance.'
                            ''.format(uspace))

        if not isinstance(dspace, NtuplesBase):
            raise TypeError('data space {} not an `NtuplesBase` instance.'
                            ''.format(dspace))

        if restr is not None:
            if not isinstance(restr, Operator):
                raise TypeError('restriction operator {} not an `Operator` '
                                'instance.'.format(restr))

            if restr.domain != uspace:
                raise ValueError('restriction operator domain {} not equal to '
                                 'the undiscretized space {}.'
                                 ''.format(restr.domain, dspace))

            if restr.range != dspace:
                raise ValueError('restriction operator range {} not equal to'
                                 'the data space {}.'
                                 ''.format(restr.range, dspace))

        if ext is not None:
            if not isinstance(ext, Operator):
                raise TypeError('extension operator {} not an `Operator` '
                                'instance.'.format(ext))

            if ext.domain != dspace:
                raise ValueError('extension operator domain {} not equal to'
                                 'the data space {}.'
                                 ''.format(ext.domain, dspace))

            if ext.range != uspace:
                raise ValueError('extension operator range {} not equal to'
                                 'the undiscretized space {}.'
                                 ''.format(ext.range, uspace))

        self._uspace = uspace
        self._dspace = dspace
        self._restriction = restr
        self._extension = ext

    @property
    def uspace(self):
        """The undiscretized space."""
        return self._uspace

    @property
    def dspace(self):
        """The data space."""
        return self._dspace

    @property
    def restriction(self):
        """The operator mapping a `uspace` element to an n-tuple."""
        if self._restriction is not None:
            return self._restriction
        else:
            raise NotImplementedError('no restriction operator provided.')

    @property
    def extension(self):
        """The operator mapping an n-tuple to a `uspace` element."""
        if self._extension is not None:
            return self._extension
        else:
            raise NotImplementedError('no extension operator provided.')

    def element(self, inp=None):
        """Create an element from `inp` or from scratch.

        Parameters
        ----------
        inp : `object`, optional
            The input data to create an element from. Must be
            recognizable by the `element()` method of either `dspace`
            or `uspace`.

        Returns
        -------
        element : `Discretization.Vector`
            The discretized element, calculated as
            `dspace.element(inp)` or
            `restriction(uspace.element(inp))`, tried in this order.
        """
        try:
            elem = self.dspace.element(inp)
        except TypeError:
            elem = self.restriction(self.uspace.element(inp))
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
            all attributes `uspace`, `dspace`, `restriction` and
            `extension` of `other` and this discretization are equal,
            `False` otherwise.
        """
        return (isinstance(other, Discretization) and
                other.uspace == self.uspace and
                other.dspace == self.dspace and
                other.restriction == self.restriction and
                other.extension == self.extension)

    # Pass-through attributes of the wrapped `dspace`
    @property
    def dim(self):
        """The dimension of this discretization.

        Equals the dimension of the `dspace` attribute, i.e. the
        number of values representing a discretized element.
        """
        return self.dspace.dim

    @property
    def dtype(self):
        """The data type of this discretization.

        Equals the data type of the `dspace` attribute.
        """
        return self.dspace.dtype

    class Vector(NtuplesBase.Vector):

        """Representation of a `Discretization` element."""

        def __init__(self, space, data):
            """Initialize a new instance.

            Since `Discretization` does not subclass `Ntuples`,
            we must work around the error raised by the
            `Ntuples.Vector` initializer.
            """
            super().__init__(space.dspace, data)

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
    its members. The `uspace` is a linear space, the `dspace`
    for the data representation is an implementation of :math:`F^n`,
    where `F` is some field, and both `restriction` and `extension`
    are linear operators.
    """

    def __init__(self, uspace, dspace, restr=None, ext=None):
        """Abstract initialization method.

        Intended to be called by subclasses for proper type checking
        and setting of attributes.

        Parameters
        ----------
        uspace : `LinearSpace`
            The (abstract) space to be discretized
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `field` attribute must be the same
            as `uspace.field`.
        restr : `LinearOperator`, optional
            Operator mapping a `uspace` element to a `dspace` element.
            Must satisfy `restr.domain == uspace` and
            `restr.range == dspace`.
        ext : `LinearOperator`, optional
            Operator mapping an `dspace` element to a `uspace` element.
            Must satisfy `ext.domain == dspace` and
            `ext.range == uspace`.
        """
        super().__init__(uspace, dspace, restr, ext)

        if not isinstance(uspace, LinearSpace):
            raise TypeError('undiscretized space {} not a `LinearSpace` '
                            'instance.'.format(uspace))

        if not isinstance(dspace, FnBase):
            raise TypeError('data space {} not an `FnBase` instance.'
                            ''.format(dspace))

        if uspace.field != dspace.field:
            raise ValueError('fields {} and {} of the undiscretized and '
                             'data spaces, resp., are not equal.'
                             ''.format(uspace.field, dspace.field))

        if restr is not None:
            if not isinstance(restr, LinearOperator):
                raise TypeError('restriction operator {} is not a '
                                '`LinearOperator` instance.'.format(restr))

        if ext is not None:
            if not isinstance(ext, LinearOperator):
                raise TypeError('extension operator {} is not a '
                                '`LinearOperator` instance.'.format(ext))

    # Pass-through attributes of the wrapped `dspace`
    def _lincomb(self, z, a, x, b, y):
        """Raw linear combination."""
        return self.dspace._lincomb(z, a, x, b, y)

    @property
    def field(self):
        """The field of this discretization."""
        return self.dspace.field

    class Vector(Discretization.Vector, FnBase.Vector):

        """Representation of a `LinearSpaceDiscretization` element."""
