# Copyright 2014-2016 The ODL development group
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

"""Default operators defined on any (reasonable) space."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from copy import copy

from odl.operator.operator import Operator
from odl.space import ProductSpace
from odl.set import LinearSpace, LinearSpaceElement, Field, RealNumbers


__all__ = ('ScalingOperator', 'ZeroOperator', 'IdentityOperator',
           'LinCombOperator', 'MultiplyOperator', 'PowerOperator',
           'InnerProductOperator', 'NormOperator', 'DistOperator',
           'ConstantOperator', 'ResidualOperator')


class ScalingOperator(Operator):

    """Operator of multiplication with a scalar.

        ``ScalingOperator(s)(x) == s * x``
    """

    def __init__(self, space, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements on which this operator acts.
        scalar : ``space.field`` element
            Fixed scaling factor of this operator.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> op = ScalingOperator(r3, 2.0)
        >>> op(vec, out)  # In-place, Returns out
        rn(3).element([2.0, 4.0, 6.0])
        >>> out
        rn(3).element([2.0, 4.0, 6.0])
        >>> op(vec)  # Out-of-place
        rn(3).element([2.0, 4.0, 6.0])
        """
        if not isinstance(space, LinearSpace):
            raise TypeError('`space` {!r} not a LinearSpace instance'
                            ''.format(space))

        super().__init__(space, space, linear=True)
        self.__scalar = space.field.element(scalar)

    @property
    def scalar(self):
        """Fixed scaling factor of this operator."""
        return self.__scalar

    def _call(self, x, out=None):
        """Scale ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.scalar * x
        else:
            out.lincomb(self.scalar, x)
        return out

    @property
    def inverse(self):
        """Return the inverse operator.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = ScalingOperator(r3, 2.0)
        >>> inv = op.inverse
        >>> inv(op(vec)) == vec
        True
        >>> op(inv(vec)) == vec
        True
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('scaling operator not invertible for '
                                    'scalar==0')
        return ScalingOperator(self.domain, 1.0 / self.scalar)

    @property
    def adjoint(self):
        """Adjoint, given as scaling with the conjugate of the scalar.

        Returns
        -------
        adjoint : `ScalingOperator`
            ``self`` if `scalar` is real, else `scalar` is conjugated.
        """
        if complex(self.scalar).imag == 0.0:
            return self
        else:
            return ScalingOperator(self.domain, self.scalar.conjugate())

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.scalar)

    def __str__(self):
        """Return ``str(self)``."""
        return '{} * I'.format(self.scalar)


class IdentityOperator(ScalingOperator):

    """Operator mapping each element to itself.

        ``IdentityOperator()(x) == x``
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements which the operator is acting on.
        """
        super().__init__(space, 1)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return "I"


class LinCombOperator(Operator):

    """Operator mapping two space elements to a linear combination.

        ``LinCombOperator(a, b)(x, y) == a * x + b * y``
    """

    def __init__(self, space, a, b):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements which the operator is acting on.
        a, b : ``space.field`` elements
            Scalars to multiply ``x[0]`` and ``x[1]`` with, respectively.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> r3xr3 = odl.ProductSpace(r3, r3)
        >>> xy = r3xr3.element([[1, 2, 3], [1, 2, 3]])
        >>> z = r3.element()
        >>> op = LinCombOperator(r3, 1.0, 1.0)
        >>> op(xy, out=z)  # Returns z
        rn(3).element([2.0, 4.0, 6.0])
        >>> z
        rn(3).element([2.0, 4.0, 6.0])
        """
        domain = ProductSpace(space, space)
        super().__init__(domain, space, linear=True)
        self.a = a
        self.b = b

    def _call(self, x, out=None):
        """Linearly combine ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.range.element()
        out.lincomb(self.a, x[0], self.b, x[1])
        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r})'.format(self.__class__.__name__,
                                             self.range, self.a, self.b)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}*x + {}*y".format(self.a, self.b)


class MultiplyOperator(Operator):

    """Operator multiplying by a fixed space or field element.

        ``MultiplyOperator(y)(x) == x * y``

    Here, ``y`` is a `LinearSpaceElement` or `Field` element and
    ``x`` is a `LinearSpaceElement`.
    Hence, this operator can be defined either on a `LinearSpace` or on
    a `Field`. In the first case it is the pointwise multiplication,
    in the second the scalar multiplication.
    """

    def __init__(self, multiplicand, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        multiplicand : `LinearSpaceElement` or scalar
            Value to multiply by.
        domain : `LinearSpace` or `Field`, optional
            Set to which the operator can be applied.
            Default: ``multiplicand.space``.
        range : `LinearSpace` or `Field`, optional
            Set to which the operator maps. Default: ``multiplicand.space``.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])

        Multiply by vector:

        >>> op = MultiplyOperator(x)
        >>> op(x)
        rn(3).element([1.0, 4.0, 9.0])
        >>> out = r3.element()
        >>> op(x, out)
        rn(3).element([1.0, 4.0, 9.0])

        Multiply by scalar:

        >>> op2 = MultiplyOperator(x, domain=r3.field)
        >>> op2(3)
        rn(3).element([3.0, 6.0, 9.0])
        >>> out = r3.element()
        >>> op2(3, out)
        rn(3).element([3.0, 6.0, 9.0])
        """
        if domain is None:
            domain = multiplicand.space

        if range is None:
            range = multiplicand.space

        self.__multiplicand = multiplicand
        self.__domain_is_field = isinstance(domain, Field)
        self.__range_is_field = isinstance(range, Field)
        super().__init__(domain, range, linear=True)

    @property
    def multiplicand(self):
        """Value to multiply by."""
        return self.__multiplicand

    def _call(self, x, out=None):
        """Multiply ``x`` and write to ``out`` if given."""
        if out is None:
            return x * self.multiplicand
        elif not self.__range_is_field:
            if self.__domain_is_field:
                out.lincomb(x, self.multiplicand)
            else:
                x.multiply(self.multiplicand, out=out)
        else:
            raise ValueError('can only use `out` with `LinearSpace` range')

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `InnerProductOperator` or `MultiplyOperator`
            If the domain of this operator is the scalar field of a
            `LinearSpace` the adjoint is the inner product with ``y``,
            else it is the multiplication with ``y``.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])

        Multiply by a space element:

        >>> op = MultiplyOperator(x)
        >>> out = r3.element()
        >>> op.adjoint(x)
        rn(3).element([1.0, 4.0, 9.0])

        Multiply by a scalar:

        >>> op2 = MultiplyOperator(x, domain=r3.field)
        >>> op2.adjoint(x)
        14.0
        """
        if self.__domain_is_field:
            return InnerProductOperator(self.multiplicand)
        else:
            # TODO: complex case
            return MultiplyOperator(self.multiplicand)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.multiplicand)

    def __str__(self):
        """Return ``str(self)``."""
        return "x * {}".format(self.y)


class PowerOperator(Operator):

    """Operator taking a fixed power of a space or field element.

        ``PowerOperator(p)(x) == x ** p``

    Here, ``x`` is a `LinearSpaceElement` or `Field` element and ``p`` is
    a number. Hence, this operator can be defined either on a
    `LinearSpace` or on a `Field`.
    """

    def __init__(self, domain, exponent):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace` or `Field`, optional
            Set of elements on which the operator can be applied.
        exponent : float
            Exponent parameter of the power function applied to an element.

        Examples
        --------
        Use with vectors

        >>> import odl
        >>> op = PowerOperator(odl.rn(3), exponent=2)
        >>> op([1, 2, 3])
        rn(3).element([1.0, 4.0, 9.0])

        or scalars

        >>> op = PowerOperator(odl.RealNumbers(), exponent=2)
        >>> op(2.0)
        4.0
        """

        self.__exponent = float(exponent)
        self.__domain_is_field = isinstance(domain, Field)
        super().__init__(domain, domain, linear=(exponent == 1))

    @property
    def exponent(self):
        """Power of the input element to take."""
        return self.__exponent

    def _call(self, x, out=None):
        """Take the power of ``x`` and write to ``out`` if given."""
        if out is None:
            return x ** self.exponent
        elif self.__domain_is_field:
            raise ValueError('cannot use `out` with field')
        else:
            out.assign(x)
            out **= self.exponent

    def derivative(self, point):
        """Derivative of this operator.

            ``PowerOperator(p).derivative(y)(x) == p * y ** (p - 1) * x``

        Parameters
        ----------
        point : `domain` element
            The point in which to take the derivative

        Returns
        -------
        derivative : `Operator`
            The derivative in ``point``

        Examples
        --------
        >>> import odl

        Use on vector spaces:

        >>> op = PowerOperator(odl.rn(3), exponent=2)
        >>> dop = op.derivative(op.domain.element([1, 2, 3]))
        >>> dop([1, 1, 1])
        rn(3).element([2.0, 4.0, 6.0])

        Use with scalars:

        >>> op = PowerOperator(odl.RealNumbers(), exponent=2)
        >>> dop = op.derivative(2.0)
        >>> dop(2.0)
        8.0
        """
        return self.exponent * MultiplyOperator(point ** (self.exponent - 1),
                                                domain=self.domain,
                                                range=self.range)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self.domain, self.exponent)

    def __str__(self):
        """Return ``str(self)``."""
        return "x ** {}".format(self.exponent)


class InnerProductOperator(Operator):
    """Operator taking the inner product with a fixed space element.

        ``InnerProductOperator(y)(x) <==> y.inner(x)``

    This is only applicable in inner product spaces.

    See Also
    --------
    DistOperator : Distance to a fixed space element.
    NormOperator : Vector space norm as operator.
    """

    def __init__(self, vector):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceElement`
            The element to take the inner product with.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = InnerProductOperator(x)
        >>> op(r3.element([1, 2, 3]))
        14.0
        """
        self.__vector = vector
        super().__init__(vector.space, vector.space.field, linear=True)

    @property
    def vector(self):
        """Element to take the inner product with."""
        return self.__vector

    def _call(self, x):
        """Return the inner product with ``x``."""
        return x.inner(self.vector)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `MultiplyOperator`
            The operator of multiplication with `vector`.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = InnerProductOperator(x)
        >>> op.adjoint(2.0)
        rn(3).element([2.0, 4.0, 6.0])
        """
        return MultiplyOperator(self.vector, self.vector.space.field)

    @property
    def T(self):
        """Fixed vector of this operator.

        Returns
        -------
        vector : `LinearSpaceElement`
            The fixed space element used in this inner product operator.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> x.T
        InnerProductOperator(rn(3).element([1.0, 2.0, 3.0]))
        >>> x.T.T
        rn(3).element([1.0, 2.0, 3.0])
        """
        return self.vector

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}.T'.format(self.vector)


class NormOperator(Operator):

    """Vector space norm as an operator.

        ``NormOperator()(x) <==> x.norm()``

    This is only applicable in normed spaces.

    See Also
    --------
    InnerProductOperator : Inner product with a fixed space element.
    DistOperator : Distance to a fixed space element.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space to take the norm in.

        Examples
        --------
        >>> import odl
        >>> r2 = odl.rn(2)
        >>> op = NormOperator(r2)
        >>> op([3, 4])
        5.0
        """
        super().__init__(space, RealNumbers(), linear=False)

    def _call(self, x):
        """Return the norm of ``x``."""
        return x.norm()

    def derivative(self, point):
        """Derivative of this operator in ``point``.

            ``NormOperator().derivative(y)(x) == (y / y.norm()).inner(x)``

        This is only applicable in inner product spaces.

        Parameters
        ----------
        point : `domain` `element-like`
            Point in which to take the derivative.

        Returns
        -------
        derivative : `InnerProductOperator`

        Raises
        ------
        ValueError
            If ``point.norm() == 0``, in which case the derivative is not well
            defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::

            (D \|\cdot\|)(y)(x) = \langle y / \|y\|, x \\rangle

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> op = NormOperator(r3)
        >>> derivative = op.derivative([1, 0, 0])
        >>> derivative([1, 0, 0])
        1.0
        """
        point = self.domain.element(point)
        norm = point.norm()
        if norm == 0:
            raise ValueError('not differentiable in 0')

        return InnerProductOperator(point / norm)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({})'.format(self.__class__.__name__, self.domain)


class DistOperator(Operator):

    """Operator taking the distance to a fixed space element.

        ``DistOperator(y)(x) == y.dist(x)``

    This is only applicable in metric spaces.

    See Also
    --------
    InnerProductOperator : Inner product with fixed space element.
    NormOperator : Vector space norm as an operator.
    """

    def __init__(self, vector):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceElement`
            Point to calculate the distance to.

        Examples
        --------
        >>> import odl
        >>> r2 = odl.rn(2)
        >>> x = r2.element([1, 1])
        >>> op = DistOperator(x)
        >>> op([4, 5])
        5.0
        """
        self.__vector = vector
        super().__init__(vector.space, RealNumbers(), linear=False)

    @property
    def vector(self):
        """Element to which to take the distance."""
        return self.__vector

    def _call(self, x):
        """Return the distance from ``self.vector`` to ``x``."""
        return self.vector.dist(x)

    def derivative(self, point):
        """The derivative operator.

            ``DistOperator(y).derivative(z)(x) ==
            ((y - z) / y.dist(z)).inner(x)``

        This is only applicable in inner product spaces.

        Parameters
        ----------
        x : `domain` `element-like`
            Point in which to take the derivative.

        Returns
        -------
        derivative : `InnerProductOperator`

        Raises
        ------
        ValueError
            If ``point == self.vector``, in which case the derivative is not
            well defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::

            (D d(\cdot, y))(z)(x) = \\langle (y-z) / d(y, z), x \\rangle

        Examples
        --------
        >>> import odl
        >>> r2 = odl.rn(2)
        >>> x = r2.element([1, 1])
        >>> op = DistOperator(x)
        >>> derivative = op.derivative([2, 1])
        >>> derivative([1, 0])
        1.0
        """
        point = self.domain.element(point)
        diff = point - self.vector
        dist = self.vector.dist(point)
        if dist == 0:
            raise ValueError('not differentiable at the reference vector {!r}'
                             ''.format(self.vector))

        return InnerProductOperator(diff / dist)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return '{}({})'.format(self.__class__.__name__, self.vector)


class ConstantOperator(Operator):

    """Operator that always returns the same value.

        ``ConstantOperator(y)(x) == y``
    """

    def __init__(self, constant, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        constant : `LinearSpaceElement` or ``range`` `element-like`
            The constant space element to be returned. If ``range`` is not
            provided, ``constant`` must be a `LinearSpaceElement` since the
            operator range is then inferred from it.
        domain : `LinearSpace`, optional
            Domain of the operator. Default: ``vector.space``
        range : `LinearSpace`, optional
            Range of the operator. Default: ``vector.space``

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = ConstantOperator(x)
        >>> op(x, out=r3.element())
        rn(3).element([1.0, 2.0, 3.0])
        """

        if ((domain is None or range is None) and
                not isinstance(constant, LinearSpaceElement)):
                raise TypeError('If either domain or range is unspecified '
                                '`constant` must be LinearSpaceVector, got '
                                '{!r}.'.format(constant))

        if domain is None:
            domain = constant.space
        if range is None:
            range = constant.space

        super().__init__(domain, range)
        self.__constant = self.range.element(constant)

    @property
    def constant(self):
        """Constant space element returned by this operator."""
        return self.__constant

    def _call(self, x, out=None):
        """Return the constant vector or assign it to ``out``."""
        if out is None:
            return self.range.element(copy(self.constant))
        else:
            out.assign(self.constant)

    def derivative(self, point):
        """Derivative of this operator, always zero.

        Returns
        -------
        derivative : `ZeroOperator`

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = ConstantOperator(x)
        >>> deriv = op.derivative([1, 1, 1])
        >>> deriv([2, 2, 2])
        rn(3).element([0.0, 0.0, 0.0])
        """
        return ZeroOperator(domain=self.domain, range=self.range)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.constant)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}".format(self.constant)


class ZeroOperator(ConstantOperator):

    """Operator mapping each element to the zero element::

        ZeroOperator(space)(x) == space.zero()
    """

    def __init__(self, domain, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Domain of the operator.
        range : `LinearSpace`, optional
            Range of the operator. Default: ``domain``
        """
        if range is None:
            range = domain

        super().__init__(constant=range.zero(), domain=domain, range=range)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '0'


class ResidualOperator(Operator):

    """Operator that calculates the residual ``op(x) - y``.

        ``ResidualOperator(op, y)(x) == op(x) - y``
    """

    def __init__(self, operator, vector):
        """Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            Operator to be used in the residual expression. Its
            `Operator.range` must be a `LinearSpace`.
        vector : ``operator.range`` `element-like`
            Vector to be subtracted from the operator result.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> y = r3.element([1, 2, 3])
        >>> ident_op = odl.IdentityOperator(r3)
        >>> res_op = odl.ResidualOperator(ident_op, y)
        >>> x = r3.element([4, 5, 6])
        >>> res_op(x)
        rn(3).element([3.0, 3.0, 3.0])
        """
        if not isinstance(operator, Operator):
            raise TypeError('`op` {!r} not a Operator instance'
                            ''.format(operator))

        if not isinstance(operator.range, LinearSpace):
            raise TypeError('`op.range` {!r} not a LinearSpace instance'
                            ''.format(operator.range))

        self.__operator = operator
        self.__vector = operator.range.element(vector)
        super().__init__(operator.domain, operator.range)

    @property
    def operator(self):
        """The operator to apply."""
        return self.__operator

    @property
    def vector(self):
        """The constant operator range element to subtract."""
        return self.__vector

    def _call(self, x, out=None):
        """Evaluate the residual at ``x`` and write to ``out`` if given."""
        if out is None:
            out = self.operator(x)
        else:
            self.operator(x, out=out)

        out -= self.vector
        return out

    def derivative(self, point):
        """Derivative the residual operator.

        It is equal to the derivative of the "inner" operator:

            ``ResidualOperator(op, y).derivative(z) == op.derivative(z)``

        Parameters
        ----------
        point : `domain` element
            Any element in the domain where the derivative should be taken

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> op = IdentityOperator(r3)
        >>> res = ResidualOperator(op, r3.element([1, 2, 3]))
        >>> x = r3.element([4, 5, 6])
        >>> res.derivative(x)(x)
        rn(3).element([4.0, 5.0, 6.0])
        """
        return self.operator.derivative(point)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self.operator,
                                       self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{} - {}".format(self.op, self.vector)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
