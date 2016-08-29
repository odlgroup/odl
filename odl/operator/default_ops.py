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

from odl.operator.operator import Operator
from odl.space import ProductSpace
from odl.set import LinearSpace, LinearSpaceVector, Field, RealNumbers


__all__ = ('ScalingOperator', 'ZeroOperator', 'IdentityOperator',
           'LinCombOperator', 'MultiplyOperator', 'PowerOperator',
           'InnerProductOperator', 'NormOperator', 'DistOperator',
           'ConstantOperator', 'ResidualOperator')


class ScalingOperator(Operator):

    """Operator of multiplication with a scalar."""

    def __init__(self, space, scalar):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements on which this operator acts.
        scalar : `LinearSpace.field` `element-like`
            Fixed scaling factor of this operator.
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
        """Scale input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            input vector to be scaled
        out : ``range`` element, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` element
            Result of the scaling. If ``out`` was provided, the
            returned object is a reference to it.

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
        return 'ScalingOperator({!r}, {!r})'.format(self.domain, self.scalar)

    def __str__(self):
        """Return ``str(self)``."""
        return '{} * I'.format(self.scalar)


class ZeroOperator(ScalingOperator):

    """Operator mapping each element to the zero element."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements which the operator is acting on.
        """
        super().__init__(space, 0)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ZeroOperator({!r})'.format(self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return '0'


class IdentityOperator(ScalingOperator):

    """Operator mapping each element to itself."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : LinearSpace
            Space of elements which the operator is acting on.
        """
        super().__init__(space, 1)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'IdentityOperator({!r})'.format(self.domain)

    def __str__(self):
        """Return ``str(self)``."""
        return "I"


class LinCombOperator(Operator):

    """Operator mapping two space elements to a linear combination.

    This opertor calculates:

    ``out = a*x[0] + b*x[1]``
    """

    def __init__(self, space, a, b):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space of elements which the operator is acting on.
        a, b : scalar
            Scalars to multiply ``x[0]`` and ``x[1]`` with, respectively.
        """
        domain = ProductSpace(space, space)
        super().__init__(domain, space, linear=True)
        self.a = a
        self.b = b

    def _call(self, x, out=None):
        """Linearly combine the input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            An element of the operator domain (2-tuple of space
            elements) whose linear combination is calculated
        out : ```range`` element
            Vector to which the result is written

        Returns
        -------
        out : ``range`` element
            Result of the linear combination. If ``out`` was provided,
            the returned object is a reference to it.

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
        if out is None:
            out = self.range.element()
        out.lincomb(self.a, x[0], self.b, x[1])
        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'LinCombOperator({!r}, {!r}, {!r})'.format(
            self.range, self.a, self.b)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}*x + {}*y".format(self.a, self.b)


class MultiplyOperator(Operator):

    """Operator multiplying two elements.

    ``MultiplyOperator(y)(x) <==> x * y``

    Here, ``x`` is a `LinearSpaceVector` or `Field` element and
    ``y`` is a `LinearSpaceVector`.
    Hence, this operator can be defined either on a `LinearSpace` or on
    a `Field`. In the first case it is the pointwise multiplication,
    in the second the scalar multiplication.
    """

    def __init__(self, multiplicand, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        multiplicand : `LinearSpaceVector` or `Number`
            Value to multiply by.
        domain : `LinearSpace` or `Field`, optional
            Set to take values in. Default: ``x.space``.
        range : `LinearSpace` or `Field`, optional
            Set to map to. Default: ``x.space``.
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
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            An element in the operator domain whose elementwise product is
            calculated.
        out : ``range`` element, optional
            Vector to which the result is written

        Returns
        -------
        out : ``range`` element
            Result of the multiplication. If ``out`` was provided, the
            returned object is a reference to it.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])

        Multiply by vector

        >>> op = MultiplyOperator(x)
        >>> op(x)
        rn(3).element([1.0, 4.0, 9.0])
        >>> out = r3.element()
        >>> op(x, out)
        rn(3).element([1.0, 4.0, 9.0])

        Multiply by scalar

        >>> op2 = MultiplyOperator(x, domain=r3.field)
        >>> op2(3)
        rn(3).element([3.0, 6.0, 9.0])
        >>> out = r3.element()
        >>> op2(3, out)
        rn(3).element([3.0, 6.0, 9.0])
        """
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
        adjoint : {`InnerProductOperator`, `MultiplyOperator`}
            If the domain of this operator is the scalar field of a
            `LinearSpace` the adjoint is the inner product with ``y``,
            else it is the multiplication with ``y``

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])

        Multiply by vector

        >>> op = MultiplyOperator(x)
        >>> out = r3.element()
        >>> op.adjoint(x)
        rn(3).element([1.0, 4.0, 9.0])

        Multiply by scalar

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
        return 'MultiplyOperator({!r})'.format(self.y)

    def __str__(self):
        """Return ``str(self)``."""
        return "x * {}".format(self.y)


class PowerOperator(Operator):

    """Power of a vector or scalar.

    ``PowerOperator(p)(x) <==> x ** p``

    Here, ``x`` is a `LinearSpaceVector` or `Field` element and ``p`` is
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
        """

        self.__exponent = float(exponent)
        self.__domain_is_field = isinstance(domain, Field)
        super().__init__(domain, domain, linear=(exponent == 1))

    @property
    def exponent(self):
        """Power of the input element to take."""
        return self.__exponent

    def _call(self, x, out=None):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``domain`` element
            An element in the operator domain (2-tuple of space
            elements) whose elementwise product is calculated
        out : ``range`` element, optional
            Vector to which the result is written

        Returns
        -------
        out : ``range`` element
            Result of the multiplication. If ``out`` was provided, the
            returned object is a reference to it.

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
        if out is None:
            return x ** self.exponent
        elif self.__domain_is_field:
            raise ValueError('cannot use `out` with field')
        else:
            out.assign(x)
            out **= self.exponent

    def derivative(self, point):
        """Derivative of this operator.

        ``MultiplyOperator(n).derivative(x)(y) <==> n * x ** (n - 1) * y``

        Parameters
        ----------
        point : ``domain`` element
            The point in which to take the derivative

        Returns
        -------
        derivative : `Operator`
            The derivative in ``point``

        Examples
        --------
        >>> import odl

        Use with vectors

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
    """Operator taking the inner product with a fixed vector.

    ``InnerProductOperator(vec)(x) <==> x.inner(vec)``

    This is only applicable in inner product spaces.

    See Also
    --------
    DistOperator : Distance to fixed vector.
    NormOperator : Norm of a vector.
    """

    def __init__(self, vector):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector to take the inner product with
        """
        self.__vector = vector
        super().__init__(vector.space, vector.space.field, linear=True)

    @property
    def vector(self):
        """Vector to take the inner product with."""
        return self.__vector

    def _call(self, x):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``vector.space`` element
            An element in the space of the vector

        Returns
        -------
        out : ``field`` element
            Result of the inner product calculation

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = InnerProductOperator(x)
        >>> op(r3.element([1, 2, 3]))
        14.0
        """
        return x.inner(self.vector)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `MultiplyOperator`
            The operator of multiplication with ``vector``.

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
        """Vector of this operator.

        Returns
        -------
        vector : `LinearSpaceVector`
            Vector used in this operator

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
        return 'InnerProductOperator({!r})'.format(self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}.T".format(self.vector)


class NormOperator(Operator):

    """Operator taking the norm of a vector.

    ``NormOperator(space)(x) <==> space.norm(x)``

    This is only applicable in normed spaces.

    See Also
    --------
    InnerProductOperator : Inner product with fixed vector.
    DistOperator : Distance to fixed vector.
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `LinearSpace`
            Space to take the norm in.
        """
        super().__init__(space, RealNumbers(), linear=False)

    def _call(self, x):
        """Return the norm of ``x``.

        Parameters
        ----------
        x : `domain` element
            Element to take the norm of.

        Returns
        -------
        norm : float
            Norm of ``x``.

        Examples
        --------
        >>> import odl
        >>> r2 = odl.rn(2)
        >>> op = NormOperator(r2)
        >>> op([3, 4])
        5.0
        """
        return x.norm()

    def derivative(self, point):
        """Derivative of this operator in ``point``.

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
            If ``point.norm() == 0``, in which case the derivative is not well
            defined in the Frechet sense.

        Notes
        -----
        The derivative cannot be written in a general sense except in Hilbert
        spaces, in which case it is given by

        .. math::

            (D ||.||)(x)(y) = < x / ||x||, y >

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

    """Operator taking the distance to a fixed vector.

    ``DistOperator(x)(y) <==> x.dist(y)``

    This is only applicable in metric spaces.

    See Also
    --------
    InnerProductOperator : Inner product with fixed vector.
    NormOperator : Norm of a vector.
    """

    def __init__(self, vector):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            Point to calculate the distance to.
        """
        self.vector = vector
        super().__init__(vector.space, RealNumbers(), linear=False)

    def _call(self, x):
        """Return the distance to x.

        Parameters
        ----------
        x : `domain` element
            An element in the domain.

        Returns
        -------
        out : float
            Distance from of ``x`` to ``self.vector``.

        Examples
        --------
        >>> import odl
        >>> r2 = odl.rn(2)
        >>> x = r2.element([1, 1])
        >>> op = DistOperator(x)
        >>> op([4, 5])
        5.0
        """
        return self.vector.dist(x)

    def derivative(self, point):
        """The derivative operator.

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

            (D d(x, y))(y)(z) = < (x-y) / d(x, y), y >

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

    """Operator that always returns the same value

    ``ConstantOperator(vector)(x) <==> vector``
    """

    def __init__(self, vector, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector constant to be returned

        domain : `LinearSpace`, default : vector.space
            The domain of the operator.
        """
        if not isinstance(vector, LinearSpaceVector):
            raise TypeError('`vector` {!r} not a LinearSpaceVector instance'
                            ''.format(vector))

        if domain is None:
            domain = vector.space

        self.__vector = vector
        super().__init__(domain, vector.space, linear=False)

    @property
    def vector(self):
        """Constant value."""
        return self.__vector

    def _call(self, x, out=None):
        """Return the constant vector or assign it to ``out``.

        Parameters
        ----------
        x : ``domain`` element
            An element of the domain
        out : ``range`` element
            Vector that gets assigned to the constant vector

        Returns
        -------
        out : ``range`` element
            Result of the assignment. If ``out`` was provided, the
            returned object is a reference to it.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> op = ConstantOperator(x)
        >>> op(x, out=r3.element())
        rn(3).element([1.0, 2.0, 3.0])
        """
        if out is None:
            return self.range.element(self.vector.copy())
        else:
            out.assign(self.vector)

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
        return ZeroOperator(self.domain)

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r})'.format(self.__class__.__name__, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}".format(self.vector)


class ResidualOperator(Operator):

    """Operator that calculates the residual ``op(x) - vec``.

    ``ResidualOperator(op, vector)(x) <==> op(x) - vec``
    """

    def __init__(self, operator, vector):
        """Initialize a new instance.

        Parameters
        ----------
        operator : `Operator`
            Operator to be used in the residual expression. Its
            `Operator.range` must be a `LinearSpace`.
        vector : `Operator.range` `element-like`
            Vector to be subtracted from the operator result.
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
        """The constant vector to subtract."""
        return self.__vector

    def _call(self, x, out=None):
        """Evaluate the residual at ``x``.

        Parameters
        ----------
        x : ``domain`` element
            Any element of the domain
        out : ``range`` element
            Vector that gets assigned to the constant vector

        Returns
        -------
        out : ``range`` element
            Result of the evaluation. If ``out`` was provided, the
            returned object is a reference to it.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> op = IdentityOperator(r3)
        >>> res = ResidualOperator(op, vec)
        >>> x = r3.element([4, 5, 6])
        >>> res(x, out=r3.element())
        rn(3).element([3.0, 3.0, 3.0])
        """
        if out is None:
            out = self.operator(x)
        else:
            self.operator(x, out=out)

        out -= self.vector
        return out

    def derivative(self, point):
        """Derivative the residual operator.

        It is equal to the derivative of the "inner" operator:

        ``ResidualOperator(op, vec).derivative(x) <==> op.derivative(x)``

        Parameters
        ----------
        x : ``domain`` element
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
        return 'ResidualOperator({!r}, {!r})'.format(self.operator,
                                                     self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{} - {}".format(self.op, self.vector)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
