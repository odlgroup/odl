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
from odl.space.pspace import ProductSpace
from odl.set.space import LinearSpace, LinearSpaceVector
from odl.set.sets import Field


__all__ = ('ScalingOperator', 'ZeroOperator', 'IdentityOperator',
           'LinCombOperator', 'MultiplyOperator', 'PowerOperator',
           'InnerProductOperator', 'ConstantOperator', 'ResidualOperator')


class ScalingOperator(Operator):

    """Operator of multiplication with a scalar."""

    def __init__(self, space, scalar):
        """Initialize a ScalingOperator instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space of elements which the operator is acting on
        scalar : `LinearSpace.field` `element`
            An element of the field of the space which vectors are
            scaled with
        """
        if not isinstance(space, LinearSpace):
            raise TypeError('`space` {!r} not a LinearSpace instance'
                            ''.format(space))

        super().__init__(space, space, linear=True)
        self._space = space
        self._scal = space.field.element(scalar)

    def _call(self, x, out=None):
        """Scale input and write to output.

        Parameters
        ----------
        x : ``domain`` `element`
            input vector to be scaled
        out : ``range`` `element`, optional
            Output vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
            Result of the scaling. If ``out`` was provided, the
            returned object is a reference to it.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> vec = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> op = ScalingOperator(r3, 2.0)
        >>> op(vec, out)  # In place, Returns out
        rn(3).element([2.0, 4.0, 6.0])
        >>> out
        rn(3).element([2.0, 4.0, 6.0])
        >>> op(vec)  # Out of place
        rn(3).element([2.0, 4.0, 6.0])
        """
        if out is None:
            out = self._scal * x
        else:
            out.lincomb(self._scal, x)
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
        if self._scal == 0.0:
            raise ZeroDivisionError('scaling operator not invertible for '
                                    'scalar==0')
        return ScalingOperator(self._space, 1.0 / self._scal)

    @property
    def adjoint(self):
        """Adjoint, given as scaling with the conjugate of the scalar."""
        if complex(self._scal).imag == 0.0:
            return self
        else:
            return ScalingOperator(self._space, self._scal.conjugate())

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ScalingOperator({!r}, {!r})'.format(self._space, self._scal)

    def __str__(self):
        """Return ``str(self)``."""
        return '{} * I'.format(self._scal)


class ZeroOperator(ScalingOperator):

    """Operator mapping each element to the zero element."""

    def __init__(self, space):
        """Initialize a ZeroOperator instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space of elements which the operator is acting on
        """
        super().__init__(space, 0)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ZeroOperator({!r})'.format(self._space)

    def __str__(self):
        """Return ``str(self)``."""
        return '0'


class IdentityOperator(ScalingOperator):

    """Operator mapping each element to itself."""

    def __init__(self, space):
        """Initialize an IdentityOperator instance.

        Parameters
        ----------
        space : LinearSpace
            The space of elements which the operator is acting on
        """
        super().__init__(space, 1)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'IdentityOperator({!r})'.format(self._space)

    def __str__(self):
        """Return ``str(self)``."""
        return "I"


class LinCombOperator(Operator):

    """Operator mapping two space elements to a linear combination.

    This opertor calculates:

    ``out = a*x[0] + b*x[1]``
    """

    def __init__(self, space, a, b):
        """Initialize a LinCombOperator instance.

        Parameters
        ----------
        space : `LinearSpace`
            The space of elements which the operator is acting on
        a, b : scalar
            Scalars to multiply ``x[0]`` and ``x[1]`` with, respectively
        """
        domain = ProductSpace(space, space)
        super().__init__(domain, space, linear=True)
        self.a = a
        self.b = b

    def _call(self, x, out=None):
        """Linearly combine the input and write to output.

        Parameters
        ----------
        x : ``domain`` `element`
            An element of the operator domain (2-tuple of space
            elements) whose linear combination is calculated
        out : ```range`` `element`
            Vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
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

    def __init__(self, y, domain=None, range=None):
        """Initialize a MultiplyOperator instance.

        Parameters
        ----------
        y : `LinearSpaceVector`
            The value to multiply by
        domain : `LinearSpace` or `Field`, optional
            The set to take values in. Default: ``x.space``
        """
        if domain is None:
            domain = y.space

        if range is None:
            range = y.space

        self.y = y
        self._domain_is_field = isinstance(domain, Field)
        self._range_is_field = isinstance(range, Field)
        super().__init__(domain, range, linear=True)

    def _call(self, x, out=None):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``domain`` `element`
            An element in the operator domain whose elementwise product is
            calculated.
        out : ``range`` `element`, optional
            Vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
            Result of the multiplication. If ``out`` was provided, the
            returned object is a reference to it.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])

        Multiply by vector

        >>> op = MultiplyOperator(x)
        >>> out = r3.element()
        >>> op(x, out)
        rn(3).element([1.0, 4.0, 9.0])

        Multiply by scalar

        >>> op2 = MultiplyOperator(x, domain=r3.field)
        >>> out = r3.element()
        >>> op2(3, out)
        rn(3).element([3.0, 6.0, 9.0])
        """
        if out is None:
            return x * self.y
        elif not self._range_is_field:
            if self._domain_is_field:
                out.lincomb(x, self.y)
            else:
                out.multiply(x, self.y)
        else:
            raise ValueError('can only use `out` with `LinearSpace` range')

    @property
    def adjoint(self):
        """The adjoint operator.

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
        if self._domain_is_field:
            return InnerProductOperator(self.y)
        else:
            # TODO: complex case
            return MultiplyOperator(self.y)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'MultiplyOperator({!r})'.format(self.y)

    def __str__(self):
        """Return ``str(self)``."""
        return "x * {}".format(self.y)


class PowerOperator(Operator):

    """The power of a vector or scalar.

    ``MultiplyOperator(n)(x) <==> x ** n``

    Here, ``x`` is a `LinearSpaceVector` or `Field` element and
    ``y`` is a number.
    Hence, this operator can be defined either on a `LinearSpace` or on
    a `Field`.
    """

    def __init__(self, domain, exponent):
        """Initialize a PowerOperator instance.

        Parameters
        ----------
        exponent : Number
            The power to take
        domain : `LinearSpace` or `Field`, optional
            The set to take values in
        """

        self.exponent = float(exponent)
        self._domain_is_field = isinstance(domain, Field)
        super().__init__(domain, domain, linear=(exponent == 1))

    def _call(self, x, out=None):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``domain`` `element`
            An element in the operator domain (2-tuple of space
            elements) whose elementwise product is calculated
        out : ``range`` `element`, optional
            Vector to which the result is written

        Returns
        -------
        out : ``range`` `element`
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
        elif self._domain_is_field:
            raise ValueError('cannot use `out` with field')
        else:
            out.assign(x)
            out **= self.exponent

    def derivative(self, point):
        """The derivative operator.

        ``MultiplyOperator(n).derivative(x)(y) <==> n * x ** (n - 1) * y``

        Parameters
        ----------
        point : ``domain`` `element`
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
    """

    def __init__(self, vector):
        """Initialize a InnerProductOperator instance.

        Parameters
        ----------
        vector : `LinearSpaceVector`
            The vector to take the inner product with
        """
        self.vector = vector
        super().__init__(vector.space, vector.space.field, linear=True)

    def _call(self, x):
        """Multiply the input and write to output.

        Parameters
        ----------
        x : ``vector.space`` `element`
            An element in the space of the vector

        Returns
        -------
        out : ``field`` `element`
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
        """The adjoint operator.

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
        """The vector of this operator.

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


class ConstantOperator(Operator):

    """Operator that always returns the same value

    ``ConstantOperator(vector)(x) <==> vector``
    """

    def __init__(self, vector, domain=None):
        """Initialize an instance.

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

        self.vector = vector
        super().__init__(domain, vector.space)

    def _call(self, x, out=None):
        """Return the constant vector or assign it to ``out``.

        Parameters
        ----------
        x : ``domain`` `element`
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

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ConstantOperator({!r})'.format(self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{}".format(self.vector)


class ResidualOperator(Operator):

    """Operator that calculates the residual ``op(x) - vec``.

    ``ResidualOperator(op, vector)(x) <==> op(x) - vec``
    """

    def __init__(self, op, vec):
        """Initialize a new instance.

        Parameters
        ----------
        op : `Operator`
            Operator to be used in the residual expression. Its
            `Operator.range` must be a `LinearSpace`.
        vec : `Operator.range` `element-like`
            Vector to be subtracted from the operator result
        """
        if not isinstance(op, Operator):
            raise TypeError('`op` {!r} not a Operator instance'
                            ''.format(op))

        if not isinstance(op.range, LinearSpace):
            raise TypeError('`op.range` {!r} not a LinearSpace instance'
                            ''.format(op.range))

        self.op = op
        self.vector = op.range.element(vec)
        super().__init__(op.domain, op.range)

    def _call(self, x, out=None):
        """Evaluate the residual at ``x``.

        Parameters
        ----------
        x : ``domain`` `element`
            Any element of the domain
        out : ``range`` `element`
            Vector that gets assigned to the constant vector

        Returns
        -------
        out : ``range`` `element`
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
            out = self.op(x)
        else:
            self.op(x, out=out)

        out -= self.vector
        return out

    def derivative(self, point):
        """The derivative the residual operator.

        It is equal to the derivative of the "inner" operator:

        ``ResidualOperator(op, vec).derivative(x) <==> op.derivative(x)``

        Parameters
        ----------
        x : ``domain`` `element`
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
        return self.op.derivative(point)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ResidualOperator({!r}, {!r})'.format(self.op, self.vector)

    def __str__(self):
        """Return ``str(self)``."""
        return "{} - {}".format(self.op, self.vector)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
