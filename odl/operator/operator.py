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

"""Abstract mathematical (linear) operators.

Operators are in the most general sense mappings from one set (`Set`)
to another. More common and useful are operators mapping a vector
space (`LinearSpace`) into another. Many of those are linear, and
as such, they have additional properties. See the class documentation
for further details.
In addition, this module defines classes for sums, compositions and
further combinations of operators of operators.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object, super
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta
from numbers import Number

# ODL imports
from odl.set.space import LinearSpace, UniversalSpace
from odl.set.sets import Set, UniversalSet

__all__ = ('Operator', 'OperatorComp', 'OperatorSum', 'OperatorLeftScalarMult',
           'OperatorRightScalarMult', 'OperatorPointwiseProduct')


def _bound_method(function):
    """Add a `self` argument to a function.

    This way, the decorated function may be used as a bound method.
    """
    def method(_, *args, **kwargs):
        return function(*args, **kwargs)

    # Do the minimum and copy function name and docstring
    if hasattr(function, '__name__'):
        method.__name__ = function.__name__
    if hasattr(function, '__doc__'):
        method.__doc__ = function.__doc__

    return method


def _default_call(self, x, *args, **kwargs):
    """Default out-of-place operator evaluation using `_apply()`.

    Parameters
    ----------
    x : domain element
        An object in the operator domain. The operator is applied
        to it.

    Returns
    -------
    out : range element
        An object in the operator range. The result of an operator
        evaluation.
    """
    out = self.range.element()
    self._apply(x, out, *args, **kwargs)
    return out


def _default_apply(self, x, out, *args, **kwargs):
    """Default in-place operator evaluation using `_call()`.

    Parameters
    ----------
    x : domain element
        An object in the operator domain. The operator is applied
        to it.

    out : range element
        An object in the operator range. The result of an operator
        evaluation.

    Returns
    -------
    None
    """
    out.assign(self._call(x, *args, **kwargs))


class _OperatorMeta(ABCMeta):

    """Metaclass used by Operator to ensure correct methods.

    If either `_apply` or `_call` does not exist in the class to be
    created, this metaclass attempts to add a default implmentation.
    This only works if the `range` is a `LinearSpace`.
    """

    def __new__(mcs, name, bases, attrs):
        """Create a new `_OperatorMeta` instance."""
        if '_call' in attrs and '_apply' in attrs:
            pass
        elif '_call' in attrs:
            attrs['_apply'] = _default_apply
        elif '_apply' in attrs:
            attrs['_call'] = _default_call

        return super().__new__(mcs, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        """Create a new class `cls` from given arguments."""
        obj = ABCMeta.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'domain'):
            raise NotImplementedError('`Operator` instances must have a '
                                      '`domain` attribute.')
        if not hasattr(obj, 'range'):
            raise NotImplementedError('`Operator` instances must have a '
                                      '`range` attribute.')
        if not hasattr(obj, '_call') and not hasattr(obj, '_apply'):
            raise NotImplementedError('`Operator` instances must either '
                                      'have `_call` or `_apply` as '
                                      'attribute.')
        return obj


class Operator(with_metaclass(_OperatorMeta, object)):

    """Abstract operator.

    Abstract attributes and methods
    -------------------------------
    `Operator` is an **abstract** class, i.e. it can only be
    subclassed, not used directly.

    **Any subclass of `Operator` must have the following attributes:**

    domain : `Set`
        The set of elements this operator can be applied to

    range : `Set`
        The set this operator maps to

    It is **highly** recommended to call `super().__init__(dom, ran)` in
    the `__init__()` method of any subclass, where `dom` and `ran` are
    the arguments specifying domain and range of the new operator. In
    that case, the attributes `domain` and `range` are automatically
    provided by `Operator`.

    In addition, **any subclass needs to implement at least one of the
    methods `_call()` and `_apply()`.**
    These are explained in the following.

    Out-of-place evaluation: `_call()`
    ----------------------------------
    Out-of-place evaluation means that the operator is applied,
    and the result is written to a new element which is returned.
    In this case, a subclass has to implement the method

    `_call(self, x)  <==>  operator(x)`

    **Parameters:**

    x : `domain` element
        An object in the operator domain to which the operator is
        applied.

    **Returns:**

    out : `range` element
        An object in the operator range, the result of the operator
        evaluation.

    In-place evaluation: `_apply()`
    -------------------------------
    In-place evaluation means that the operator is applied, and the
    result is written to an existing element provided as an additional
    argument. In this case, a subclass has to implement the method

    `_apply(self, x, out)  <==>  out <-- operator(x)`

    **Parameters:**

    x : `domain` element
        An object in the operator domain to which the operator is
        applied.

    out : `range` element
        An object in the operator range to which the result of the
        operator evaluation is written.

    **Returns:**

    None

    Notes
    -----
    If not both `_apply()` and `_call()` are implemented and the
    `range` is a `LinearSpace`, a default implementation of the
    respective other is provided.
    """

    def __init__(self, domain, range, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Set`
            The domain of this operator, i.e., the set of elements to
            which this operator can be applied

        ran : `Set`
            The range of this operator, i.e., the set this operator
            maps to
        """
        if not isinstance(domain, Set):
            raise TypeError('domain {!r} not a `Set` instance.'.format(domain))
        if not isinstance(range, Set):
            raise TypeError('range {!r} not a `Set` instance.'.format(range))

        self._domain = domain
        self._range = range
        self._is_linear = bool(linear)

        if self.is_linear:
            if not isinstance(domain, LinearSpace):
                raise TypeError('domain {!r} not a `LinearSpace` instance.'
                                ''.format(domain))
            if not isinstance(range, LinearSpace):
                raise TypeError('range {!r} not a `LinearSpace` instance.'
                                ''.format(range))

    @property
    def domain(self):
        """The domain of this operator."""
        return self._domain

    @property
    def range(self):
        """The range of this operator."""
        return self._range

    @property
    def is_linear(self):
        """True if this operator is linear."""
        return self._is_linear

    @property
    def adjoint(self):
        """The operator adjoint."""
        raise NotImplementedError('adjoint not implemented for operator {!r}.'
                                  ''.format(self))

    @property
    def T(self):
        """Shorthand for `adjoint`."""
        return self.adjoint

    def derivative(self, point):
        """Return the operator derivative at `point`."""
        if self.is_linear:
            return self
        else:
            raise NotImplementedError('derivative not implemented for operator'
                                      ' {!r}'.format(self))

    @property
    def inverse(self):
        """Return the operator inverse."""
        raise NotImplementedError('inverse not implemented for operator '
                                  '{!r}'.format(self))

    @property
    def I(self):
        """Shorthand for `inverse`."""
        return self.inverse

    # Implicitly defined operators
    def __call__(self, x, out=None, *args, **kwargs):
        """`op.__call__(x) <==> op(x)`.

        Implementation of the call pattern `op(x)` with the private
        `_call()` method and added error checking.

        Parameters
        ----------
        x : domain element
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.
        out : `range` element, optional
            An object in the operator range to which the result of the
            operator evaluation is written. The result is independent
            of the initial state of this object.
        *args, **kwargs : Further arguments to the function, optional

        Returns
        -------
        elem : range element
            An object in the operator range, the result of the operator
            evaluation. It is identical to `out` if provided.

        Examples
        --------
        >>> from odl import Rn, ScalingOperator
        >>> rn = Rn(3)
        >>> op = ScalingOperator(rn, 2.0)
        >>> x = rn.element([1, 2, 3])

        Out-of-place evaluation:

        >>> op(x)
        Rn(3).element([2.0, 4.0, 6.0])

        In-place evaluation:

        >>> y = rn.element()
        >>> op(x, out=y)
        Rn(3).element([2.0, 4.0, 6.0])
        >>> y
        Rn(3).element([2.0, 4.0, 6.0])
        """
        if x not in self.domain:
            raise TypeError('input {!r} not an element of the domain {!r} '
                            'of {!r}.'
                            ''.format(x, self.domain, self))

        if out is not None:  # In-place evaluation
            if out not in self.range:
                raise TypeError('output {!r} not an element of the range {!r} '
                                'of {!r}.'
                                ''.format(out, self.range, self))

            self._apply(x, out, *args, **kwargs)
            return out

        else:  # Out-of-place evaluation
            result = self._call(x, *args, **kwargs)

            if result not in self.range:
                raise TypeError('result {!r} not an element of the range {!r} '
                                'of {!r}.'
                                ''.format(result, self.range, self))
            return result

    def __add__(self, other):
        """`op.__add__(other) <==> op + other`."""
        return OperatorSum(self, other)

    def __sub__(self, other):
        """`op.__add__(other) <==> op - other`."""
        return OperatorSum(self, -1 * other)

    def __mul__(self, other):
        """`op.__mul__(other) <==> op * other`.

        If `other` is an operator, this corresponds to the pointwise
        operator product:

        `op1 * op2 <==> (x --> (op1(x) * op2(x)))`

        If `other` is a scalar, this corresponds to right
        multiplication of scalars with operators:

        `op * scalar <==> (x --> op(scalar * x))`

        Note that left and right multiplications are usually different.

        Parameters
        ----------
        other : `Operator` or scalar
            If `other` is an `Operator`, their `domain` and `range`
            must be equal, and `range` must be an `Algebra`.

            If `other` is a scalar and `self.domain` is a
            `LinearSpace`, `scalar` must be an element of
            `self.domain.field`.

        Returns
        -------
        mul : `Operator`
            The multiplication operator. If `other` is a scalar, a
            `OperatorRightScalarMult` is returned. If `other` is
            an operator, an `OperatorPointwiseProduct` is returned.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> op(x)
        Rn(3).element([1.0, 2.0, 3.0])
        >>> Scaled = op * 3
        >>> Scaled(x)
        Rn(3).element([3.0, 6.0, 9.0])
        """
        if isinstance(other, Operator):
            return OperatorComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear operator.
            if self.is_linear:
                return OperatorLeftScalarMult(self, other)
            else:
                return OperatorRightScalarMult(self, other)
        else:
            raise TypeError('multiplicant {!r} is neither operator nor '
                            'scalar.'.format(other))

    def __rmul__(self, other):
        """`op.__rmul__(s) <==> s * op`.

        If `other` is an operator, this corresponds to the pointwise
        operator product:

        `op1 * op2 <==> (x --> (op1(x) * op2(x)))`

        If `other` is a scalar, this corresponds to left
        multiplication of scalars with operators:

        `op * scalar <==> (x --> scalar * op(x))`

        Note that left and right multiplications are usually different.

        Parameters
        ----------
        other : `Operator` or scalar
            If `other` is an `Operator`, their `domain` and `range`
            must be equal, and `range` must be an `Algebra`.

            If `other` is a scalar and `self.range` is a
            `LinearSpace`, `scalar` must be an element of
            `self.range.field`.

        Returns
        -------
        rmul : `Operator`
            The multiplication operator. If `other` is a scalar, a
            `OperatorLeftScalarMult` is returned. If `other` is
            an operator, an `OperatorPointwiseProduct` is returned.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> op(x)
        Rn(3).element([1.0, 2.0, 3.0])
        >>> Scaled = 3 * op
        >>> Scaled(x)
        Rn(3).element([3.0, 6.0, 9.0])
        """
        if isinstance(other, Operator):
            return OperatorComp(other, self)
        elif isinstance(other, Number):
            return OperatorLeftScalarMult(self, other)
        else:
            raise TypeError('multiplicant {!r} is neither operator nor '
                            'scalar.'.format(other))

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`.

        The default `repr` implementation. Should be overridden by
        subclasses.
        """
        return '{}: {!r} -> {!r}'.format(self.__class__.__name__, self.domain,
                                         self.range)

    def __str__(self):
        """`op.__str__() <==> str(op)`.

        The default `str` implementation. Should be overridden by
        subclasses.
        """
        return self.__class__.__name__


class OperatorSum(Operator):

    """Expression type for the sum of operators.

    `OperatorSum(op1, op2) <==> (x --> op1(x) + op2(x))`

    The sum is only well-defined for `Operator` instances where
    `range` is a `LinearSpace`.

    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp_ran=None, tmp_dom=None):
        """Initialize a new `OperatorSum` instance.

        Parameters
        ----------
        op1 : `Operator`
            The first summand. Its `range` must be a `LinearSpace`.
        op2 : `Operator`
            The second summand. Must have the same `domain` and `range` as
            `op1`.
        tmp_ran : range element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        tmp_dom : domain element, optional
            Used to avoid the creation of a temporary when applying the
            operator adjoint.
        """
        if op1.range != op2.range:
            raise TypeError('operator ranges {!r} and {!r} do not match.'
                            ''.format(op1.range, op2.range))

        if not isinstance(op1.range, LinearSpace):
            raise TypeError('range {!r} not a `LinearSpace` instance.'
                            ''.format(op1.range))

        if op1.domain != op2.domain:
            raise TypeError('operator domains {!r} and {!r} do not match.'
                            ''.format(op1.domain, op2.domain))

        if tmp_ran is not None and tmp_ran not in op1.range:
            raise TypeError('tmp_ran {!r} not an element of the operator '
                            'range {!r}.'.format(tmp_ran, op1.range))

        if tmp_dom is not None and tmp_dom not in op1.domain:
            raise TypeError('tmp_dom {!r} not an element of the operator '
                            'domain {!r}.'.format(tmp_dom, op1.domain))

        super().__init__(op1.domain, op1.range,
                         linear=op1.is_linear and op2.is_linear)
        self._op1 = op1
        self._op2 = op2
        self._tmp_ran = tmp_ran
        self._tmp_dom = tmp_dom

    def _apply(self, x, out):
        """`op._apply(x, out) <==> out <-- op(x)`.

        Examples
        --------
        >>> from odl import Rn, IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> x = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> OperatorSum(op, op)(x, out)
        Rn(3).element([2.0, 4.0, 6.0])
        >>> out
        Rn(3).element([2.0, 4.0, 6.0])
        """
        # pylint: disable=protected-access
        tmp = self._tmp_ran if self._tmp_ran is not None else self.range.element()
        self._op1._apply(x, out)
        self._op2._apply(x, tmp)
        out += tmp

    def _call(self, x):
        """`op.__call__(x) <==> op(x)`.

        Examples
        --------
        >>> from odl import Rn, ScalingOperator
        >>> r3 = Rn(3)
        >>> A = ScalingOperator(r3, 3.0)
        >>> B = ScalingOperator(r3, -1.0)
        >>> C = OperatorSum(A, B)
        >>> C(r3.element([1, 2, 3]))
        Rn(3).element([2.0, 4.0, 6.0])
        """
        # pylint: disable=protected-access
        return self._op1._call(x) + self._op2._call(x)

    def derivative(self, x):
        """Return the operator derivative at `x`.

        # TODO: finish doc

        The derivative of a sum of two operators is equal to the sum of
        the derivatives.
        """
        return OperatorSum(self._op1.derivative(x), self._op2.derivative(x))

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator sum is the sum of the operator
        adjoints:

        `OperatorSum(op1, op2).adjoint ==
        `OperatorSum(op1.adjoint, op2.adjoint)`
        """
        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorSum(self._op1.adjoint, self._op2.adjoint,
                           self._tmp_dom, self._tmp_ran)

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op1, self._op2)

    def __str__(self):
        """`op.__str__() <==> str(op)`."""
        return '{} + {}'.format(self._op1, self._op2)


class OperatorComp(Operator):

    """Expression type for the composition of operators.

    `OperatorComp(left, right) <==> (x --> left(right(x)))`

    The composition is only well-defined if
    `left.domain == right.range`.
    """

    def __init__(self, left, right, tmp=None):
        """Initialize a new `OperatorComp` instance.

        Parameters
        ----------
        left : `Operator`
            The left ("outer") operator
        right : `Operator`
            The right ("inner") operator. Its range must coincide with the
            domain of `left`.
        tmp : element of the range of `right`, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if right.range != left.domain:
            raise TypeError('range {!r} of the right operator {!r} not equal '
                            'to the domain {!r} of the left operator {!r}.'
                            ''.format(right.range, right,
                                      left.domain, left))

        if tmp is not None and tmp not in left.domain:
            raise TypeError('temporary {!r} not an element of the left '
                            'operator domain {!r}.'.format(tmp, left.domain))

        super().__init__(right.domain, left.range,
                         linear=left.is_linear and right.is_linear)
        self._left = left
        self._right = right
        self._tmp = tmp

    def _apply(self, x, out):
        """`op._apply(x, out) <==> out <-- op(x)`."""
        # pylint: disable=protected-access
        tmp = (self._tmp if self._tmp is not None
               else self._right.range.element())
        self._right._apply(x, tmp)
        self._left._apply(tmp, out)

    def _call(self, x):
        """`op.__call__(x) <==> op(x)`."""
        # pylint: disable=protected-access
        return self._left._call(self._right._call(x))

    @property
    def inverse(self):
        """The operator inverse.

        The inverse of the operator composition is the composition of
        the inverses in reverse order:

        `OperatorComp(left, right).inverse ==
        OperatorComp(right.inverse, left.inverse)`
        """
        return OperatorComp(self._right.inverse, self._left.inverse, self._tmp)

    def derivative(self, point):
        """Return the operator derivative.

        The derivative of the operator composition follows the chain
        rule:

        `OperatorComp(left, right).derivative(point) ==
        OperatorComp(left.derivative(right(point)),
        right.derivative(point))`
        """
        left_deriv = self._left.derivative(self._right(point))
        right_deriv = self._right.derivative(point)

        return OperatorComp(left_deriv, right_deriv)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator composition is the composition of
        the operator adjoints in reverse order:

        `OperatorComp(left, right).adjoint ==
        `OperatorComp(right.adjoint, left.adjoint)`
        """
        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorComp(self._right.adjoint, self._left.adjoint,
                            self._tmp)

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._left, self._right)

    def __str__(self):
        """`op.__str__() <==> str(op)`."""
        return '{} o {}'.format(self._left, self._right)


class OperatorPointwiseProduct(Operator):

    """Expression type for the pointwise operator mulitplication.

    `OperatorPointwiseProduct(op1, op2) <==> (x --> op1(x) * op2(x))`

    The product is only well-defined for `Operator` instances where
    `range` is an `Algebra`.
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2):
        """Initialize a new instance.

        Parameters
        ----------
        op1 : `Operator`
            The first factor
        op2 : `Operator`
            The second factor. Must have the same domain and range as
            `op1`.
        """
        if op1.range != op2.range:
            raise TypeError('operator ranges {!r} and {!r} do not match.'
                            ''.format(op1.range, op2.range))

        if not isinstance(op1.range, LinearSpace):
            raise TypeError('range {!r} not a `LinearSpace` instance.'
                            ''.format(op1.range))

        if op1.domain != op2.domain:
            raise TypeError('operator domains {!r} and {!r} do not match.'
                            ''.format(op1.domain, op2.domain))

        super().__init__(op1.domain, op1.range, linear=False)
        self._op1 = op1
        self._op2 = op2

    def _apply(self, x, out):
        """`op._apply(x, out) <==> out <-- op(x)`."""
        # pylint: disable=protected-access
        tmp = self._op2.range.element()
        self._op1._apply(x, out)
        self._op2._apply(x, tmp)
        out *= tmp

    def _call(self, x):
        """`op.__call__(x) <==> op(x)`."""
        # pylint: disable=protected-access
        return self._op1._call(x) * self._op2._call(x)

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op1, self._op2)

    def __str__(self):
        """`op.__str__() <==> str(op)`."""
        return '{} * {}'.format(self._op1, self._op2)


class OperatorLeftScalarMult(Operator):

    """Expression type for the operator left scalar multiplication.

    `OperatorLeftScalarMult(op, scalar) <==> (x --> scalar * op(x))`

    The scalar multiplication is well-defined only if `op.range` is
    a `LinearSpace`.
    """

    def __init__(self, op, scalar):
        """Initialize a new `OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : `Operator`
            The range of `op` must be a `LinearSpace`.
        scalar : `op.range.field` element
            A real or complex number, depending on the field of
            the range.
        """
        if not isinstance(op.range, LinearSpace):
            raise TypeError('range {!r} not a `LinearSpace` instance.'
                            ''.format(op.range))

        if scalar not in op.range.field:
            raise TypeError('scalar {!r} not in the field {!r} of the '
                            'operator range {!r}.'
                            ''.format(scalar, op.range.field, op.range))

        super().__init__(op.domain, op.range, linear=op.is_linear)
        self._op = op
        self._scalar = scalar

    def _call(self, x):
        """`op.__call__(x) <==> op(x)`."""
        # pylint: disable=protected-access
        return self._scalar * self._op._call(x)

    def _apply(self, x, out):
        """`op._apply(x, out) <==> out <-- op(x)`."""
        # pylint: disable=protected-access
        self._op._apply(x, out)
        out *= self._scalar

    @property
    def inverse(self):
        """The inverse operator.

        The inverse of `scalar * op` is given by
        `op.inverse * 1/scalar` if `scalar != 0`. If `scalar == 0`,
        the inverse is not defined.

        OperatorLeftScalarMult(op, scalar).inverse <==>
        OperatorRightScalarMult(op.inverse, 1.0/scalar)
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('{} not invertible.'.format(self))
        return OperatorLeftScalarMult(self._op.inverse, 1.0/self._scalar)

    def derivative(self, x):
        """Return the derivative at 'x'.

        Left scalar multiplication and derivative are commutative:

        OperatorLeftScalarMult(op, scalar).derivative(x) <==>
        OperatorLeftScalarMult(op.derivative(x), scalar)

        See also
        --------
        OperatorLeftScalarMult : the result
        """
        return OperatorLeftScalarMult(self._op.derivative(x), self._scalar)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        `OperatorLeftScalarMult(op, scalar).adjoint ==
        `OperatorLeftScalarMult(op.adjoint, scalar)`
        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorRightScalarMult(self._op.adjoint, self._scalar.conjugate())

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._scalar)

    def __str__(self):
        """`op.__str__() <==> str(op)`."""
        return '{} * {}'.format(self._scalar, self._op)


class OperatorRightScalarMult(Operator):

    """Expression type for the operator right scalar multiplication.

    OperatorRightScalarMult(op, scalar) <==> (x --> op(scalar * x))

    The scalar multiplication is well-defined only if `op.domain` is
    a `LinearSpace`.
    """

    def __init__(self, op, scalar, tmp=None):
        """Initialize a new `OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : `Operator`
            The domain of `op` must be a `LinearSpace`.
        scalar : `op.range.field` element
            A real or complex number, depending on the field of
            the operator domain.
        tmp : domain element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if not isinstance(op.domain, LinearSpace):
            raise TypeError('domain {!r} not a `LinearSpace` instance.'
                            ''.format(op.domain))

        if scalar not in op.domain.field:
            raise TypeError('scalar {!r} not in the field {!r} of the '
                            'operator domain {!r}.'
                            ''.format(scalar, op.domain.field, op.domain))

        if tmp is not None and tmp not in op.domain:
            raise TypeError('temporary {!r} not an element of the '
                            'operator domain {!r}.'.format(tmp, op.domain))

        super().__init__(op.domain, op.range, op.is_linear)
        self._op = op
        self._scalar = scalar
        self._tmp = tmp

    def _call(self, x):
        """`op.__call__(x) <==> op(x)`."""
        # pylint: disable=protected-access
        return self._op._call(self._scalar * x)

    def _apply(self, x, out):
        """`op._apply(x, out) <==> out <-- op(x)`."""
        # pylint: disable=protected-access
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, x)
        self._op._apply(tmp, out)

    @property
    def inverse(self):
        """The inverse operator.

        The inverse of `op * scalar` is given by
        `1/scalar * op.inverse` if `scalar != 0`. If `scalar == 0`,
        the inverse is not defined.

        `OperatorRightScalarMult(op, scalar).inverse <==>
        OperatorLeftScalarMult(op.inverse, 1.0/scalar)`
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('{} not invertible.'.format(self))

        return OperatorLeftScalarMult(self._op.inverse, 1.0/self._scalar)

    def derivative(self, x):
        """Return the derivative at 'x'.

        The derivative of the right scalar operator multiplication
        follows the chain rule:

        `OperatorRightScalarMult(op, scalar).derivative(x) <==>
        OperatorLeftScalarMult(op.derivative(scalar * x),
        scalar)`
        """
        return OperatorLeftScalarMult(self._op.derivative(self._scalar * x),
                                      self._scalar)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        `OperatorLeftScalarMult(op, scalar).adjoint ==
        `OperatorLeftScalarMult(op.adjoint, scalar)`
        """

        if not self.is_linear:
            raise NotImplementedError('Nonlinear operators have no adjoint')

        return OperatorRightScalarMult(self._op.adjoint, self._scalar.conjugate())

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return '{}({!r}, {!r})'.format(self.__class__.__name__,
                                       self._op, self._scalar)

    def __str__(self):
        """`op.__str__() <==> str(op)`."""
        return '{} * {}'.format(self._op, self._scalar)


def operator(call=None, apply=None, inv=None, deriv=None,
             dom=None, ran=None, linear=False):
    """Create a simple operator.

    Mostly intended for simple prototyping rather than final use.

    Parameters
    ----------
    call : callable
        A function taking one argument and returning the result.
        It will be used for the operator call pattern
        `out = op(x)`.
    apply : callable
        A function taking two arguments.
        It will be used for the operator apply pattern
        `op._apply(x, out) <==> out <-- op(x)`. Return value
        is assumed to be `None` and is ignored.
    inv : `Operator`, optional
        The operator inverse
        Default: `None`
    deriv : `Operator`, optional
        The operator derivative, linear
        Default: `None`
    dom : `Set`, optional
        The domain of the operator
        Default: `UniversalSpace` if linear, else `UniversalSet`
    ran : `Set`, optional
        The range of the operator
        Default: `UniversalSpace` if linear, else `UniversalSet`
    linear : `boolean`, optional
        True if the operator is linear
        Default: False

    Returns
    -------
    op : `SimpleOperator`
        An operator with the provided attributes and methods.
        `SimpleOperator` is a subclass of `Operator`.

    Notes
    -----
    It suffices to supply one of the functions `call` and `apply`.
    If `dom` is a `LinearSpace`, a default implementation of the
    respective other method is automatically provided; if not, a
    `NotImplementedError` is raised when the other method is called.

    Examples
    --------
    >>> A = operator(lambda x: 3*x)
    >>> A(5)
    15
    """
    if call is None and apply is None:
        raise ValueError('at least one argument `call` or `apply` must be '
                         'given.')

    if linear:
        dom = ran = UniversalSpace()
    else:
        dom = ran = UniversalSet()

    attrs = {'inverse': inv, 'derivative': deriv, 'domain': dom, 'range': ran}

    if call is not None:
        attrs['_call'] = _bound_method(call)

    if apply is not None:
        attrs['_apply'] = _bound_method(apply)

    simple_operator = _OperatorMeta('SimpleOperator', (Operator,), attrs)
    return simple_operator(dom, ran, linear)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
