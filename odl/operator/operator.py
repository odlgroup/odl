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
from odl.sets.space import LinearSpace
from odl.sets.set import UniversalSet

__all__ = ('Operator', 'OperatorComp', 'OperatorSum', 'OperatorLeftScalarMult',
           'OperatorRightScalarMult', 'OperatorPointwiseProduct',
           'LinearOperator', 'LinearOperatorComp', 'LinearOperatorSum',
           'LinearOperatorScalarMult')


class _DefaultCallOperator(object):

    """Decorator class that adds a `_call`  method to an `Operator`.

    This default implementation assumes that the operator implements
    `_apply()` and that the `range` of the operator implements
    `element()`. The latter is true for all vector spaces.
    """

    # pylint: disable=too-few-public-methods
    def _call(self, inp):
        """Apply the operator out-of-place using `_apply()`.

        Implemented as:

        `outp = self.range.element()`
        `self._apply(inp, outp)`
        `return outp`

        Parameters
        ----------

        inp : `domain` element
            An object in the operator domain. The operator is applied
            to it.

        Returns
        -------

        out : `range` element
            An object in the operator range. The result of an operator
            evaluation.
        """
        out = self.range.element()
        self._apply(inp, out)
        return out


class _DefaultApplyOperator(object):

    """Decorator class that adds an `_apply` method to `Operator`.

    The default implementation assumes that the operator implements
    `_call()` and that elements in the `range` of the operator
    implement `assign()`.
    """

    # pylint: disable=too-few-public-methods
    def _apply(self, inp, out):
        """Apply the operator in-place using `_call()`.

        Implemented as:

        `out.assign(self._call(inp))`

        inp : `self.domain` element
            An object in the operator domain. The operator is applied
            to it.

        out : `self.range` element
            An object in the operator range. The result of an operator
            evaluation.

        Returns
        -------
        None
        """
        out.assign(self._call(inp))


class _OperatorMeta(ABCMeta):

    """Metaclass used by Operator to ensure correct methods.

    If either `_apply` or `_call` does not exist in the class to be
    created, this metaclass attempts to add a default implmentation.
    This only works if the `range` is a `LinearSpace`.
    """

    def __new__(mcs, name, bases, attrs):
        """Create a new `_OperatorMeta` instance."""
        if '_call' in attrs and '_apply' in attrs:
            return super().__new__(mcs, name, bases, attrs)
        elif '_call' in attrs:
            return super().__new__(mcs, name, (_DefaultApplyOperator,) + bases,
                                   attrs)
        elif '_apply' in attrs:
            return super().__new__(mcs, name, (_DefaultCallOperator,) + bases,
                                   attrs)
        else:
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

    In addition, **any subclass needs to implement at least one of the
    methods `_call()` and `_apply()`.**
    These are explained in the following.

    Out-of-place evaluation: `_call()`
    ----------------------------------
    Out-of-place evaluation means that the operator is applied,
    and the result is written to a new element which is returned.
    In this case, a subclass has to implement the method

    `_call(self, inp)`     <==>     `operator(inp)`

    **Parameters:**

    inp : `domain` element
        An object in the operator domain to which the operator is
        applied.

    **Returns:**

    outp : `range` element
        An object in the operator range, the result of the operator
        evaluation.

    In-place evaluation: `_apply()`
    -------------------------------
    In-place evaluation means that the operator is applied, and the
    result is written to an existing element provided as an additional
    argument. In this case, a subclass has to implement the method

    `_apply(self, inp, outp)`     <==>     `outp <-- operator(inp)`

    **Parameters:**

    inp : `domain` element
        An object in the operator domain to which the operator is
        applied.

    outp : `range` element
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

    def derivative(self, point):
        """Return the operator derivative at `point`."""
        raise NotImplementedError('derivative not implemented for operator '
                                  '{!r}'.format(self))

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
    def apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        Implementation of in-place operator evaluation with the private
        `_apply()` method and added error checking.

        Parameters
        ----------
        inp : domain element
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.

        outp : range element
            An object in the operator range to which the result of the
            operator evaluation is written. The result is independent
            of the initial state of this object.

        Returns
        -------
        None

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element()
        >>> op.apply(x, y)
        >>> y
        Rn(3).element([1.0, 2.0, 3.0])
        """
        if inp not in self.domain:
            raise TypeError('input {!r} not an element of the domain {!r} '
                            'of {!r}.'
                            ''.format(inp, self.domain, self))

        if outp not in self.range:
            raise TypeError('output {!r} not an element of the range {!r} '
                            'of {!r}.'
                            ''.format(outp, self.range, self))

        if inp is outp:
            raise ValueError('aliased (identical) input and output not '
                             'allowed.')
        self._apply(inp, outp)

    def __call__(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        Implementation of the call pattern `op(inp)` with the private
        `_call()` method and added error checking.

        Parameters
        ----------
        inp : domain element
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.

        Returns
        -------
        elem : range element
            An object in the operator range, the result of the operator
            evaluation.

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> op(x)
        Rn(3).element([1.0, 2.0, 3.0])

        >>> from odl.operator.operator import operator
        >>> A = operator(lambda x: 3*x)
        >>> A(3)
        9
        >>> A.__call__(5)
        15
        """
        if inp not in self.domain:
            raise TypeError('input {!r} not an element of the domain {!r} '
                            'of {!r}.'
                            ''.format(inp, self.domain, self))

        result = self._call(inp)

        if result not in self.range:
            raise TypeError('result {!r} not an element of the range {!r} '
                            'of {!r}.'
                            ''.format(result, self.range, self))
        return result

    def __add__(self, other):
        """`op.__add__(other) <==> op + other`."""
        return OperatorSum(self, other)

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
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
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
            return OperatorPointwiseProduct(self, other)
        elif isinstance(other, Number):
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
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
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
            return OperatorPointwiseProduct(self, other)
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
        return '{}: {} -> {}'.format(self.__class__.__name__, self.domain,
                                     self.range)
        # return self.__class__.__name__


class OperatorSum(Operator):

    """Expression type for the sum of operators.

    `OperatorSum(op1, op2) <==> (x --> op1(x) + op2(x))`

    The sum is only well-defined for `Operator` instances where
    `range` is a `LinearSpace`.

    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp=None):
        """Initialize a new `OperatorSum` instance.

        Parameters
        ----------
        op1 : `Operator`
            The first summand. Its `range` must be a `LinearSpace`.
        op2 : `Operator`
            The second summand. Must have the same `domain` and `range` as
            `op1`.
        tmp : range element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
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

        if tmp is not None and tmp not in op1.domain:
            raise TypeError('temporary {!r} not an element of the operator '
                            'domain {!r}.'.format(tmp, op1.domain))

        self._op1 = op1
        self._op2 = op2
        self._tmp = tmp

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> inp = r3.element([1, 2, 3])
        >>> outp = r3.element()
        >>> OperatorSum(op, op).apply(inp, outp)
        >>> outp
        Rn(3).element([2.0, 4.0, 6.0])
        """
        # pylint: disable=protected-access
        tmp = self._tmp if self._tmp is not None else self.range.element()
        self._op1._apply(inp, outp)
        self._op2._apply(inp, tmp)
        outp += tmp

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import ScalingOperator
        >>> r3 = Rn(3)
        >>> A = ScalingOperator(r3, 3.0)
        >>> B = ScalingOperator(r3, -1.0)
        >>> C = OperatorSum(A, B)
        >>> C(r3.element([1, 2, 3]))
        Rn(3).element([2.0, 4.0, 6.0])
        """
        # pylint: disable=protected-access
        return self._op1._call(inp) + self._op2._call(inp)

    @property
    def domain(self):
        """The operator domain.

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> OperatorSum(op, op).domain
        Rn(3)
        """
        return self._op1.domain

    @property
    def range(self):
        """The operator range (or codomain).

        Examples
        --------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default_ops import IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> OperatorSum(op, op).range
        Rn(3)
        """
        return self._op1.range

    def derivative(self, point):
        """Return the operator derivative at `point`.

        # TODO: finish doc

        The derivative of a sum of two operators is equal to the sum of
        the derivatives.
        """
        return LinearOperatorSum(self._op1.derivative(point),
                                 self._op2.derivative(point))

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return 'OperatorSum({!r}, {!r})'.format(self._op1, self._op2)

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

        self._left = left
        self._right = right
        self._tmp = tmp

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`."""
        # pylint: disable=protected-access
        return self._left._call(self._right._call(inp))

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`."""
        # pylint: disable=protected-access
        tmp = (self._tmp if self._tmp is not None
               else self._right.range.element())
        self._right._apply(inp, tmp)
        self._left._apply(tmp, outp)

    @property
    def domain(self):
        """The operator `domain`.

        Corresponds to `right.domain`
        """
        return self._right.domain

    @property
    def range(self):
        """The operator `range`.

        Corresponds to `left.range`
        """
        return self._left.range

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
        LinearOperatorComp(left.derivative(right(point)),
        right.derivative(point))`
        """
        left_deriv = self._left.derivative(self._right(point))
        right_deriv = self._right.derivative(point)

        return LinearOperatorComp(left_deriv, right_deriv)

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return 'OperatorComp({!r}, {!r})'.format(self._left, self._right)

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

        self._op1 = op1
        self._op2 = op2

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`."""
        # pylint: disable=protected-access
        return self._op1._call(inp) * self._op2._call(inp)

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`."""
        # pylint: disable=protected-access
        tmp = self._op2.range.element()
        self._op1._apply(inp, outp)
        self._op2._apply(inp, tmp)
        outp *= tmp

    @property
    def domain(self):
        """The operator `domain`."""
        return self._op1.domain

    @property
    def range(self):
        """The operator `range`."""
        return self._op1.range

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return 'OperatorPointwiseProduct({!r}, {!r})'.format(self._op1,
                                                             self._op2)

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

        self._op = op
        self._scalar = scalar

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`."""
        # pylint: disable=protected-access
        return self._scalar * self._op._call(inp)

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`."""
        # pylint: disable=protected-access
        self._op._apply(inp, outp)
        outp *= self._scalar

    @property
    def domain(self):
        """The operator `domain`."""
        return self._op.domain

    @property
    def range(self):
        """The operator `range`."""
        return self._op.range

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
        return OperatorRightScalarMult(self._op.inverse, 1.0/self._scalar)

    def derivative(self, point):
        """Return the derivative at 'point'.

        Left scalar multiplication and derivative are commutative:

        OperatorLeftScalarMult(op, scalar).derivative(point) <==>
        LinearOperatorScalarMult(op.derivative(point), scalar)

        See also
        --------
        LinearOperatorScalarMult: the result
        """
        return LinearOperatorScalarMult(self._op.derivative(point),
                                        self._scalar)

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return 'OperatorLeftScalarMult({!r}, {!r})'.format(self._op,
                                                           self._scalar)

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

        self._op = op
        self._scalar = scalar
        self._tmp = tmp

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`."""
        # pylint: disable=protected-access
        return self._op._call(self._scalar * inp)

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`."""
        # pylint: disable=protected-access
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, inp)
        self._op._apply(tmp, outp)

    @property
    def domain(self):
        """The operator domain."""
        return self._op.domain

    @property
    def range(self):
        """The operator range."""
        return self._op.range

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

    def derivative(self, point):
        """Return the derivative at 'point'.

        The derivative of the right scalar operator multiplication
        follows the chain rule:

        `OperatorRightScalarMult(op, scalar).derivative(point) <==>
        LinearOperatorScalarMult(op.derivative(scalar * point),
        scalar)`
        """
        return LinearOperatorScalarMult(
            self._op.derivative(self._scalar * point), self._scalar)

    def __repr__(self):
        """`op.__repr__() <==> repr(op)`."""
        return 'OperatorRightScalarMult({!r}, {!r})'.format(self._op,
                                                            self._scalar)

    def __str__(self):
        """`op.__str__() <==> str(op)`."""
        return '{} * {}'.format(self._op, self._scalar)


class LinearOperator(Operator):

    """Abstract linear operators.

    A `LinearOperator` is an `Operator` which satisfies the linearity
    relation

    `op(scalar * x + y) == scalar * op(x) + op(y)`

    for all scalars and all vectors `x` and `y`. A `LinearOperator`
    can only be defined if `domain` and `range` are both `LinearSpace`
    instances.
    """

    @property
    def adjoint(self):
        """The operator adjoint."""
        raise NotImplementedError('adjoint not implemented for operator {!r}.'
                                  ''.format(self))

    # Implicitly defined operators
    @property
    def T(self):
        """Shorthand for `adjoint`."""
        return self.adjoint

    def derivative(self, point):
        """Return the operator derivative at 'point'.

        For a `LinearOperator`, this is equivalent to `op(point)`, i.e.
        a linear operator is its own derivative.
        """
        return self

    def __add__(self, other):
        """`op.__add__(other) <==> op + other`.

        If `other` is a `LinearOperator`, a `LinearOperatorSum` is
        returned.
        """
        if isinstance(other, LinearOperator):
            return LinearOperatorSum(self, other)
        else:
            return super().__add__(other)

    def __mul__(self, other):
        """`op.__mul__(other) <==> op * other`.

        If `other` is a scalar, this is equivalent to
        `op.__rmul__(other)`.
        """
        if isinstance(other, Operator):
            return OperatorPointwiseProduct(self, other)
        elif isinstance(other, Number):
            return LinearOperatorScalarMult(self, other)
        else:
            raise TypeError('multiplicant {!r} is neither operator nor '
                            'scalar.'.format(other))

    def __rmul__(self, other):
        """`op.__rmul__(other) <==> other * op`.

        Equivalent to `op.__mul__(other)`.
        """
        return self.__mul__(other)


class SelfAdjointOperator(LinearOperator):

    """Abstract self-adjoint operator.

    A self-adjoint operator is a `LinearOperator` with
    `op.adjoint == op`. This implies in particular that
    `op.domain == op.range`.
    """

    # pylint: disable=abstract-method
    @property
    def adjoint(self):
        """The operator adjoint, equal to the operator itself."""
        return self

    @property
    def range(self):
        """The operator range, equal to its `domain`."""
        return self.domain


class LinearOperatorSum(OperatorSum, LinearOperator):

    """Expression type for the sum of operators.

    `LinearOperatorSum(op1, op2) <==> (x --> op1(x) + op2(x))`
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp_ran=None, tmp_dom=None):
        """Initialize a new `LinearOperatorSum` instance.

        Parameters
        ----------

        op1 : `LinearOperator`
            The first summand
        op2 : `LinearOperator`
            The second summand. Must have the same `domain` and `range` as
            `op1`.
        tmp_ran : range element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        tmp_dom : domain element, optional
            Used to avoid the creation of a temporary when applying the
            operator adjoint.
        """
        if not isinstance(op1, LinearOperator):
            raise TypeError('first operator {!r} is not a LinearOperator '
                            'instance.'.format(op1))

        if not isinstance(op2, LinearOperator):
            raise TypeError('second operator {!r} is not a LinearOperator '
                            'instance'.format(op2))

        if tmp_dom is not None and tmp_dom not in op1.domain:
            raise TypeError('domain temporary {!r} not an element of the '
                            'operator domain {!r}.'
                            ''.format(tmp_dom, op1.domain))

        super().__init__(op1, op2, tmp_ran)
        self._tmp_dom = tmp_dom

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator sum is the sum of the operator
        adjoints:

        `LinearOperatorSum(op1, op2).adjoint ==
        `LinearOperatorSum(op1.adjoint, op2.adjoint)`
        """
        return LinearOperatorSum(self._op1.adjoint, self._op2.adjoint,
                                 self._tmp_dom, self._tmp)


class LinearOperatorComp(OperatorComp, LinearOperator):

    """Expression type for the composition of linear operators.

    `LinearOperatorComp(left, right) <==> (x --> left(right(x)))`

    The composition is only well-defined if
    `left.domain == right.range`.
    """

    def __init__(self, left, right, tmp=None):
        """Initialize a new `LinearOperatorComp` instance.

        Parameters
        ----------
        left : `LinearOperator`
            The left ("outer") operator
        right : `LinearOperator`
            The right ("inner") operator. Its range must coincide with the
            `domain` of `left`.
        tmp : `right.range` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if not isinstance(left, LinearOperator):
            raise TypeError('left operator {!r} is not a LinearOperator '
                            'instance.'.format(left))

        if not isinstance(right, LinearOperator):
            raise TypeError('right operator {!r} is not a LinearOperator '
                            'instance'.format(right))

        super().__init__(left, right, tmp)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator composition is the composition of
        the operator adjoints in reverse order:

        `LinearOperatorComp(left, right).adjoint ==
        `LinearOperatorComp(right.adjoint, left.adjoint)`
        """
        return LinearOperatorComp(self._right.adjoint, self._left.adjoint,
                                  self._tmp)


class LinearOperatorScalarMult(OperatorLeftScalarMult, LinearOperator):

    """Expression type for the linear operator scalar multiplication.

    `LinearOperatorScalarMult(op, scalar) <==> (x --> scalar * op(x))`

    For linear operators, left and right scalar multiplications are
    equal.
    """

    def __init__(self, op, scalar):
        """Initialize a new `OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : `Operator`
            The `domain` of `op` must be a `LinearSpace`.
        scalar : `op.range.field` element
            A real or complex number, depending on the field of
            `op.domain`.
        """
        if not isinstance(op, LinearOperator):
            raise TypeError('operator {!r} is not a LinearOperator instance.'
                            ''.format(op))

        super().__init__(op, scalar)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        `LinearOperatorScalarMult(op, scalar).adjoint ==
        `LinearOperatorScalarMult(op.adjoint, scalar)`
        """
        # TODO: take conj(scalar) if complex
        return LinearOperatorScalarMult(self._op.adjoint, self._scalar)


def _bound_method(function):
    """Add a `self` argument to a function.

    This way, the decorated function may be used as a bound method.
    """
    def method(_, *args, **kwargs):
        """Call function with *args, **kwargs."""
        return function(*args, **kwargs)

    return method


def operator(call=None, apply=None, inv=None, deriv=None,
             dom=UniversalSet(), ran=UniversalSet()):
    """Create a simple operator.

    Mostly intended for simple prototyping rather than final use.

    Parameters
    ----------
    call : callable
        A function taking one argument and returning the result.
        It will be used for the operator call pattern
        `outp = op(inp)`.
    apply : callable
        A function taking two arguments.
        It will be used for the operator apply pattern
        `op.apply(inp, outp) <==> outp <-- op(inp)`. Return value
        is assumed to be `None` and is ignored.
    inv : `Operator`, optional
        The operator inverse
        Default: `None`
    deriv : `LinearOperator`, optional
        The operator derivative
        Default: `None`
    dom : `Set`, optional
        The domain of the operator
        Default: `UniversalSet`
    ran : `Set`, optional
        The range of the operator
        Default: `UniversalSet`

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
        raise ValueError("at least one argument 'call' or 'apply' must be "
                         "given.")

    simple_operator = _OperatorMeta(
        'SimpleOperator', (Operator,),
        {'_call': _bound_method(call), '_apply': _bound_method(apply),
         'inverse': inv, 'derivative': deriv, 'domain': dom, 'range': ran})

    return simple_operator()


def linear_operator(call=None, apply=None, inv=None, adj=None,
                    dom=UniversalSet(), ran=UniversalSet()):
    """Create a simple linear operator.

    Mostly intended for simple prototyping rather than final use.

    Parameters
    ----------
    call : callable
        A function taking one argument and returning the result.
        It will be used for the operator call pattern
        `outp = op(inp)`.
    apply : callable
        A function taking two arguments.
        It will be used for the operator apply pattern
        `op.apply(inp, outp) <==> outp <-- op(inp)`. Return value
        is assumed to be `None` and is ignored.
    inv : `LinearOperator`, optional
        The operator inverse
        Default: `None`
    adj : `LinearOperator`, optional
        The operator adjoint
        Default: `None`
    dom : `LinearSpace`, optional
        The domain of the operator
        Default: `UniversalSet`
    ran : `Set`, optional
        The range of the operator
        Default: `UniversalSet`

    Returns
    -------
    op : `SimpleLinearOperator`
        An operator with the provided attributes and methods.
        `SimpleLinearOperator` is a subclass of `LinearOperator`.

    Notes
    -----
    It suffices to supply one of the functions `call` and `apply`.
    If `dom` is a `LinearSpace`, a default implementation of the
    respective other method is automatically provided; if not, a
    `NotImplementedError` is raised when the other method is called.

    Examples
    --------
    >>> A = linear_operator(lambda x: 3*x)
    >>> A(5)
    15
    """
    # FIXME: This function is inconsistent witht the LinearOperator
    # class requirements (domain and range). Either fix this or delete
    # the function!
    if call is None and apply is None:
        raise ValueError("Need to supply at least one of call or apply")

    simple_linear_operator = _OperatorMeta(
        'SimpleLinearOperator', (LinearOperator,),
        {'_call': _bound_method(call), '_apply': _bound_method(apply),
         'inverse': inv, 'adjoint': adj, 'domain': dom, 'range': ran})

    return simple_linear_operator()


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
