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

+--------------------------+------------------------------------------+
|Class name                |Description                               |
+==========================+==========================================+
|`Operator`                |Basic operator class                      |
+--------------------------+------------------------------------------+
|`OperatorSum`             |Sum of two operators, `S = A + B`, defined|
|                          |by `x` --> `(A + B)(x) = A(x) + B(x)`     |
+--------------------------+------------------------------------------+
|`OperatorComp`            |Composition of two operators, `C = A o B`,|
|                          |defined by `x` --> `(A o B)(x) = A(B(x))` |
+--------------------------+------------------------------------------+
|`OperatorPointwiseProduct`|Product of two operators,`P = A * B`,     |
|                          |defined by                                |
|                          |`x --> (A * B)(x) = A(x) * B(x)`. The     |
|                          |operators need to be mappings to an       |
|                          |algebra for the multiplication to be      |
|                          |well-defined.                             |
+--------------------------+------------------------------------------+
|`OperatorLeftScalarMult`  |Multiplication of an operator from left   |
|                          |with a scalar, `L = c * A`, defined by    |
|                          |`x --> (c * A)(x) = c * A(x)`             |
+--------------------------+------------------------------------------+
|`OperatorRightScalarMult` |Multiplication of an operator from right  |
|                          |with a scalar, `S = A * c`, defined by    |
|                          |`x --> (A * c)(x) =  A(c * x)`            |
+--------------------------+------------------------------------------+
|`LinearOperator`          |Basic linear operator class               |
+--------------------------+------------------------------------------+
|`SelfAdjointOperator`     |Basic class for linear operators between  |
|                          |Hilbert spaces which are self-adjoint     |
+--------------------------+------------------------------------------+
|`LinearOperatorSum`       |Sum of two linear operators, again a      |
|                          |linear operator (see `OperatorSum` for the|
|                          |definition)                               |
+--------------------------+------------------------------------------+
|`LinearOperatorScalarMult`|Multiplication of a linear operator with a|
|                          |scalar. Left and right multiplications are|
|                          |equivalent (see `OperatorLeftScalarMult`  |
|                          |for the definition)                       |
+--------------------------+------------------------------------------+

Furthermore, there are two convenience functions for quick operator
prototyping:

+-----------------+----------------------+----------------------------+
|Name             |Return type           |Description                 |
+=================+======================+============================+
|`operator`       |`SimpleOperator`      |Create an `Operator` by     |
|                 |(subclass of          |specifying either a `call`  |
|                 |`Operator`)           |or an `apply` method (or    |
|                 |                      |both) for evaluation. See   |
|                 |                      |the function doc for a full |
|                 |                      |description.                |
+-----------------+----------------------+----------------------------+
|`linear_operator`|`SimpleLinearOperator`|Create an `Operator` by     |
|                 |(subclass of          |specifying either a `call`  |
|                 |`LinearOperator`)     |or an `apply` method (or    |
|                 |                      |both) for evaluation as well|
|                 |                      |`domain` and `range`. See   |
|                 |                      |the function doc for a full |
|                 |                      |description.                |
+-----------------+----------------------+----------------------------+
"""

# Imports for common Python 2/3 codebase
from __future__ import (print_function, division, absolute_import)
from future import standard_library
standard_library.install_aliases()
from builtins import object, super
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta
from numbers import Number

# ODL imports
from odl.utility.utility import errfmt
from odl.space.space import LinearSpace
from odl.space.set import UniversalSet


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
            raise NotImplementedError(errfmt('''
            `Operator` instances must have a `domain` attribute.'''))
        if not hasattr(obj, 'range'):
            raise NotImplementedError(errfmt('''
            `Operator` instances must have a `range` attribute.'''))
        if not hasattr(obj, '_call') and not hasattr(obj, '_apply'):
            raise NotImplementedError(errfmt('''
            `Operator` instances must either have `_call` or `_apply`
            as attribute.'''))

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

    Attributes
    ----------

    +------------+----------------+-----------------------------------+
    |Name        |Type            |Description                        |
    +============+================+===================================+
    |`domain`    |`Set`           |Elements to which the operator can |
    |            |                |be applied                         |
    +------------+----------------+-----------------------------------+
    |`range`     |`Set`           |Results of operator application are|
    |            |                |elements of this set.              |
    +------------+----------------+-----------------------------------+
    |`inverse`   |`Operator`      |The inverse operator. Raises       |
    |(short:`I`) |                |`NotImplementedError` by default.  |
    +------------+----------------+-----------------------------------+

    Methods
    -------
    +---------------+----------------+--------------------------------+
    |Signature      |Return type     |Description                     |
    +===============+================+================================+
    |`apply(inp,    |`None`          |Apply the operator to `inp` and |
    |outp)`         |                |write to `outp`. In addition to |
    |               |                |the private method `_apply()`,  |
    |               |                |error checks are performed.     |
    +---------------+----------------+--------------------------------+
    |`__call__(inp)`|element of      |Implements the call pattern     |
    |               |`range`         |`op(inp)`. In addition to the   |
    |               |                |private method `_call()`, error |
    |               |                |checks are performed.           |
    +---------------+----------------+--------------------------------+
    |`derivative    |`LinearOperator`|The operator derivative at      |
    |(point)`       |                |`point`. Raises                 |
    |               |                |`NotImplementedError` by        |
    |               |                |default.                        |
    +---------------+----------------+--------------------------------+
    |`__add__(op2)` |`OperatorSum`   |Implements `op + op2`.          |
    +---------------+-------------------------+-----------------------+
    |`__mul__`      |depends         |Implements `other * op`. If     |
    |(other)        |                |`other is a scalar, an          |
    |               |                |`OperatorLeftScalarMult` is     |
    |               |                |created. If `other` is an       |
    |               |                |`Operator`, the result is an    |
    |               |                |`OperatorPointwiseProduct`.     |
    +---------------+-------------------------+-----------------------+
    |`__rmul__`     |depends         |Implements `op * other`. If     |
    |(other)        |                |`other is a scalar, an          |
    |               |                |`OperatorRightScalarMult` is    |
    |               |                |created. If `other` is an       |
    |               |                |`Operator`, the result is an    |
    |               |                |`OperatorPointwiseProduct`.     |
    +---------------+-------------------------+-----------------------+
    """

    def derivative(self, point):
        """Return the operator derivative at `point`."""
        raise NotImplementedError(errfmt('''
        `derivative` not implemented for operator {!r}'''.format(self)))

    @property
    def inverse(self):
        """Return the operator inverse."""
        raise NotImplementedError(errfmt('''
        `inverse` not implemented for operator {!r}.'''.format(self)))

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
        inp : element of `self.domain`
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.

        outp : element of `self.range`
            An object in the operator range to which the result of the
            operator evaluation is written. The result is independent
            of the initial state of this object.

        Returns
        -------
        None

        Example
        -------

        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
        >>> rn = Rn(3)
        >>> op = IdentityOperator(rn)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element()
        >>> op.apply(x, y)
        >>> y
        Rn(3).element([1.0, 2.0, 3.0])
        """
        if not self.domain.contains(inp):
            raise TypeError(errfmt('''
            inp ({}) is not in the operator domain ({})
            '''.format(repr(inp), repr(self))))

        if not self.range.contains(outp):
            raise TypeError(errfmt('''
            outp ({}) is not in the operator range ({})
            '''.format(repr(outp), repr(self))))

        if inp is outp:
            raise ValueError(errfmt('''
            inp ({}) is the same as outp ({}) operators do not permit
            aliased arguments
            '''.format(repr(inp), repr(outp))))

        self._apply(inp, outp)

    def __call__(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        Implementation of the call pattern `op(inp)` with the private
        `_call()` method and added error checking.

        Parameters
        ----------
        inp : element of `self.domain`
            An object in the operator domain to which the operator is
            applied. The object is treated as immutable, hence it is
            not modified during evaluation.

        Returns
        -------
        element of `self.range`
            An object in the operator range, the result of the operator
            evaluation.

        Example
        -------

        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
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
        if not self.domain.contains(inp):
            raise TypeError(errfmt('''
            inp ({}) is not in the operator domain ({})
            '''.format(repr(inp), repr(self))))

        result = self._call(inp)

        if not self.range.contains(result):
            raise TypeError(errfmt('''
            result ({}) is not in the operator domain ({})
            '''.format(repr(result), repr(self))))

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
        mul : `OperatorPointwiseProduct` or `OperatorRightScalarMult`

        Example
        -------

        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
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
            raise TypeError(errfmt('''
            type {} of `other` is neither `Operator` nor scalar.
            '''.format(type(other))))

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
        mul : `OperatorPointwiseProduct` or `OperatorLeftScalarMult`

        Example
        -------

        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
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
            raise TypeError(errfmt('''
            type {} of `other` is neither `Operator` nor scalar.
            '''.format(type(other))))

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

    See also
    --------
    See `Operator` for a list of public attributes and methods.
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
        tmp : `range` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if op1.range != op2.range:
            raise TypeError(errfmt('''
            `op1.range` {} and `op2.range` {} not equal.
            '''.format(op1.range, op2.range)))

        if not isinstance(op1.range, LinearSpace):
            raise TypeError(errfmt('''
            `range` {} not a `LinearSpace`.'''.format(op1.range)))

        if op1.domain != op2.domain:
            raise TypeError(errfmt('''
            `op1.domain` {} and `op2.domain` {} not equal.
            '''.format(op1.domain, op2.domain)))

        if tmp is not None and tmp not in op1.domain:
            raise TypeError(errfmt('''
            `tmp` {} not in `domain` {}.'''.format(tmp, op1.domain)))

        self._op1 = op1
        self._op2 = op2
        self._tmp = tmp

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.

        Example
        -------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
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

        See also
        --------
        See `Operator` for an explanation of the method.

        Example
        -------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import ScalingOperator
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
        """The operator `domain`.

        Example
        -------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> OperatorSum(op, op).domain
        Rn(3)
        """
        return self._op1.domain

    @property
    def range(self):
        """The operator `range`.

        Example
        -------
        >>> from odl.space.cartesian import Rn
        >>> from odl.operator.default import IdentityOperator
        >>> r3 = Rn(3)
        >>> op = IdentityOperator(r3)
        >>> OperatorSum(op, op).range
        Rn(3)
        """
        return self._op1.range

    def derivative(self, point):
        """Return the operator derivative at `point`.

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

    See also
    --------
    See `Operator` for a list of public attributes and methods.
    """

    def __init__(self, left, right, tmp=None):
        """Initialize a new `OperatorComp` instance.

        Parameters
        ----------
        left : `Operator`
            The left ("outer") operator
        right : `Operator`
            The right ("inner") operator. Its range must coincide with the
            `domain` of `left`.
        tmp : `right.range` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if right.range != left.domain:
            raise TypeError(errfmt('''
            `right.range` {} not equal to `left.domain` {}.
            '''.format(right.range, left.domain)))

        if tmp is not None and tmp not in right.range:
            raise TypeError(errfmt('''
            `tmp` {} not in `right.range` {}.
            '''.format(tmp, left.domain)))

        self._left = left
        self._right = right
        self._tmp = tmp

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
        # pylint: disable=protected-access
        return self._left._call(self._right._call(inp))

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
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

    See also
    --------
    See `Operator` for a list of public attributes and methods.
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2):
        """Initialize a new `OperatorPointwiseProduct` instance.

        Parameters
        ----------
        op1 : `Operator`
            The first factor. Its `range` must be an `Algebra`.
        op2 : `Operator`
            The second factor. Must have the same `domain` and `range`
            as `op1`.
        """
        if op1.range != op2.range:
            raise TypeError(errfmt('''
            `op1.range` {} and `op2.range` {} not equal.
            '''.format(op1.range, op2.range)))

        if not isinstance(op1.range, Algebra):
            raise TypeError(errfmt('''
            `range` {} not a `LinearSpace`.'''.format(op1.range)))

        if op1.domain != op2.domain:
            raise TypeError(errfmt('''
            `op1.domain` {} and `op2.domain` {} not equal.
            '''.format(op1.domain, op2.domain)))

        self._op1 = op1
        self._op2 = op2

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
        # pylint: disable=protected-access
        return self._op1._call(inp) * self._op2._call(inp)

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
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
            The `range` of `op` must be a `LinearSpace`.
        scalar : `op.range.field` element
            A real or complex number, depending on the field of
            `op.range`.
        """
        if not isinstance(op.range, LinearSpace):
            raise TypeError(errfmt('''
            `op.range` {} is not a `LinearSpace`.'''.format(op.range)))

        if scalar not in op.range.field:
            raise TypeError(errfmt('''
            `scalar` {} not in `op.range.field` {}.
            '''.format(scalar, op.range.field)))

        self._op = op
        self._scalar = scalar

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
        # pylint: disable=protected-access
        return self._scalar * self._op._call(inp)

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
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

        `OperatorLeftScalarMult(op, scalar).inverse <==>
        OperatorRightScalarMult(op.inverse, 1.0/scalar)`

        See also
        --------
        `OperatorRightScalarMult`
        """
        if self.scalar == 0.0:
            raise ZeroDivisionError('{} not invertible.'.format(self))
        return OperatorRightScalarMult(self._op.inverse, 1.0/self._scalar)

    def derivative(self, point):
        """Return the derivative at 'point'.

        Left scalar multiplication and derivative are commutative:

        `OperatorLeftScalarMult(op, scalar).derivative(point) <==>
        LinearOperatorScalarMult(op.derivative(point), scalar)`

        See also
        --------
        `LinearOperatorScalarMult`
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

    `OperatorRightScalarMult(op, scalar) <==> (x --> op(scalar * x))`

    The scalar multiplication is well-defined only if `op.domain` is
    a `LinearSpace`.
    """

    def __init__(self, op, scalar, tmp=None):
        """Initialize a new `OperatorLeftScalarMult` instance.

        Parameters
        ----------
        op : `Operator`
            The `domain` of `op` must be a `LinearSpace`.
        scalar : `op.range.field` element
            A real or complex number, depending on the field of
            `op.domain`.
        tmp : `domain` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        """
        if not isinstance(op.domain, LinearSpace):
            raise TypeError(errfmt('''
            `op.domain` {} is not a `LinearSpace`.'''.format(op.domain)))

        if scalar not in op.domain.field:
            raise TypeError(errfmt('''
            `scalar` {} not in `op.domain.field` {}.
            '''.format(scalar, op.domain.field)))

        if tmp is not None and tmp not in op.domain:
            raise TypeError(errfmt('''
            'tmp' {} not in `op.domain {}`.'''.format(tmp, op.domain)))

        self._op = op
        self._scalar = scalar
        self._tmp = tmp

    def _call(self, inp):
        """`op.__call__(inp) <==> op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
        # pylint: disable=protected-access
        return self._op._call(self._scalar * inp)

    def _apply(self, inp, outp):
        """`op.apply(inp, outp) <==> outp <-- op(inp)`.

        See also
        --------
        See `Operator` for an explanation of the method.
        """
        # pylint: disable=protected-access
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, inp)
        self._op._apply(tmp, outp)

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

        The inverse of `op * scalar` is given by
        `1/scalar * op.inverse` if `scalar != 0`. If `scalar == 0`,
        the inverse is not defined.

        `OperatorRightScalarMult(op, scalar).inverse <==>
        OperatorLeftScalarMult(op.inverse, 1.0/scalar)`

        See also
        --------
        `OperatorLeftScalarMult`
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

        See also
        --------
        `LinearOperatorScalarMult`
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

    Differences to `Operator`
    -------------------------

    +----------------+----------------+-------------------------------+
    |Attribute/Method|(Return) type   |Description                    |
    +================+================+===============================+
    |`adjoint`       |`LinearOperator`|Additional attribute. Satisfies|
    |(short: `T`)    |                |`op.adjoint.domain ==          |
    |                |                |op.range`, `op.adjoint.range ==|
    |                |                |op.domain` and                 |
    |                |                |`op.domain.inner(x,            |
    |                |                |op.adjoint(y)) ==              |
    |                |                |op.range.inner(op(x), y)`.     |
    +----------------+----------------+-------------------------------+
    |`derivative     |`LinearOperator`|`op.derivative == op`          |
    |(point)`        |                |                               |
    +----------------+----------------+-------------------------------+
    |`__add__(other)`|depends         |If `other` is a                |
    |                |                |`LinearOperator` the sum is a  |
    |                |                |`LinearOperatorSum`            |
    +----------------+----------------+-------------------------------+
    |`__add__(other)`|depends         |If `other` is a                |
    |                |                |`LinearOperator`, the sum is a |
    |                |                |`LinearOperatorSum`            |
    +----------------+----------------+-------------------------------+
    |`__mul__(other)`|depends         |If `other` is a scalar, the    |
    |                |                |product is a                   |
    |                |                |`LinearOperatorScalarMult`     |
    +----------------+----------------+-------------------------------+
    |`__rmul__       |depends         |If `other` is a scalar, the    |
    |(other)`        |                |product is a                   |
    |                |                |`LinearOperatorScalarMult`     |
    +----------------+----------------+-------------------------------+

    See also
    --------
    See `Operator` for a list of public attributes and methods as well
    as further help.
    """

    @property
    def adjoint(self):
        """The operator adjoint."""
        raise NotImplementedError(errfmt('''
        `adjoint` not implemented for operator {!r}.'''.format(self)))

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

        See also
        --------
        `Operator.__mul__()`
        """
        if isinstance(other, Operator):
            return OperatorPointwiseProduct(self, other)
        elif isinstance(other, Number):
            return LinearOperatorScalarMult(self, other)
        else:
            raise TypeError(errfmt('''
            type {} of `other` is neither `Operator` nor scalar.
            '''.format(type(other))))

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

    See also
    --------
    See `LinearOperator` and `Operator` for a list of public attributes
    and methods as well as further help.
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

    See also
    --------
    See `Operator` for a list of public attributes and methods as well
    as further help.
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
        tmp_ran : `range` element, optional
            Used to avoid the creation of a temporary when applying the
            operator.
        tmp_dom : `domain` element, optional
            Used to avoid the creation of a temporary when applying the
            operator adjoint.
        """
        if not isinstance(op1, LinearOperator):
            raise TypeError(errfmt('''
            `op1` {} not a `LinearOperator`.'''.format(op1)))

        if not isinstance(op2, LinearOperator):
            raise TypeError(errfmt('''
            `op2` {} not a `LinearOperator`.'''.format(op2)))

        if op1.range != op2.range:
            raise TypeError(errfmt('''
            `op1.range` {} and `op2.range` {} not equal.
            '''.format(op1.range, op2.range)))

        if op1.domain != op2.domain:
            raise TypeError(errfmt('''
            `op1.domain` {} and `op2.domain` {} not equal.
            '''.format(op1.domain, op2.domain)))

        if tmp_ran is not None and tmp_ran not in op1.range:
            raise TypeError(errfmt('''
            `tmp_ran` {} not in `range` {}.'''.format(tmp_ran, op1.range)))

        if tmp_dom is not None and tmp_dom not in op1.domain:
            raise TypeError(errfmt('''
            `tmp_dom` {} not in `domain` {}.'''.format(tmp_dom, op1.domain)))

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

    See also
    --------
    See `LinearOperator` and `Operator` for a list of public
    attributes and methods as well as further help.
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
            raise TypeError(errfmt('''
            `left` {} not a `LinearOperator`.'''.format(left)))

        if not isinstance(right, LinearOperator):
            raise TypeError(errfmt('''
            `right` {} not a `LinearOperator`.'''.format(right)))

        if right.range != left.domain:
            raise TypeError(errfmt('''
            `right.range` {} not equal to `left.domain` {}.
            '''.format(right.range, left.domain)))

        if tmp is not None and tmp not in right.range:
            raise TypeError(errfmt('''
            `tmp` {} not in `right.range` {}.
            '''.format(tmp, left.domain)))

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

    See also
    --------
    See `LinearOperator` and `Operator` for a list of public
    attributes and methods as well as further help.
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
            raise TypeError(errfmt('''
            `op` {} not a `LinearOperator`.'''.format(op)))

        if scalar not in op.range.field:
            raise TypeError(errfmt('''
            `scalar` {} not in `op.range.field`.'''.format(scalar)))

        super().__init__(op, scalar)

    @property
    def adjoint(self):
        """The operator adjoint.

        The adjoint of the operator scalar multiplication is the
        scalar multiplication of the operator adjoint:

        `LinearOperatorScalarMult(op, scalar).adjoint ==
        `LinearOperatorScalarMult(op.adjoint, scalar)`
        """
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

    Remark
    ------
    It suffices to supply one of the functions `call` and `apply`.
    If `dom` is a `LinearSpace`, a default implementation of the
    respective other method is automatically provided; if not, a
    `NotImplementedError` is raised when the other method is called.

    See also
    --------
    See `Operator` for a list of public attributes and methods as well
    as further help.

    Example
    -------
    >>> A = operator(lambda x: 3*x)
    >>> A(5)
    15
    """
    if call is None and apply is None:
        raise ValueError(errfmt('''
        At least one argument `call` or `apply` must be given.'''))

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

    Remark
    ------
    It suffices to supply one of the functions `call` and `apply`.
    If `dom` is a `LinearSpace`, a default implementation of the
    respective other method is automatically provided; if not, a
    `NotImplementedError` is raised when the other method is called.

    See also
    --------
    See `LinearOperator` and `Operator` for a list of public attributes
    and methods as well as further help.

    Example
    -------
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
