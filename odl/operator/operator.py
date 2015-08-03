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
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
standard_library.install_aliases()
from builtins import str, object, super
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

        inp : element of `domain`
            An object in the operator domain. The operator is applied
            to it.

        Returns
        -------

        out : element of `range`
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

        inp : element of self.domain
            An object in the operator domain. The operator is applied
            to it.

        out : element in self.range
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
        if "_call" in attrs and "_apply" in attrs:
            return super().__new__(mcs, name, bases, attrs)
        elif "_call" in attrs:
            return super().__new__(mcs, name, (_DefaultApplyOperator,) + bases,
                                   attrs)
        elif "_apply" in attrs:
            return super().__new__(mcs, name, (_DefaultCallOperator,) + bases,
                                   attrs)
        else:
            return super().__new__(mcs, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        """Create a new class `cls` from given arguments."""
        obj = type.__call__(cls, *args, **kwargs)
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
    ===============================

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

    inp : element of `domain`
        An object in the operator domain to which the operator is
        applied.

    **Returns:**

    outp : element of `range`
        An object in the operator range, the result of the operator
        evaluation.

    In-place evaluation: `_apply()`
    -------------------------------
    In-place evaluation means that the operator is applied, and the
    result is written to an existing element provided as an additional
    argument. In this case, a subclass has to implement the method

    `_apply(self, inp, outp)`     <==>     `outp <-- operator(inp)`

    **Parameters:**

    inp : element of self.domain
        An object in the operator domain to which the operator is
        applied.

    outp : element of self.range
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
    +---------------+----------------+--------+-----------------------+
    |`__add__(op2)` |`OperatorSum`            |Implements `op + op2`. |
    +---------------+-------------------------+-----------------------+
    |`__mul__(s)`   |`OperatorLeftScalarMult` |Implements `s * op`,   |
    |               |                         |with a scalar `s`.     |
    +---------------+-------------------------+-----------------------+
    |`__rmul__(s)`  |`OperatorRightScalarMult`|Implements `op * s`.   |
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

        >>> from odl.operator.default import operator
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
        """`op1.__add__(op2) <==> op1 + op2`."""
        return OperatorSum(self, other)

    def __mul__(self, other):
        """`op.__mul__(s) <==> op * s`.

        This corresponds to right multiplication of scalars with
        operators: `(A * s)(x) = A(s * x)`.

        Note that left and right multiplications are usually different.

        Parameters
        ----------
        other : scalar
            If `self.domain` is a `LinearSpace`, `scalar` must be an
            element of `self.domain.field`.

        Returns
        -------
        mul : `OperatorRightScalarMult`

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

        >>> from odl.operator.default import operator
        >>> A = operator(lambda x: 3*x)
        >>> Scaled = A*3
        >>> Scaled(5)
        45
        """
        return OperatorRightScalarMult(self, other)

    def __rmul__(self, other):
        """`op.__rmul__(s) <==> s * op`.

        This corresponds to left multiplication of scalars with
        operators: `(s * A)(x) = s * A(x)`.

        Note that left and right multiplications are usually different.

        Parameters
        ----------
        other : scalar
            If `self.range` is a `LinearSpace`, `scalar` must be an
            element of `self.range.field`.

        Returns
        -------
        mul : `OperatorLeftScalarMult`

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

        >>> from odl.operator.default import operator
        >>> A = operator(lambda x: 3*x)
        >>> Scaled = 3*A
        >>> Scaled(5)
        45
        """
        return OperatorLeftScalarMult(self, other)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return (self.__class__.__name__ + ": " + str(self.domain) +
                "->" + str(self.range))


class OperatorSum(Operator):

    """Expression type for the sum of operators.

    The sum is only well defined for Operators whose `range`s are
    `LinearSpace`s.

    Parameters
    ----------

    op1 : Operator
          The first operator
    op2 : Operator
          Operator with the same domain and range as op1
    tmp : Element in the range
          Used to avoid the creation of a
          temporary when applying the operator.
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp=None):
        if op1.range != op2.range:
            raise TypeError(errfmt('''
            OperatorSum requires that ranges of operators are equal. Given
            ranges do not match ({}, {}).
            '''.format(op1.range, op2.range)))

        if op1.domain != op2.domain:
            raise TypeError(errfmt('''
            OperatorSum requires that domains of operators are equal. Given
            domains do not match ({}, {}).
            '''.format(op1.domain, op2.domain)))

        if tmp is not None and not op1.domain.contains(tmp):
            raise TypeError(errfmt('''
            Tmp ({}) must be an element in the range of the operators ({}).
            '''.format(tmp, op1.domain)))

        self._op1 = op1
        self._op2 = op2
        self._tmp = tmp

    def _call(self, inp):
        """ Calculates op1(inp) + op2(inp)

        Parameters
        ----------
        inp : self.domain element
              The point to evaluate the sum in

        Returns
        -------
        result : self.range element
                 Result of the evaluation

        Example
        -------
        >>> from odl.operator.default import operator
        >>> A = operator(lambda x: 3*x)
        >>> B = operator(lambda x: 5*x)
        >>> OperatorSum(A, B)(3)
        24
        """
        # pylint: disable=protected-access
        return self._op1._call(inp) + self._op2._call(inp)

    def _apply(self, inp, outp):
        """
        Calculates op1(inp) + op2(inp)

        Parameters
        ----------
        inp : self.domain element
              The point to evaluate the sum in
        outp : self.range element
              Object to store the result in

        Returns
        -------
        None

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

    @property
    def domain(self):
        """
        Get the operator domain

        Parameters
        ----------
        None

        Returns
        -------
        domain : Set
                 The domain of the operator

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
        """
        Get the operator range

        Parameters
        ----------
        None

        Returns
        -------
        domain : Set
                 The domain of the operator

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
        return LinearOperatorSum(self._op1.derivative(point),
                                 self._op2.derivative(point))

    def __repr__(self):
        return 'OperatorSum( ' + repr(self._op1) + ", " + repr(self._op2) + ')'

    def __str__(self):
        return '(' + str(self._op1) + ' + ' + str(self._op2) + ')'


class OperatorComp(Operator):
    """Expression type for the composition of operators

    OperatorComp(left, right)(x) = left(right(x))

    Parameters
    ----------

    left : Operator
           The first operator
    right : Operator
            Operator with the same domain and range as op1
    tmp : Element in the operator range
          Used to avoid the creation of a
          temporary when applying the operator.
    """

    def __init__(self, left, right, tmp=None):
        if right.range != left.domain:
            raise TypeError(errfmt('''
            Range of right operator ({}) does not equal domain of left
            operator ({})
            '''.format(right.range, left.domain)))

        if tmp is not None and not right.range.contains(tmp):
            raise TypeError(errfmt('''
            Tmp ({}) must be an element in the range of the operators ({}).
            '''.format(tmp, left.domain)))

        self._left = left
        self._right = right
        self._tmp = tmp

    def _call(self, inp):
        # pylint: disable=protected-access
        return self._left._call(self._right._call(inp))

    def _apply(self, inp, outp):
        # pylint: disable=protected-access
        tmp = (self._tmp if self._tmp is not None
               else self._right.range.element())
        self._right._apply(inp, tmp)
        self._left._apply(tmp, outp)

    @property
    def domain(self):
        return self._right.domain

    @property
    def range(self):
        return self._left.range

    @property
    def inverse(self):
        return OperatorComp(self._right.inverse, self._left.inverse, self._tmp)

    def derivative(self, point):
        left_deriv = self._left.derivative(self._right(point))
        right_deriv = self._right.derivative(point)

        return LinearOperatorComp(left_deriv, right_deriv)

    def __repr__(self):
        return ('OperatorComp( ' + repr(self._left) + ', ' +
                repr(self._right) + ')')

    def __str__(self):
        return str(self._left) + ' o ' + str(self._right)


class OperatorPointwiseProduct(Operator):
    """Pointwise multiplication of operators defined on Banach Algebras
    (with pointwise multiplication)

    OperatorPointwiseProduct(op1, op2)(x) = op1(x) * op2(x)
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2):
        if op1.range != op2.range:
            raise TypeError(errfmt('''
            Ranges ({}, {}) of operators are not equal
            '''.format(op1.range, op2.range)))

        if op1.domain != op2.domain:
            raise TypeError(errfmt('''
            Domains ({}, {}) of operators are not equal
            '''.format(op1.domain, op2.domain)))

        self._op1 = op1
        self._op2 = op2

    def _call(self, inp):
        # pylint: disable=protected-access
        return self._op1._call(inp) * self._op2._call(inp)

    def _apply(self, inp, outp):
        # pylint: disable=protected-access
        tmp = self._op2.range.element()
        self._op1._apply(inp, outp)
        self._op2._apply(inp, tmp)
        outp *= tmp

    @property
    def domain(self):
        return self._op1.domain

    @property
    def range(self):
        return self._op1.range


class OperatorLeftScalarMult(Operator):
    """Expression type for the left multiplication of operators with
    scalars

    OperatorLeftScalarMult(op, scalar)(x) = scalar * op(x)
    """

    def __init__(self, op, scalar):
        if (isinstance(op.range, LinearSpace) and
                scalar not in op.range.field):
            raise TypeError(errfmt('''
            'scalar' ({}) not compatible with field of range ({}) of 'op'
            '''.format(scalar, op.range.field)))

        self._op = op
        self._scalar = scalar

    def _call(self, inp):
        # pylint: disable=protected-access
        return self._scalar * self._op._call(inp)

    def _apply(self, inp, outp):
        # pylint: disable=protected-access
        self._op._apply(inp, outp)
        outp *= self._scalar

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range

    @property
    def inverse(self):
        return OperatorRightScalarMult(self._op.inverse, 1.0/self._scalar)

    def derivative(self, point):
        return LinearOperatorScalarMult(self._op.derivative(point),
                                        self._scalar)

    def __repr__(self):
        return ('OperatorLeftScalarMult( ' + repr(self._op) +
                ', ' + repr(self._scalar) + ')')

    def __str__(self):
        return str(self._scalar) + " * " + str(self._op)


class OperatorRightScalarMult(Operator):
    """Expression type for the right multiplication of operators with
    scalars.

    OperatorRightScalarMult(op, scalar)(x) = op(scalar * x)

    Typically slower than left multiplication since this requires a
    copy.

    Parameters
    ----------

    op : Operator
        Any operator whose range supports `*= scalar`
    scalar : Number
        An element in the field of the domain of op
    tmp : Element in the operator range
        Used to avoid the creation of a
        temporary when applying the operator.
    """

    def __init__(self, op, scalar, tmp=None):
        if (isinstance(op.domain, LinearSpace) and
                scalar not in op.domain.field):
            raise TypeError(errfmt('''
            'scalar' ({}) not compatible with field of domain ({}) of 'op'
            '''.format(scalar, op.domain.field)))

        if tmp is not None and not op.domain.contains(tmp):
            raise TypeError(errfmt('''
            'tmp' ({}) must be an element in the domain of 'op' ({}).
            '''.format(tmp, op.domain)))

        self._op = op
        self._scalar = scalar
        self._tmp = tmp

    def _call(self, inp):
        # pylint: disable=protected-access
        return self._op._call(self._scalar * inp)

    def _apply(self, inp, outp):
        # pylint: disable=protected-access
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, inp)
        self._op._apply(tmp, outp)

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range

    @property
    def inverse(self):
        return OperatorLeftScalarMult(self._op.inverse, 1.0/self._scalar)

    def derivative(self, point):
        return LinearOperatorScalarMult(self._op.derivative(point),
                                        self._scalar)

    def __repr__(self):
        return ('OperatorRightScalarMult( ' + self._op.__repr__() +
                ', ' + repr(self._scalar) + ')')

    def __str__(self):
        return str(self._op) + " * " + str(self._scalar)


# DEFINITION OF LINEAR OPERATOR AND RELATED METHODS
class LinearOperator(Operator):
    """ Linear operator, satisfies A(a*x + b*y) = a*A(x) + b*A(y)

    LinearOperators are only defied on LinearSpace:s.
    """

    @property
    def adjoint(self):
        """Get the adjoint of the operator. Abstract, should be
        implemented by subclasses.

        op.T.apply(inp, outp) = op.adjoint.apply(inp,outp)
        and
        op.T.adjoint.apply(inp, outp) = op.apply(inp,outp)
        """
        raise NotImplementedError(errfmt('''
        Adjoint not implemented for this operator ({})
        '''.format(self)))

    # Implicitly defined operators
    @property
    def T(self):
        """ Get the adjoint of this operator
        """
        return self.adjoint

    def derivative(self, point):
        """ The derivative of linear operators is the operator itself
        """
        return self

    def __add__(self, other):
        """Operator addition

        (self + other)(x) = self(x) + other(x)
        """

        if isinstance(other, LinearOperator):
            # Special if both are linear
            return LinearOperatorSum(self, other)
        else:
            return super().__add__(other)

    def __mul__(self, other):
        """Multiplication of operators with scalars.

        (a*A)(x) = a * A(x)
        (A*a)(x) = a * A(x)
        """

        if isinstance(other, Number):
            return LinearOperatorScalarMult(self, other)
        else:
            raise TypeError('Expected an operator or a scalar')

    __rmul__ = __mul__


class SelfAdjointOperator(LinearOperator):
    """ Special case of self adjoint operators where A(x) = A.T(x)
    """

    # pylint: disable=abstract-method
    @property
    def adjoint(self):
        return self

    @property
    def range(self):
        return self.domain


class LinearOperatorSum(OperatorSum, LinearOperator):
    """Expression type for the sum of linear operators
    """

    # pylint: disable=abstract-method
    def __init__(self, op1, op2, tmp_ran=None, tmp_dom=None):
        """Create the abstract operator sum defined by:

        LinearOperatorSum(op1, op2)(x) = op1(x) + op2(x)

        Args:
            LinearOperator  `op1`      The first operator
            LinearOperator  `op2`      The second operator
            Vector          `tmp_ran`  A vector in the operator domain.
                                       Used to avoid the creation of a
                                       temporary when applying the operator.
            Vector          `tmp_dom`  A vector in the operator range.
                                       Used to avoid the creation of a
                                       temporary when applying the adjoint of
                                       this operator.
        """
        if not isinstance(op1, LinearOperator):
            raise TypeError(errfmt('''
            op1 ({}) is not a LinearOperator. LinearOperatorSum is only
            defined for LinearOperators.'''.format(op1)))
        if not isinstance(op2, LinearOperator):
            raise TypeError(errfmt('''
            op2 ({}) is not a LinearOperator. LinearOperatorSum is only
            defined for LinearOperators.'''.format(op2)))

        super().__init__(op1, op2, tmp_ran)
        self._tmp_dom = tmp_dom

    @property
    def adjoint(self):
        return LinearOperatorSum(self._op1.adjoint, self._op2.adjoint,
                                 self._tmp_dom, self._tmp)


class LinearOperatorComp(OperatorComp, LinearOperator):
    """Expression type for the composition of linear operators
    """

    def __init__(self, left, right, tmp=None):
        """ Create the abstract operator composition defined by:

        LinearOperatorComp(left, right)(x) = left(right(x))

        With adjoint defined by
        LinearOperatorComp(left, right).T(y) = right.T(left.T(y))

        Args:
            LinearOperator  `left`      The first operator
            LinearOperator  `right`     The second operator
            Vector          `tmp`       A vector in the range of `right`.
                                        Used to avoid the creation of a
                                        temporary when applying the operator.
        """

        if not isinstance(left, LinearOperator):
            raise TypeError(errfmt('''
            left ({}) is not a LinearOperator. LinearOperatorComp is
            only defined for LinearOperators'''.format(left)))
        if not isinstance(right, LinearOperator):
            raise TypeError(errfmt('''
            right ({}) is not a LinearOperator. LinearOperatorComp is
            only defined for LinearOperators'''.format(right)))

        super().__init__(left, right, tmp)

    @property
    def adjoint(self):
        return LinearOperatorComp(self._right.adjoint, self._left.adjoint,
                                  self._tmp)


class LinearOperatorScalarMult(OperatorLeftScalarMult,
                               LinearOperator):
    """Expression type for the multiplication of operators with scalars
    """

    def __init__(self, op, scalar):
        """ Create the LinearOperatorScalarMult defined by

        LinearOperatorScalarMult(op, scalar)(x) = scalar * op(x)

        Args:
            Operator    `op`        Any operator
            Scalar      `scalar`    An element in the field of the domain
                                    of `op`
        """
        if not isinstance(op, LinearOperator):
            raise TypeError(errfmt('''
            op ({}) is not a LinearOperator.
            LinearOperatorScalarMult is only defined for
            LinearOperators'''.format(op)))

        super().__init__(op, scalar)

    @property
    def adjoint(self):
        return LinearOperatorScalarMult(self._op.adjoint, self._scalar)


def _instance_method(function):
    """ Adds a self argument to a function
    such that it may be used as a instance method
    """
    def method(_, *args, **kwargs):
        """  Calls function with *args, **kwargs
        """
        return function(*args, **kwargs)

    return method


def operator(call=None, apply=None, inv=None, deriv=None,
             dom=UniversalSet(), ran=UniversalSet()):
    """ Creates a simple operator.

    Mostly intended for testing.

    Parameters
    ----------
    call : Function taking one argument (inp) returns result
           The operators _call method
    apply : Function taking two arguments (inp, outp) returns None
            The operators _apply method
    inv : Operator, optional
          The inverse operator
          Default: None
    deriv : LinearOperator, optional
            The derivative operator
            Default: None
    dom : Set, optional
             The domain of the operator
             Default: UniversalSet
    ran : Set, optional
            The range of the operator
            Default: UniversalSet

    Returns
    -------
    operator : Operator
               An operator with the required properties

    Example
    -------
    >>> A = operator(lambda x: 3*x)
    >>> A(5)
    15
    """

    if call is None and apply is None:
        raise ValueError("Need to supply at least one of call or apply")

    metaclass = Operator.__metaclass__

    simple_operator = metaclass('SimpleOperator',
                                (Operator,),
                                {'_call': _instance_method(call),
                                 '_apply': _instance_method(apply),
                                 'inverse': inv,
                                 'derivative': deriv,
                                 'domain': dom,
                                 'range': ran})

    return simple_operator()


def linear_operator(call=None, apply=None, inv=None, deriv=None, adj=None,
                    dom=UniversalSet(), ran=UniversalSet()):
    """ Creates a simple operator.

    Mostly intended for testing.

    Parameters
    ----------
    call : Function taking one argument (inp) returns result
           The operators _call method
    apply : Function taking two arguments (inp, outp) returns None
            The operators _apply method
    inv : Operator, optional
          The inverse operator
          Default: None
    deriv : LinearOperator, optional
            The derivative operator
            Default: None
    adj : LinearOperator, optional
          The adjoint of the operator
          Defualt: None
    dom : Set, optional
             The domain of the operator
             Default: UniversalSet
    ran : Set, optional
            The range of the operator
            Default: UniversalSet

    Returns
    -------
    operator : LinearOperator
               An operator with the required properties

    Example
    -------
    >>> A = linear_operator(lambda x: 3*x)
    >>> A(5)
    15
    """

    if call is None and apply is None:
        raise ValueError("Need to supply at least one of call or apply")

    metaclass = LinearOperator.__metaclass__

    simple_linear_operator = metaclass('SimpleOperator',
                                       (LinearOperator,),
                                       {'_call': _instance_method(call),
                                        '_apply': _instance_method(apply),
                                        'inverse': inv,
                                        'derivative': deriv,
                                        'adjoint': adj,
                                        'domain': dom,
                                        'range': ran})

    return simple_linear_operator()


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
