# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.

# pylint: disable=protected-access
"""
General abstract operators defined on Set's aswell as sums
and compositions of such.

Also contains LinearOperator specializations for linear operators
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, object, super
from future import standard_library

from abc import ABCMeta

# External module imports
from numbers import Number

# RL imports
from RL.utility.utility import errfmt
from RL.space.space import LinearSpace

standard_library.install_aliases()


class _DefaultCallOperator(object):
    """ Decorator class that adds a '_call'  method an 'Operator'

    The default implementation assumes that the 'range' of the
    operator implements 'element()'.
    """

    def _call(self, rhs):
        """ Default '_call' implementation using '_apply'.

        Implemented as:

        out = self.range.element()
        self._apply(rhs, out)
        return out

        Parameters
        ----------

        rhs : element in self.domain
              An object in the domain of this operator.
              This is the point that the operator should be applied in.

        Returns
        -------

        out : element in self.range
              An object in the range of this operator.
              The result of an operator evaluation.
        """

        out = self.range.element()
        self._apply(rhs, out)
        return out


class _DefaultApplyOperator(object):
    """ Decorator class that adds a '_apply' method to an 'Operator'

    The default implementation assumes that elements in the 'range' of the
    operator implements 'assign()'.
    """
    def _apply(self, rhs, out):
        """ Default '_apply' implementation using '_call'.

        Implemented as:

        out.assign(self._call(rhs))

        Parameters
        ----------
        rhs : element in self.domain
              An object in the domain of this operator.
              This is the point that the operator should be applied in.

        out : element in self.range
              An object in the range of this operator.
              The result of an operator evaluation.

        Returns
        -------
        None

        """

        out.assign(self._call(rhs))


class _OperatorMeta(ABCMeta):
    """ Metaclass used by Operator to ensure correct methods

    If an '_apply' method or '_call' method does not exist, it
    attempts to add a default implmented version. This only works
    if the range is a `LinearSpace`.
    """
    def __new__(mcs, name, bases, attrs):
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
        obj = type.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'domain'):
            raise NotImplementedError(errfmt('''
            'Operator' instances must have a 'domain' attribute'''))
        if not hasattr(obj, 'range'):
            raise NotImplementedError(errfmt('''
            'Operator' instances must have a 'range' attribute'''))
        if not hasattr(obj, '_call') and not hasattr(obj, '_apply'):
            raise NotImplementedError(errfmt('''
            'Operator' instances must have a '_call' and/or '_apply'
            attribute'''))

        return obj


class Operator(object):
    """Abstract operator

    An operator is a mapping from a 'Set' to another 'Set'.
    In RL all mappings know their domain and range, and have
    some method of evaluation. They also provide some convenience
    functions and error checking.

    Domain and Range
    ----------------

    A subclass of this has to have the attributes

    domain : AbstractSet
            The set this operator takes values from

    range : AbstractSet
            The set this operator takes values in

    Evaluation
    ----------

    It also needs to implement a method that evaluates the operator.
    There are two ways to do this, and any implementation needs
    to implement at least one of them.

    Out-of-place evaluation
    ~~~~~~~~~~~~~~~~~~~~~~~

    Out-of-place evaluation means the operator is applied and returns
    the result. In this case, a subclass has to implement

    _call(self, rhs)     <==>     result = operator(rhs)

    Where the arguments are

    rhs : domain element
          The point the operator should be evaluated in.

    With return value

    result : range element
             The result of the evaluation.

    In-place evaluation
    ~~~~~~~~~~~~~~~~~~~

    In-place evaluation means the operator is applied writes
    the result to a provided argument. This is for performance
    reasons. In this case, a subclass has to implement

    _apply(self, rhs, out)     <==>     out = operator(rhs)

    Where the arguments are

    rhs : domain element
          The point the operator should be evaluated in.

    out : range element
          The result of the evaluation.

    Notes
    -----
    If the user only provides one of '_apply' or '_call' and
    the underlying range is a 'LinearSpace', a default implementation
    of the other is provided.
    """

    __metaclass__ = _OperatorMeta

    def derivative(self, point):
        """ Get the derivative operator of this operator at `point`
        """
        raise NotImplementedError(errfmt('''
        Derivative not implemented for this operator ({})
        '''.format(self)))

    @property
    def inverse(self):
        """ Get the inverse operator of this operator
        """
        raise NotImplementedError(errfmt('''
        Inverse not implemented for this operator ({})
        '''.format(self)))

    @property
    def I(self):
        """ Get the inverse of this operator
        """
        return self.inverse

    # Implicitly defined operators
    def apply(self, rhs, out):
        """ Apply this operator in place.   Informally: out = f(rhs)

        Calls the underlying implementation '_apply' with error checking.

        Parameters
        ----------

        rhs : element in self.domain
              An object in the domain of this operator. This object is
              "constant", and will not be modified.
              This is the point that the operator should be applied in.

        out : element in self.range
              An object in the range of this operator. This object is
              "mutable", the result will be written to it. The result
              is independent of the state of this element.

        Returns
        -------
        None

        Example
        -------
        #TODO

        """

        if not self.domain.contains(rhs):
            raise TypeError(errfmt('''
            rhs ({}) is not in the domain of this operator ({})
            '''.format(repr(rhs), repr(self))))

        if not self.range.contains(out):
            raise TypeError(errfmt('''
            out ({}) is not in the range of this operator ({})
            '''.format(repr(out), repr(self))))

        if rhs is out:
            raise ValueError(errfmt('''
            rhs ({}) is the same as out ({}) operators do not permit
            aliased arguments
            '''.format(repr(rhs), repr(out))))

        self._apply(rhs, out)

    def __call__(self, rhs):
        """ Evaluates the operator. The output element is allocated
        dynamically.

        Calls the underlying implementation '_call' with error checking.

        Parameters
        ----------
        rhs : element in self.domain
              An object in the domain of this operator. This object is
              "constant", and will not be modified.
              This is the point that the operator should be applied in.

        Returns
        -------
        element in self.range
            The result of evaluating the operator.

        Example
        -------

        >>> from default_operators import operator
        >>> A = operator(lambda x: 3*x)
        >>> A(3)
        9
        >>> A.__call__(5)
        15
        """

        if not self.domain.contains(rhs):
            raise TypeError(errfmt('''
            rhs ({}) is not in the domain of this operator ({})
            '''.format(repr(rhs), repr(self))))

        result = self._call(rhs)

        if not self.range.contains(result):
            raise TypeError(errfmt('''
            result ({}) is not in the domain of this operator ({})
            '''.format(repr(result), repr(self))))

        return result

    def __add__(self, other):
        """ Operator addition (A+B)(x) = A(x) + B(x)
        """

        return OperatorSum(self, other)

    def __mul__(self, other):
        """Right multiplication of operators with scalars
        (A*a)(x) = A(a*x)

        Note that left and right multiplication of operators is
        different.

        Parameters
        ----------
        other : Operator
                Operator with same domain and range as this operator

        Returns
        -------
        OperatorRightScalarMultiplication instance

        Example
        -------

        >>> from default_operators import operator
        >>> A = operator(lambda x: 3*x)
        >>> Scaled = A*3
        >>> Scaled(5)
        45
        """

        return OperatorRightScalarMultiplication(self, other)

    def __rmul__(self, other):
        """ Left multiplication of operators with scalars
        (a*A)(x) = a*A(x)

        Note that left and right multiplication of operators is
        different.

        Parameters
        ----------
        other : Operator
                Operator with same domain and range as this operator

        Returns
        -------
        OperatorLeftScalarMultiplication instance

        Example
        -------

        >>> from default_operators import operator
        >>> A = operator(lambda x: 3*x)
        >>> Scaled = 3*A
        >>> Scaled(5)
        45
        """

        return OperatorLeftScalarMultiplication(self, other)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return (self.__class__.__name__ + ": " + str(self.domain) +
                "->" + str(self.range))


class OperatorSum(Operator):
    """ Expression type for the sum of operators.

    Sum is only well defined for Operators between LinearSpace:s

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

    def _call(self, rhs):
        """ Calculates op1(rhs) + op2(rhs)

        Parameters
        ----------
        rhs : self.domain element
              The point to evaluate the sum in

        Returns
        -------
        result : self.range element
                 Result of the evaluation

        Example
        -------
        >>> from default_operators import operator
        >>> A = operator(lambda x: 3*x)
        >>> B = operator(lambda x: 5*x)
        >>> OperatorSum(A, B)(3)
        24
        """
        return self._op1._call(rhs) + self._op2._call(rhs)

    def _apply(self, rhs, out):
        """
        Calculates op1(rhs) + op2(rhs)

        Parameters
        ----------
        rhs : self.domain element
              The point to evaluate the sum in
        out : self.range element
              Object to store the result in

        Returns
        -------
        None

        Example
        -------
        >>> from RL.space.euclidean import RN
        >>> from RL.operator.default_operators import IdentityOperator
        >>> r3 = RN(3)
        >>> op = IdentityOperator(r3)
        >>> rhs = r3.element([1, 2, 3])
        >>> out = r3.element()
        >>> OperatorSum(op, op).apply(rhs, out)
        >>> out
        RN(3).element([ 2.,  4.,  6.])
        """
        tmp = self._tmp if self._tmp is not None else self.range.element()
        self._op1._apply(rhs, out)
        self._op2._apply(rhs, tmp)
        out += tmp

    @property
    def domain(self):
        """
        Get the domain of this operator

        Parameters
        ----------
        None

        Returns
        -------
        domain : Set
                 The domain of the operator

        Example
        -------
        >>> from RL.space.euclidean import RN
        >>> from RL.operator.default_operators import IdentityOperator
        >>> r3 = RN(3)
        >>> op = IdentityOperator(r3)
        >>> OperatorSum(op, op).domain
        RN(3)
        """
        return self._op1.domain

    @property
    def range(self):
        """
        Get the range of this operator

        Parameters
        ----------
        None

        Returns
        -------
        domain : Set
                 The domain of the operator

        Example
        -------
        >>> from RL.space.euclidean import RN
        >>> from RL.operator.default_operators import IdentityOperator
        >>> r3 = RN(3)
        >>> op = IdentityOperator(r3)
        >>> OperatorSum(op, op).range
        RN(3)
        """
        return self._op1.range

    def derivative(self, point):
        return LinearOperatorSum(self._op1.derivative(point),
                                 self._op2.derivative(point))

    def __repr__(self):
        return 'OperatorSum( ' + repr(self._op1) + ", " + repr(self._op2) + ')'

    def __str__(self):
        return '(' + str(self._op1) + ' + ' + str(self._op2) + ')'


class OperatorComposition(Operator):
    """Expression type for the composition of operators

    OperatorComposition(left, right)(x) = left(right(x))

    Parameters
    ----------

    left : Operator
           The first operator
    right : Operator
            Operator with the same domain and range as op1
    tmp : Element in the range of this operator
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

    def _call(self, rhs):
        return self._left._call(self._right._call(rhs))

    def _apply(self, rhs, out):
        tmp = (self._tmp if self._tmp is not None
               else self._right.range.element())
        self._right._apply(rhs, tmp)
        self._left._apply(tmp, out)

    @property
    def domain(self):
        return self._right.domain

    @property
    def range(self):
        return self._left.range

    @property
    def inverse(self):
        return OperatorComposition(self._right.inverse,
                                   self._left.inverse,
                                   self._tmp)

    def derivative(self, point):
        left_deriv = self._left.derivative(self._right(point))
        right_deriv = self._right.derivative(point)

        return LinearOperatorComposition(left_deriv,
                                         right_deriv)

    def __repr__(self):
        return ('OperatorComposition( ' + repr(self._left) + ', ' +
                repr(self._right) + ')')

    def __str__(self):
        return str(self._left) + ' o ' + str(self._right)


class OperatorPointwiseProduct(Operator):
    """Pointwise multiplication of operators defined on Banach Algebras
    (with pointwise multiplication)

    OperatorPointwiseProduct(op1, op2)(x) = op1(x) * op2(x)
    """

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

    def _call(self, rhs):
        return self._op1._call(rhs) * self._op2._call(rhs)

    def _apply(self, rhs, out):
        tmp = self._op2.range.element()
        self._op1._apply(rhs, out)
        self._op2._apply(rhs, tmp)
        out *= tmp

    @property
    def domain(self):
        return self._op1.domain

    @property
    def range(self):
        return self._op1.range


class OperatorLeftScalarMultiplication(Operator):
    """Expression type for the left multiplication of operators with
    scalars

    OperatorLeftScalarMultiplication(op, scalar)(x) = scalar * op(x)
    """

    def __init__(self, op, scalar):
        if (isinstance(op.range, LinearSpace)
            and not op.range.field.contains(scalar)):
            raise TypeError(errfmt('''
            'scalar' ({}) not compatible with field of range ({}) of 'op'
            '''.format(scalar, op.range.field)))

        self._op = op
        self._scalar = scalar

    def _call(self, rhs):
        return self._scalar * self._op._call(rhs)

    def _apply(self, rhs, out):
        self._op._apply(rhs, out)
        out *= self._scalar

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range

    @property
    def inverse(self):
        return OperatorRightScalarMultiplication(self._op.inverse,
                                                 1.0/self._scalar)

    def derivative(self, point):
        return LinearOperatorScalarMultiplication(self._op.derivative(point),
                                                  self._scalar)

    def __repr__(self):
        return ('OperatorLeftScalarMultiplication( ' + repr(self._op) +
                ', ' + repr(self._scalar) + ')')

    def __str__(self):
        return str(self._scalar) + " * " + str(self._op)


class OperatorRightScalarMultiplication(Operator):
    """Expression type for the right multiplication of operators with
    scalars.

    OperatorRightScalarMultiplication(op, scalar)(x) = op(scalar * x)

    Typically slower than left multiplication since this requires a
    copy.

    Parameters
    ----------

    op : Operator
        Any operator whose range supports `*= scalar`
    scalar : Number
        An element in the field of the domain of op
    tmp : Element in the range of this operator
        Used to avoid the creation of a
        temporary when applying the operator.
    """

    def __init__(self, op, scalar, tmp=None):
        if (isinstance(op.domain, LinearSpace) and
            not op.domain.field.contains(scalar)):
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

    def _call(self, rhs):
        return self._op._call(self._scalar * rhs)

    def _apply(self, rhs, out):
        tmp = self._tmp if self._tmp is not None else self.domain.element()
        tmp.lincomb(self._scalar, rhs)
        self._op._apply(tmp, out)

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range

    @property
    def inverse(self):
        return OperatorLeftScalarMultiplication(self._op.inverse,
                                                1.0/self._scalar)

    def derivative(self, point):
        return LinearOperatorScalarMultiplication(self._op.derivative(point),
                                                  self._scalar)

    def __repr__(self):
        return ('OperatorRightScalarMultiplication( ' + self._op.__repr__() +
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

        op.T.apply(rhs, out) = op.adjoint.apply(rhs,out)
        and
        op.T.adjoint.apply(rhs, out) = op.apply(rhs,out)
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
            return LinearOperatorScalarMultiplication(self, other)
        else:
            raise TypeError('Expected an operator or a scalar')

    __rmul__ = __mul__


class SelfAdjointOperator(LinearOperator):
    """ Special case of self adjoint operators where A(x) = A.T(x)
    """

    @property
    def adjoint(self):
        return self

    @property
    def range(self):
        return self.domain


class LinearOperatorSum(OperatorSum, LinearOperator):
    """Expression type for the sum of linear operators
    """
    def __init__(self, op1, op2, tmp_ran=None, tmp_dom=None):
        """Create the abstract operator sum defined by:

        LinearOperatorSum(op1, op2)(x) = op1(x) + op2(x)

        Args:
            LinearOperator  `op1`      The first operator
            LinearOperator  `op2`      The second operator
            Vector          `tmp_ran`  A vector in the domain of this operator.
                                       Used to avoid the creation of a
                                       temporary when applying the operator.
            Vector          `tmp_dom`  A vector in the range of this operator.
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


class LinearOperatorComposition(OperatorComposition, LinearOperator):
    """Expression type for the composition of linear operators
    """

    def __init__(self, left, right, tmp=None):
        """ Create the abstract operator composition defined by:

        LinearOperatorComposition(left, right)(x) = left(right(x))

        With adjoint defined by
        LinearOperatorComposition(left, right).T(y) = right.T(left.T(y))

        Args:
            LinearOperator  `left`      The first operator
            LinearOperator  `right`     The second operator
            Vector          `tmp`       A vector in the range of `right`.
                                        Used to avoid the creation of a
                                        temporary when applying the operator.
        """

        if not isinstance(left, LinearOperator):
            raise TypeError(errfmt('''
            left ({}) is not a LinearOperator. LinearOperatorComposition is
            only defined for LinearOperators'''.format(left)))
        if not isinstance(right, LinearOperator):
            raise TypeError(errfmt('''
            right ({}) is not a LinearOperator. LinearOperatorComposition is
            only defined for LinearOperators'''.format(right)))

        super().__init__(left, right, tmp)

    @property
    def adjoint(self):
        return LinearOperatorComposition(self._right.adjoint,
                                         self._left.adjoint, self._tmp)


class LinearOperatorScalarMultiplication(OperatorLeftScalarMultiplication,
                                         LinearOperator):
    """Expression type for the multiplication of operators with scalars
    """

    def __init__(self, op, scalar):
        """ Create the LinearOperatorScalarMultiplication defined by

        LinearOperatorScalarMultiplication(op, scalar)(x) = scalar * op(x)

        Args:
            Operator    `op`        Any operator
            Scalar      `scalar`    An element in the field of the domain
                                    of `op`
        """
        if not isinstance(op, LinearOperator):
            raise TypeError(errfmt('''
            op ({}) is not a LinearOperator.
            LinearOperatorScalarMultiplication is only defined for
            LinearOperators'''.format(op)))

        super().__init__(op, scalar)

    @property
    def adjoint(self):
        return LinearOperatorScalarMultiplication(self._op.adjoint,
                                                  self._scalar)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
