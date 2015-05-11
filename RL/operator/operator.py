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


# Imports for common Python 2/3 codebase
from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
try:
    from builtins import str, object, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, object, super
from future.utils import with_metaclass
from future import standard_library

# External module imports
from numbers import Number
from abc import ABCMeta, abstractmethod, abstractproperty

# RL imports
from RL.utility.utility import errfmt

standard_library.install_aliases()


class Operator(with_metaclass(ABCMeta, object)):
    """Abstract operator
    """

    @abstractmethod
    def applyImpl(self, rhs, out):
        """Apply the operator.

        This method is not intended to be called by external users.
        It is intended that classes that derive from Operator derive
        from this method.

        Parameters
        ----------

        rhs : element in self.domain
              An object in the domain of this operator. This object is
              "constant", and must not be modified.
              This is the point that the operator should be applied in.

        out : element in self.range
              An object in the range of this operator. This object is
              "mutable", the result should be written to it. The result
              must not depend on the initial state of this element.

        Returns
        -------
        None
        """

    @abstractproperty
    def domain(self):
        """Get the domain of the operator.

        The domain of an operator is expected to derive from
        RL.space.set.Set
        """

    @abstractproperty
    def range(self):
        """Get the range of the operator.

        The range of an operator is expected to derive from
        RL.space.set.Set
        """

    def getDerivative(self, point):
        """ Get the derivative operator of this operator at `point`
        """
        raise NotImplementedError(errfmt('''
        getDerivative not implemented for this operator ({})
        '''.format(self)))

    # Implicitly defined operators
    def apply(self, rhs, out):
        """ Apply this operator in place.   Informally: out = f(rhs)

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

        >>> rn = RN(3)
        >>> Op = IdentityOperator(rn)
        >>> x = rn.makeVector([1, 2, 3])
        >>> y = rn.empty()
        >>> Op.apply(x, y)
        >>> y
        [1.0, 2.0, 3.0]
        """

        if not self.domain.contains(rhs):
            raise TypeError(errfmt('''
            rhs ({}) is not in the domain of this operator ({})
            '''.format(rhs, self)))

        if not self.range.contains(out):
            raise TypeError(errfmt('''
            out ({}) is not in the range of this operator ({})
            '''.format(out, self)))

        if rhs is out:
            raise ValueError(errfmt('''
            rhs ({}) is the same as out ({}) operators do not permit
            aliased arguments
            '''.format(rhs, out)))

        self.applyImpl(rhs, out)

    def __call__(self, rhs):
        """ Evaluates the operator. The output element is allocated
        dynamically.

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

        >>> rn = RN(3)
        >>> Op = IdentityOperator(rn)
        >>> x = rn.makeVector([1, 2, 3])
        >>> Op(x)
        [1.0, 2.0, 3.0]
        """

        tmp = self.range.empty()
        self.apply(rhs, tmp)
        return tmp

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

        >>> rn = RN(3)
        >>> Op = IdentityOperator(rn)
        >>> x = rn.makeVector([1, 2, 3])
        >>> Op(x)
        [1.0, 2.0, 3.0]
        >>> Scaled = Op * 3
        >>> Scaled(x)
        [3.0, 6.0, 9.0]
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

        >>> rn = RN(3)
        >>> Op = IdentityOperator(rn)
        >>> x = rn.makeVector([1, 2, 3])
        >>> Op(x)
        [1.0, 2.0, 3.0]
        >>> Scaled = 3 * Op
        >>> Scaled(x)
        [3.0, 6.0, 9.0]
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
        self.tmp = tmp

    def applyImpl(self, rhs, out):
        tmp = self.tmp if self.tmp is not None else self.range.empty()
        self._op1.applyImpl(rhs, out)
        self._op2.applyImpl(rhs, tmp)
        out += tmp

    @property
    def domain(self):
        return self._op1.domain

    @property
    def range(self):
        return self._op1.range

    def __repr__(self):
        return 'OperatorSum( ' + repr(self._op1) + ", " + repr(self._op2) + ')'

    def __str__(self):
        return '(' + str(self._op1) + ' + ' + str(self._op2) + ')'


class OperatorComposition(Operator):
    """Expression type for the composition of operators

    OperatorComposition(left, right)(x) = left(right(x))

    Parameters
    ----------

    op1 : Operator
          The first operator
    op2 : Operator
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

    def applyImpl(self, rhs, out):
        tmp = self._tmp if self._tmp is not None else self._right.range.empty()
        self._right.applyImpl(rhs, tmp)
        self._left.applyImpl(tmp, out)

    @property
    def domain(self):
        return self._right.domain

    @property
    def range(self):
        return self._left.range

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

    def applyImpl(self, rhs, out):
        tmp = self._op2.range.empty()
        self._op1.applyImpl(rhs, out)
        self._op2.applyImpl(rhs, tmp)
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
        if not op.range.field.contains(scalar):
            raise TypeError(errfmt('''
            Scalar ({}) not compatible with field of range ({}) of operator
            '''.format(scalar, op.range.field)))

        self._op = op
        self._scalar = scalar

    def applyImpl(self, rhs, out):
        self._op.applyImpl(rhs, out)
        out *= self._scalar

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range

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
        if not op.domain.field.contains(scalar):
            raise TypeError(errfmt('''
            Scalar ({}) not compatible with field of domain ({}) of operator
            '''.format(scalar, op.domain.field)))

        if tmp is not None and not op.domain.contains(tmp):
            raise TypeError(errfmt('''
            Tmp ({}) must be an element in the domain of the operator ({}).
            '''.format(tmp, op.domain)))

        self._op = op
        self._scalar = scalar
        self._tmp = tmp

    def applyImpl(self, rhs, out):
        tmp = self._tmp if self._tmp is not None else self.domain.empty()
        tmp.linComb(self._scalar, rhs)
        self._op.applyImpl(tmp, out)

    @property
    def domain(self):
        return self._op.domain

    @property
    def range(self):
        return self._op.range

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

    @abstractmethod
    def applyAdjointImpl(self, rhs, out):
        """Apply the adjoint of the operator. Abstract, should be
        implemented by subclasses.

        Public callers should instead use applyAdjoint which provides
        type checking.
        """

    # Implicitly defined operators
    @property
    def T(self):
        """ Get the adjoint of this operator such that:

        op.T.apply(rhs, out) = op.applyAdjoint(rhs,out)
        and
        op.T.applyAdjoint(rhs, out) = op.apply(rhs,out)
        """
        return OperatorAdjoint(self)

    def getDerivative(self, point):
        """ The derivative of linear operators is the operator itself
        """
        return self

    def applyAdjoint(self, rhs, out):
        """ Applies the adjoint of the operator, informally:
        out = op(rhs)


        Parameters
        ----------
        rhs : Vector
              A vector in the range of this operator.
              The point the adjoint should be evaluated in.

        out : Vector
              A vector in the domain of this operator.
              The result of the evaluation is written to this
              vector. Any previous content is overwritten.
        """
        if not self.range.contains(rhs):
            raise TypeError(errfmt('''
            rhs ({}) is not in the domain of this operators ({}) adjoint
            '''.format(rhs, self)))
        if not self.domain.contains(out):
            raise TypeError(errfmt('''
            out ({}) is not in the range of this operators ({}) adjoint
            '''.format(out, self)))
        if rhs is out:
            raise ValueError(errfmt('''
            rhs ({}) is the same as out ({}). Operators do not permit aliased
            arguments'''.format(rhs, out)))

        self.applyAdjointImpl(rhs, out)

    def __add__(self, other):
        """Operator addition

        (self + other)(x) = self(x) + other(x)
        """

        if isinstance(other, LinearOperator):  # Special if both are linear
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


class SelfAdjointOperator(with_metaclass(ABCMeta, LinearOperator)):
    """ Special case of self adjoint operators where A(x) = A.T(x)
    """

    def applyAdjointImpl(self, rhs, out):
        self.applyImpl(rhs, out)

    @property
    def range(self):
        return self.domain


class OperatorAdjoint(LinearOperator):
    """Expression type for the adjoint of an operator.
    """

    def __init__(self, op):
        """Create an operator that is the adjoint of `op`

        It is defined by:
        OperatorAdjoint(op).apply(rhs,out) = op.applyAdjoint(rhs,out)
        and
        OperatorAdjoint(op).applyAdjoint(rhs,out) = op.apply(rhs,out)
        """

        if not isinstance(op, LinearOperator):
            raise TypeError(errfmt('''
            Operator ({}) is not a LinearOperator. OperatorAdjoint is only
            defined for LinearOperators'''.format(op)))

        self._op = op

    def applyImpl(self, rhs, out):
        self._op.applyAdjointImpl(rhs, out)

    def applyAdjointImpl(self, rhs, out):
        self._op.applyImpl(rhs, out)

    @property
    def domain(self):
        return self._op.range

    @property
    def range(self):
        return self._op.domain

    def __repr__(self):
        return 'OperatorAdjoint( ' + repr(self._op) + ')'

    def __str__(self):
        return str(self._op) + "^T"


class LinearOperatorSum(OperatorSum, LinearOperator):
    """Expression type for the sum of linear operators
    """
    def __init__(self, op1, op2, tmpRan=None, tmpDom=None):
        """Create the abstract operator sum defined by:

        LinearOperatorSum(op1, op2)(x) = op1(x) + op2(x)

        Args:
            LinearOperator  `op1`      The first operator
            LinearOperator  `op2`      The second operator
            Vector          `tmpRan`   A vector in the domain of this operator.
                                       Used to avoid the creation of a
                                       temporary when applying the operator.
            Vector          `tmpDom`   A vector in the range of this operator.
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

        super().__init__(op1, op2, tmpRan)
        self._tmpDom = tmpDom

    def applyAdjointImpl(self, rhs, out):
        tmp = self._tmpDom if self._tmpDom is not None else self.domain.empty()
        self._op1.applyAdjointImpl(rhs, out)
        self._op2.applyAdjointImpl(rhs, tmp)
        out += tmp


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

        OperatorComposition.__init__(self, left, right, tmp)

    def applyAdjointImpl(self, rhs, out):
        tmp = self._tmp if self._tmp is not None else self._right.range.empty()
        self._left.applyAdjoint(rhs, tmp)
        self._right.applyAdjoint(tmp, out)


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

    def applyAdjointImpl(self, rhs, out):
        self._op.applyAdjointImpl(rhs, out)
        out *= self._scalar
