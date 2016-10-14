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

"""Default operators defined on any `ProductSpace`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np
import scipy as sp
from numbers import Integral

from odl.operator.operator import Operator
from odl.space import ProductSpace


__all__ = ('ProductSpaceOperator',
           'ComponentProjection', 'ComponentProjectionAdjoint',
           'BroadcastOperator', 'ReductionOperator', 'DiagonalOperator')


class ProductSpaceOperator(Operator):

    """A "matrix of operators" on product spaces.

    For example a matrix of operators can act on a vector by

        ``ProductSpaceOperator([[A, B], [C, D]])([x, y]) =
        [A(x) + B(y), C(x) + D(y)]``

    Notes
    -----
    This is intended for the case where an operator can be decomposed
    as a linear combination of "sub-operators", e.g.

    .. math::

        \\left(
        \\begin{array}{ccc}
        A & B & 0 \\\\
        0 & C & 0 \\\\
        0 & 0 & D
        \end{array}\\right)
        \\left(
        \\begin{array}{c}
        x \\\\
        y \\\\
        z
        \end{array}\\right)
        =
        \\left(
        \\begin{array}{c}
        A(x) + B(y) \\\\
        C(y) \\\\
        D(z)
        \end{array}\\right)

    Mathematically, a `ProductSpaceOperator` is an operator

        :math:`\mathcal{A}: \mathcal{X} \\to \mathcal{Y}`

    between product spaces
    :math:`\mathcal{X}=\mathcal{X}_1 \\times\dots\\times \mathcal{X}_m`
    and
    :math:`\mathcal{Y}=\mathcal{Y}_1 \\times\dots\\times \mathcal{Y}_n`
    which can be written in the form

        :math:`\mathcal{A} = (\mathcal{A}_{ij})_{i,j},  \quad
        i = 1, \dots, n, \\ j = 1, \dots, m`

    with *component operators*
    :math:`\mathcal{A}_{ij}: \mathcal{X}_j \\to \mathcal{Y}_i`.

    Its action on a vector :math:`x = (x_1, \dots, x_m)` is defined as
    the matrix multiplication

        :math:`[\mathcal{A}(x)]_i = \sum_{j=1}^m \mathcal{A}_{ij}(x_j)`.

    See Also
    --------
    BroadcastOperator : Case when a single argument is used by several ops.
    ReductionOperator : Calculates sum of operator results.
    DiagonalOperator : Case where the 'matrix' is diagonal.
    """

    def __init__(self, operators, domain=None, range=None):
        """Initialize a new instance.

        Parameters
        ----------
        operators : `array-like`
            An array of `Operator`'s
        domain : `ProductSpace`, optional
            Domain of the operator. If not provided, it is tried to be
            inferred from the operators. This requires each **column**
            to contain at least one operator.
        range : `ProductSpace`, optional
            Range of the operator. If not provided, it is tried to be
            inferred from the operators. This requires each **row**
            to contain at least one operator.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> X = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = X.element([[1, 2, 3], [4, 5, 6]])

        Sum of elements:

        >>> prod_op = ProductSpaceOperator([I, I])
        >>> prod_op(x)
        ProductSpace(rn(3), 1).element([
            [5.0, 7.0, 9.0]
        ])

        Diagonal operator -- 0 or ``None`` means ignore, or the implicit
        zero operator:

        >>> prod_op = ProductSpaceOperator([[I, 0], [0, I]])
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        Complicated combinations:

        >>> prod_op = ProductSpaceOperator([[I, I], [I, 0]])
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [5.0, 7.0, 9.0],
            [1.0, 2.0, 3.0]
        ])
        """

        # Validate input data
        if domain is not None:
            if not isinstance(domain, ProductSpace):
                raise TypeError('`domain` {!r} not a ProductSpace instance'
                                ''.format(domain))
            if domain.is_weighted:
                raise NotImplementedError('weighted spaces not supported')

        if range is not None:
            if not isinstance(range, ProductSpace):
                raise TypeError('`range` {!r} not a ProductSpace instance'
                                ''.format(range))
            if range.is_weighted:
                raise NotImplementedError('weighted spaces not supported')

        # Convert ops to sparse representation
        self.ops = sp.sparse.coo_matrix(operators)

        if not all(isinstance(op, Operator) for op in self.ops.data):
            raise TypeError('`operators` {!r} must be a matrix of Operators'
                            ''.format(operators))

        # Set domain and range (or verify if given)
        if domain is None:
            domains = [None] * self.ops.shape[1]
        else:
            domains = domain

        if range is None:
            ranges = [None] * self.ops.shape[0]
        else:
            ranges = range

        for row, col, op in zip(self.ops.row, self.ops.col, self.ops.data):
            if domains[col] is None:
                domains[col] = op.domain
            elif domains[col] != op.domain:
                raise ValueError('column {}, has inconsistent domains, '
                                 'got {} and {}'
                                 ''.format(col, domains[col], op.domain))

            if ranges[row] is None:
                ranges[row] = op.range
            elif ranges[row] != op.range:
                raise ValueError('row {}, has inconsistent ranges, '
                                 'got {} and {}'
                                 ''.format(row, ranges[row], op.range))

        if domain is None:
            for col, sub_domain in enumerate(domains):
                if sub_domain is None:
                    raise ValueError('col {} empty, unable to determine '
                                     'domain, please use `domain` parameter'
                                     ''.format(col))

            domain = ProductSpace(*domains)

        if range is None:
            for row, sub_range in enumerate(ranges):
                if sub_range is None:
                    raise ValueError('row {} empty, unable to determine '
                                     'range, please use `range` parameter'
                                     ''.format(row))

            range = ProductSpace(*ranges)

        # Set linearity
        linear = all(op.is_linear for op in self.ops.data)

        super().__init__(domain=domain, range=range, linear=linear)

    def _call(self, x, out=None):
        """Call the operators on the parts of ``x``."""
        # TODO: add optimization in case an operator appears repeatedly in a
        # row
        if out is None:
            out = self.range.zero()
            for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
                out[i] += op(x[j])
        else:
            has_evaluated_row = np.zeros(self.range.size, dtype=bool)
            for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
                if not has_evaluated_row[i]:
                    op(x[j], out=out[i])
                else:
                    # TODO: optimize
                    out[i] += op(x[j])

                has_evaluated_row[i] = True

            for i, evaluated in enumerate(has_evaluated_row):
                if not evaluated:
                    out[i].set_zero()

        return out

    def derivative(self, x):
        """Derivative of the product space operator.

        Parameters
        ----------
        x : `domain` element
            The point to take the derivative in

        Returns
        -------
        adjoint : linear`ProductSpaceOperator`
            The derivative

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> X = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = X.element([[1, 2, 3], [4, 5, 6]])

        Example with linear operator (derivative is itself)

        >>> prod_op = ProductSpaceOperator([[0, I], [0, 0]],
        ...                                domain=X, range=X)
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]
        ])
        >>> prod_op.derivative(x)(x)
        ProductSpace(rn(3), 2).element([
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]
        ])

        Example with affine operator

        >>> residual_op = I - r3.element([1, 1, 1])
        >>> op = ProductSpaceOperator([[0, residual_op], [0, 0]],
        ...                           domain=X, range=X)

        Calling operator gives offset by [1, 1, 1]

        >>> op(x)
        ProductSpace(rn(3), 2).element([
            [3.0, 4.0, 5.0],
            [0.0, 0.0, 0.0]
        ])

        Derivative of affine operator does not have this offset

        >>> op.derivative(x)(x)
        ProductSpace(rn(3), 2).element([
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]
        ])
        """
        deriv_ops = [op.derivative(x[col]) for op, col in zip(self.ops.data,
                                                              self.ops.col)]
        indices = [self.ops.row, self.ops.col]
        shape = self.ops.shape
        deriv_matrix = sp.sparse.coo_matrix((deriv_ops, indices), shape)
        return ProductSpaceOperator(deriv_matrix, self.domain, self.range)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        The adjoint is given by taking the transpose of the matrix
        and the adjoint of each component operator.

        In weighted product spaces, the adjoint needs to take the
        weightings into account. This is currently not supported.

        Returns
        -------
        adjoint : `ProductSpaceOperator`
            The adjoint

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> X = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = X.element([[1, 2, 3], [4, 5, 6]])

        Matrix is transposed:

        >>> prod_op = ProductSpaceOperator([[0, I], [0, 0]],
        ...                                domain=X, range=X)
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]
        ])
        >>> prod_op.adjoint(x)
        ProductSpace(rn(3), 2).element([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0]
        ])
        """
        adjoint_ops = [op.adjoint for op in self.ops.data]
        indices = [self.ops.col, self.ops.row]  # Swap col/row -> transpose
        shape = (self.ops.shape[1], self.ops.shape[0])
        adj_matrix = sp.sparse.coo_matrix((adjoint_ops, indices), shape)
        return ProductSpaceOperator(adj_matrix, self.range, self.domain)

    def __repr__(self):
        """Return ``repr(self)``."""
        return 'ProductSpaceOperator({!r})'.format(self.ops)


class ComponentProjection(Operator):

    """Projection onto the subspace identified by an index.

    For a product space :math:`\mathcal{X} = \mathcal{X}_1 \\times \dots
    \\times \mathcal{X}_n`, the component projection

        :math:`\mathcal{P}_i: \mathcal{X} \\to \mathcal{X}_i`

    is given by :math:`\mathcal{P}_i(x) = x_i` for an element
    :math:`x = (x_1, \dots, x_n) \\in \mathcal{X}`.

    More generally, for an index set :math:`I \subset \{1, \dots, n\}`,
    the projection operator :math:`\mathcal{P}_I` is defined by
    :math:`\mathcal{P}_I(x) = (x_i)_{i \\in I}`.

    Note that this is a special case of a product space operator where
    the "operator matrix" has only one row and contains only
    identity operators.
    """

    def __init__(self, space, index):
        """Initialize a new instance.

        Parameters
        ----------
        space : `ProductSpace`
            Space to project from.
        index : int, slice, or iterable
            Indices defining the subspace. If ``index`` is not an integer,
            the `Operator.range` of this operator is also a `ProductSpace`.

        Examples
        --------
        >>> r1 = odl.rn(1)
        >>> r2 = odl.rn(2)
        >>> r3 = odl.rn(3)
        >>> X = odl.ProductSpace(r1, r2, r3)

        Projection on n-th component

        >>> proj = odl.ComponentProjection(X, 0)
        >>> x = [[1.0],
        ...      [2.0, 3.0],
        ...      [4.0, 5.0, 6.0]]
        >>> proj(x)
        rn(1).element([1.0])

        Projection on sub-space

        >>> proj = odl.ComponentProjection(X, [0, 2])
        >>> proj(x)
        ProductSpace(rn(1), rn(3)).element([
            [1.0],
            [4.0, 5.0, 6.0]
        ])
        """
        self.__index = index
        super().__init__(space, space[index], linear=True)

    @property
    def index(self):
        """Index of the subspace."""
        return self.__index

    def _call(self, x, out=None):
        """Project ``x`` onto the subspace."""
        if out is None:
            out = x[self.index].copy()
        else:
            out.assign(x[self.index])
        return out

    @property
    def adjoint(self):
        """Return the adjoint operator.

        The adjoint is given by extending along `ComponentProjection.index`,
        and setting zero along the others.

        See Also
        --------
        ComponentProjectionAdjoint
        """
        return ComponentProjectionAdjoint(self.domain, self.index)


class ComponentProjectionAdjoint(Operator):

    """Adjoint operator to `ComponentProjection`.

    As a special case of the adjoint of a `ProductSpaceOperator`,
    this operator is given as a column vector of identity operators
    and zero operators, with the identities placed in the positions
    defined by `ComponentProjectionAdjoint.index`.

    In weighted product spaces, the adjoint needs to take the
    weightings into account. This is currently not supported.
    """

    def __init__(self, space, index):
        """Initialize a new instance

        Parameters
        ----------
        space : `ProductSpace`
            Space to project to.
        index : int, slice, or iterable
            Indexes to project from.

        Examples
        --------
        >>> r1 = odl.rn(1)
        >>> r2 = odl.rn(2)
        >>> r3 = odl.rn(3)
        >>> X = odl.ProductSpace(r1, r2, r3)
        >>> x = X.element([[1], [2, 3], [4, 5, 6]])

        Projection on the 0-th component:

        >>> proj = odl.ComponentProjectionAdjoint(X, 0)
        >>> proj(x[0])
        ProductSpace(rn(1), rn(2), rn(3)).element([
            [1.0],
            [0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        Projection on a sub-space corresponding to indices 0 and 2:

        >>> proj = odl.ComponentProjectionAdjoint(X, [0, 2])
        >>> proj(x[0, 2])
        ProductSpace(rn(1), rn(2), rn(3)).element([
            [1.0],
            [0.0, 0.0],
            [4.0, 5.0, 6.0]
        ])
        """
        self.__index = index
        super().__init__(space[index], space, linear=True)

    @property
    def index(self):
        """Index of the subspace."""
        return self.__index

    def _call(self, x, out=None):
        """Extend ``x`` from the subspace."""
        if out is None:
            out = self.range.zero()
        else:
            out.set_zero()

        out[self.index] = x
        return out

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `ComponentProjection`
            The adjoint is given by the `ComponentProjection` related to this
            operator's `index`.
        """
        return ComponentProjection(self.range, self.index)


class BroadcastOperator(Operator):
    """Broadcast argument to set of operators.

    An argument is broadcast by evaluating several operators in the same
    point::

        BroadcastOperator(op1, op2)(x) = [op1(x), op2(x)]

    See Also
    --------
    ProductSpaceOperator : More general case, used as backend.
    ReductionOperator : Calculates sum of operator results.
    DiagonalOperator : Case where each operator should have its own argument.
    """
    def __init__(self, *operators):
        """Initialize a new instance

        Parameters
        ----------
        operator1,...,operatorN : `Operator` or `int`
            The individual operators that should be evaluated.
            Can also be given as ``operator, n`` with ``n`` integer,
            in which case ``operator`` is repeated ``n`` times.

        Examples
        --------
        Initialize an operator:

        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> op = BroadcastOperator(I, 2 * I)
        >>> op.domain
        rn(3)
        >>> op.range
        ProductSpace(rn(3), 2)

        Evaluate the operator:

        >>> x = [1, 2, 3]
        >>> op(x)
        ProductSpace(rn(3), 2).element([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
        ])

        Can also initialize by calling an operator repeatedly:

        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> op = BroadcastOperator(I, 2)
        >>> op.operators
        (IdentityOperator(rn(3)), IdentityOperator(rn(3)))
        """
        if (len(operators) == 2 and
                isinstance(operators[0], Operator) and
                isinstance(operators[1], Integral)):
            operators = (operators[0],) * operators[1]

        self.__operators = operators
        self.__prod_op = ProductSpaceOperator([[op] for op in operators])

        super().__init__(self.prod_op.domain[0],
                         self.prod_op.range,
                         self.prod_op.is_linear)

    @property
    def prod_op(self):
        """`ProductSpaceOperator` implementation."""
        return self.__prod_op

    @property
    def operators(self):
        """Tuple of sub-operators that comprise ``self``."""
        return self.__operators

    def __getitem__(self, index):
        """Return an operator by index."""
        return self.operators[index]

    def _call(self, x, out=None):
        """Evaluate all operators in ``x`` and broadcast."""
        wrapped_x = self.prod_op.domain.element([x], cast=False)
        return self.prod_op(wrapped_x, out=out)

    def derivative(self, x):
        """Derivative of the broadcast operator.

        Parameters
        ----------
        x : `domain` element
            The point to take the derivative in

        Returns
        -------
        adjoint : linear `BroadcastOperator`
            The derivative

        Examples
        --------
        Example with an affine operator:

        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> residual_op = I - I.domain.element([1, 1, 1])
        >>> op = BroadcastOperator(residual_op, 2 * residual_op)

        Calling operator offsets by ``[1, 1, 1]``:

        >>> x = [1, 2, 3]
        >>> op(x)
        ProductSpace(rn(3), 2).element([
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 4.0]
        ])

        The derivative of this affine operator does not have an offset:

        >>> op.derivative(x)(x)
        ProductSpace(rn(3), 2).element([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
        ])
        """
        return BroadcastOperator(*[op.derivative(x) for op in
                                   self.operators])

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : linear `BroadcastOperator`

        Examples
        --------
        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> op = BroadcastOperator(I, 2 * I)
        >>> op.adjoint([[1, 2, 3], [2, 3, 4]])
        rn(3).element([5.0, 8.0, 11.0])
        """
        return ReductionOperator(*[op.adjoint for op in self.operators])


class ReductionOperator(Operator):
    """Reduce argument over set of operators.

    An argument is reduced by evaluating several operators and summing the
    result::

        ReductionOperator(op1, op2)(x) = op1(x[0]) + op2(x[1])

    See Also
    --------
    ProductSpaceOperator : More general case, used as backend.
    BroadcastOperator : Calls several operators with same argument.
    DiagonalOperator : Case where each operator should have its own argument.
    """
    def __init__(self, *operators):
        """Initialize a new instance.

        Parameters
        ----------
        operator1,...,operatorN : `Operator` or `int`
            The individual operators that should be evaluated and summed.
            Can also be given as ``operator, n`` with ``n`` integer,
            in which case ``operator`` is repeated ``n`` times.

        Examples
        --------
        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> op = ReductionOperator(I, 2 * I)
        >>> op.domain
        ProductSpace(rn(3), 2)
        >>> op.range
        rn(3)

        Evaluating in a point gives sum:

        >>> op([[1.0, 2.0, 3.0],
        ...     [4.0, 6.0, 8.0]])
        rn(3).element([9.0, 14.0, 19.0])

        Can also be created using a multiple of a single operator:

        >>> op = ReductionOperator(I, 2)
        >>> op.operators
        (IdentityOperator(rn(3)), IdentityOperator(rn(3)))
        """
        if (len(operators) == 2 and
                isinstance(operators[0], Operator) and
                isinstance(operators[1], Integral)):
            operators = (operators[0],) * operators[1]

        self.__operators = operators
        self.__prod_op = ProductSpaceOperator([operators])

        super().__init__(self.prod_op.domain,
                         self.prod_op.range[0],
                         self.prod_op.is_linear)

    @property
    def prod_op(self):
        """`ProductSpaceOperator` implementation."""
        return self.__prod_op

    @property
    def operators(self):
        """Tuple of sub-operators that comprise ``self``."""
        return self.__operators

    def __getitem__(self, index):
        """Return an operator by index."""
        return self.operators[index]

    def _call(self, x, out=None):
        """Apply operators to ``x`` and sum."""
        if out is None:
            return self.prod_op(x)[0]
        else:
            wrapped_out = self.prod_op.range.element([out], cast=False)
            return self.prod_op(x, out=wrapped_out)

    def derivative(self, x):
        """Derivative of the reduction operator.

        Parameters
        ----------
        x : `domain` element
            The point to take the derivative in.

        Returns
        -------
        derivative : linear `BroadcastOperator`

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = [1.0, 2.0, 3.0]
        >>> y = [4.0, 6.0, 8.0]

        Example with linear operator (derivative is itself)

        >>> op = ReductionOperator(I, 2 * I)
        >>> op([x, y])
        rn(3).element([9.0, 14.0, 19.0])
        >>> op.derivative([x, y])([x, y])
        rn(3).element([9.0, 14.0, 19.0])

        Example with affine operator

        >>> residual_op = I - r3.element([1, 1, 1])
        >>> op = ReductionOperator(residual_op, 2 * residual_op)

        Calling operator gives offset by [3, 3, 3]

        >>> op([x, y])
        rn(3).element([6.0, 11.0, 16.0])

        Derivative of affine operator does not have this offset

        >>> op.derivative([x, y])([x, y])
        rn(3).element([9.0, 14.0, 19.0])
        """
        return ReductionOperator(*[op.derivative(xi)
                                   for op, xi in zip(self.operators, x)])

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : linear `BroadcastOperator`

        Examples
        --------
        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> op = ReductionOperator(I, 2 * I)
        >>> op.adjoint([1, 2, 3])
        ProductSpace(rn(3), 2).element([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0]
        ])
        """
        return BroadcastOperator(*[op.adjoint for op in self.operators])


class DiagonalOperator(ProductSpaceOperator):
    """Diagonal 'matrix' of operators.

    For example, if ``A`` and ``B`` are operators, the diagonal operator
    can be seen as a matrix of operators::

        [[A, 0],
         [0, B]]

    When evaluated it gives::

         DiagonalOperator(op1, op2)(x) = [op1(x), op2(x)]

    See Also
    --------
    ProductSpaceOperator : Case when the 'matrix' is dense.
    BroadcastOperator : Case when a single argument is used by several ops.
    ReductionOperator : Calculates sum of operator results.
    """

    def __init__(self, *operators, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        operator1,...,operatorN : `Operator` or int
            The individual operators in the diagonal.
            Can be specified as ``operator, n`` with ``n`` integer,
            in which case the diagonal operator with ``n`` multiples of
            ``operator`` is created.
        kwargs :
            Keyword arguments passed to the `ProductSpaceOperator` backend.

        Examples
        --------
        >>> I = odl.IdentityOperator(odl.rn(3))
        >>> op = DiagonalOperator(I, 2 * I)
        >>> op.domain
        ProductSpace(rn(3), 2)
        >>> op.range
        ProductSpace(rn(3), 2)

        Evaluation is distributed so each argument is given to one operator.
        The argument order is the same as the order of the operators:

        >>> op([[1, 2, 3],
        ...     [4, 5, 6]])
        ProductSpace(rn(3), 2).element([
            [1.0, 2.0, 3.0],
            [8.0, 10.0, 12.0]
        ])

        Can also be created using a multiple of a single operator

        >>> op = DiagonalOperator(I, 2)
        >>> op.operators
        (IdentityOperator(rn(3)), IdentityOperator(rn(3)))
        """
        if (len(operators) == 2 and
                isinstance(operators[0], Operator) and
                isinstance(operators[1], Integral)):
            operators = (operators[0],) * operators[1]

        indices = [range(len(operators)), range(len(operators))]
        shape = (len(operators), len(operators))
        op_matrix = sp.sparse.coo_matrix((operators, indices), shape)

        self.__operators = tuple(operators)

        super().__init__(op_matrix, **kwargs)

    @property
    def operators(self):
        """Tuple of sub-operators that comprise ``self``."""
        return self.__operators

    def __getitem__(self, index):
        """Return an operator by index."""
        return self.operators[index]

    def derivative(self, point):
        """Derivative of this operator.

        For example, if A and B are operators

            [[A, 0],
             [0, B]]

        The derivative is given by:

            [[A', 0],
             [0, B']]

        This is only well defined if each sub-operator has a derivative

        Parameters
        ----------
        point : `element-like` in ``domain``
            The point in which the derivative should be taken.

        Returns
        -------
        derivative : `DiagonalOperator`
            The derivative operator

        See Also
        --------
        ProductSpaceOperator.derivative
        """
        point = self.domain.element(point)

        derivs = [op.derivative(p) for op, p in zip(self.operators, point)]
        return DiagonalOperator(*derivs,
                                domain=self.domain, range=self.range)

    @property
    def adjoint(self):
        """Adjoint of this operator.

        For example, if A and B are operators::

            [[A, 0],
             [0, B]]

        The adjoint is given by::

            [[A^*, 0],
             [0, B^*]]

        This is only well defined if each sub-operator has an adjoint

        Returns
        -------
        adjoint : `DiagonalOperator`
            The adjoint operator

        See Also
        --------
        ProductSpaceOperator.adjoint
        """
        adjoints = [op.adjoint for op in self.operators]
        return DiagonalOperator(*adjoints,
                                domain=self.range, range=self.domain)

    @property
    def inverse(self):
        """Inverse of this operator.

        For example, if A and B are operators::

            [[A, 0],
             [0, B]]

        The inverse is given by::

            [[A^-1, 0],
             [0, B^-1]]

        This is only well defined if each sub-operator has an inverse

        Returns
        -------
        inverse : `DiagonalOperator`
            The inverse operator

        See Also
        --------
        ProductSpaceOperator.inverse
        """
        inverses = [op.inverse for op in self.operators]
        return DiagonalOperator(*inverses,
                                domain=self.range, range=self.domain)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
