# Copyright 2014, 2015 Jonas Adler
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

# ODL imports
from odl.operator.operator import Operator
from odl.set.pspace import ProductSpace


__all__ = ('ProductSpaceOperator',
           'ComponentProjection', 'ComponentProjectionAdjoint')


class ProductSpaceOperator(Operator):

    """A "matrix of operators" on product spaces.

    This is intended for the case where an operator can be decomposed
    as a linear combination of "sub-operators", e.g.

        :math:`\\left(
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
        \end{array}\\right)`

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

    Notes
    -----
    In many cases it is of interest to have an operator from a `ProductSpace`
    to any `LinearSpace`. It that case this operator can be used with a slight
    modification, simply run

    ``prod_op = ProductSpaceOperator(prod_space, ProductSpace(linear_space))``

    The same can be done for operators `LinearSpace` -> `ProductSpace`

    ``prod_op = ProductSpaceOperator(ProductSpace(linear_space), prod_space)``
    """

    def __init__(self, operators, dom=None, ran=None):
        """Initialize a new instance.

        Parameters
        ----------
        operators : array-like
            An array of `Operator`'s
        dom : `ProductSpace`
            Domain of the operator. If not provided, it is tried to be
            inferred from the operators. This requires each **column**
            to contain at least one operator.
        ran : `ProductSpace`
            Range of the operator. If not provided, it is tried to be
            inferred from the operators. This requires each **row**
            to contain at least one operator.

        Examples
        --------
        >>> import odl
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)

        Sum of elements

        >>> prod_op = ProductSpaceOperator([I, I])

        Diagonal operator, 0 or None means ignore, or the implicit zero op.

        >>> prod_op = ProductSpaceOperator([[I, 0], [None, I]])

        Complicated combinations also possible

        >>> prod_op = ProductSpaceOperator([[I, I], [I, 0]])
        """

        # Validate input data
        if dom is not None:
            if not isinstance(dom, ProductSpace):
                raise TypeError('space {!r} not a ProductSpace instance.'
                                ''.format(dom))
            if dom.weights is not None:
                raise NotImplementedError('weighted spaces not supported.')

        if ran is not None:
            if not isinstance(ran, ProductSpace):
                raise TypeError('space {!r} not a ProductSpace instance.'
                                ''.format(ran))
            if ran.weights is not None:
                raise NotImplementedError('weighted spaces not supported.')

        # Convert ops to sparse representation
        self.ops = sp.sparse.coo_matrix(operators)

        if not all(isinstance(op, Operator) for op in self.ops.data):
            raise TypeError('operators {!r} must be a matrix of operators.'
                            ''.format(operators))

        # Set domain and range (or verify if given)
        if dom is None:
            domains = [None] * self.ops.shape[1]
        else:
            domains = dom

        if ran is None:
            ranges = [None] * self.ops.shape[0]
        else:
            ranges = ran

        for row, col, op in zip(self.ops.row, self.ops.col, self.ops.data):
            if domains[col] is None:
                domains[col] = op.domain
            elif domains[col] != op.domain:
                raise ValueError('Column {}, has inconsistent domains,'
                                 'got {} and {}'
                                 ''.format(col, domains[col], op.domain))

            if ranges[row] is None:
                ranges[row] = op.range
            elif ranges[row] != op.range:
                raise ValueError('Row {}, has inconsistent ranges,'
                                 'got {} and {}'
                                 ''.format(row, ranges[row], op.range))

        if dom is None:
            for col, sub_domain in enumerate(domains):
                if sub_domain is None:
                    raise ValueError('Col {} empty, unable to determine '
                                     'domain, please use `dom` parameter'
                                     ''.format(col))

            dom = ProductSpace(*domains)

        if ran is None:
            for row, sub_range in enumerate(ranges):
                if sub_range is None:
                    raise ValueError('Row {} empty, unable to determine '
                                     'range, please use `ran` parameter'
                                     ''.format(row))

            ran = ProductSpace(*ranges)

        # Set linearity
        linear = all(op.is_linear for op in self.ops.data)

        super().__init__(domain=dom, range=ran, linear=linear)

    def _apply(self, x, out):
        """Call the ProductSpace operators in-place.

        Parameters
        ----------
        x : domain element
            input vector to be evaluated
        out : range element
            output vector to write result to

        Returns
        -------
        `None`

        Examples
        --------
        See `_call`
        """
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

    def _call(self, x):
        """Call the ProductSpace operators.

        Parameters
        ----------
        x : domain element
            Input vector to be evaluated

        Returns
        -------
        out : range element
            Result of the evaluation

        Examples
        --------
        >>> import odl
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = X.element([[1, 2, 3], [4, 5, 6]])

        Sum of elements:

        >>> prod_op = ProductSpaceOperator([I, I])
        >>> prod_op(x)
        ProductSpace(Rn(3), 1).element([
            [5.0, 7.0, 9.0]
        ])

        Diagonal operator -- 0 or `None` means ignore, or the implicit
        zero operator:

        >>> prod_op = ProductSpaceOperator([[I, 0], [0, I]])
        >>> prod_op(x)
        ProductSpace(Rn(3), 2).element([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        Complicated combinations:

        >>> prod_op = ProductSpaceOperator([[I, I], [I, 0]])
        >>> prod_op(x)
        ProductSpace(Rn(3), 2).element([
            [5.0, 7.0, 9.0],
            [1.0, 2.0, 3.0]
        ])
        """
        out = self.range.zero()
        for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
            out[i] += op(x[j])
        return out

    @property
    def adjoint(self):
        """Adjoint of the product space operator.

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
        >>> import odl
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = X.element([[1, 2, 3], [4, 5, 6]])

        Matrix is transposed:

        >>> prod_op = ProductSpaceOperator([[0, I], [0, 0]],
        ...                                dom=X, ran=X)
        >>> prod_op(x)
        ProductSpace(Rn(3), 2).element([
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]
        ])
        >>> prod_op.adjoint(x)
        ProductSpace(Rn(3), 2).element([
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
        """op.__repr__() <==> repr(op)."""
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
        """
        Parameters
        ----------
        space : `ProductSpace`
            The space to project from
        index : `int`, `slice`, or `iterable` [int]
            The indices defining the subspace. If ``index`` is not
            and `int`, the `Operator.range` of this
            operator is also a `ProductSpace`.

        Examples
        --------
        >>> import odl
        >>> r1 = odl.Rn(1)
        >>> r2 = odl.Rn(2)
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r1, r2, r3)

        Projection on n-th component

        >>> proj = odl.ComponentProjection(X, 0)
        >>> proj.range
        Rn(1)

        Projection on sub-space

        >>> proj = odl.ComponentProjection(X, [0, 2])
        >>> proj.range
        ProductSpace(Rn(1), Rn(3))
        """
        self._index = index
        super().__init__(space, space[index], linear=True)

    @property
    def index(self):
        """ Index of the subspace. """
        return self._index

    def _apply(self, x, out):
        """Project x onto subspace in-place.

        See also
        --------
        ComponentProjection._call
        """
        out.assign(x[self.index])

    def _call(self, x):
        """Project x onto subspace out-of-place.

        Parameters
        ----------
        x : domain element
            input vector to be projected

        Returns
        -------
        out : range element
            Projection of x onto subspace

        Examples
        --------
        >>> import odl
        >>> r1 = odl.Rn(1)
        >>> r2 = odl.Rn(2)
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r1, r2, r3)
        >>> x = X.element([[1], [2, 3], [4, 5, 6]])

        Projection on n-th component

        >>> proj = odl.ComponentProjection(X, 0)
        >>> proj(x)
        Rn(1).element([1.0])

        Projection on sub-space

        >>> proj = odl.ComponentProjection(X, [0, 2])
        >>> proj(x)
        ProductSpace(Rn(1), Rn(3)).element([
            [1.0],
            [4.0, 5.0, 6.0]
        ])
        """
        return x[self.index].copy()

    @property
    def adjoint(self):
        """Return the adjoint operator.

        The adjoint is given by extending along `ComponentProjection.index`,
        and setting zero along the others.

        See also
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
            The space to project to
        index : `int`, `slice`, or `iterable` [int]
            The indexes to project from

        Examples
        --------
        >>> import odl
        >>> r1 = odl.Rn(1)
        >>> r2 = odl.Rn(2)
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r1, r2, r3)

        Projection on n-th component

        >>> proj = odl.ComponentProjectionAdjoint(X, 0)
        >>> proj.domain
        Rn(1)

        Projection on sub-space

        >>> proj = odl.ComponentProjectionAdjoint(X, [0, 2])
        >>> proj.domain
        ProductSpace(Rn(1), Rn(3))
        """
        self._index = index
        super().__init__(space[index], space, linear=True)

    @property
    def index(self):
        """ Index of the subspace. """
        return self._index

    def _apply(self, x, out):
        """Evaluate this operator in-place.

        Extend ``x`` from the subspace related to `index`.

        See also
        --------
        ComponentProjectionAdjoint._call
        """
        out.set_zero()
        out[self.index] = x

    def _call(self, x):
        """Evaluate this operator out-of-place.

        Extend ``x`` from the subspace related to `index`.

        Parameters
        ----------
        x : domain element
            Input vector to be projected

        Returns
        -------
        out : range element
            Projection of x to superspace

        Examples
        --------
        >>> import odl
        >>> r1 = odl.Rn(1)
        >>> r2 = odl.Rn(2)
        >>> r3 = odl.Rn(3)
        >>> X = odl.ProductSpace(r1, r2, r3)
        >>> x = X.element([[1], [2, 3], [4, 5, 6]])

        Projection on n-th component

        >>> proj = odl.ComponentProjectionAdjoint(X, 0)
        >>> proj(x[0])
        ProductSpace(Rn(1), Rn(2), Rn(3)).element([
            [1.0],
            [0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        Projection on sub-space

        >>> proj = odl.ComponentProjectionAdjoint(X, [0, 2])
        >>> proj(x[0, 2])
        ProductSpace(Rn(1), Rn(2), Rn(3)).element([
            [1.0],
            [0.0, 0.0],
            [4.0, 5.0, 6.0]
        ])
        """
        out = self.range.zero()
        out[self.index] = x
        return out

    @property
    def adjoint(self):
        """The adjoint operator.

        The adjoint is given by the `ComponentProjection`
        related to this operator's `index`.

        See also
        --------
        ComponentProjection
        """
        return ComponentProjection(self.range, self.index)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
