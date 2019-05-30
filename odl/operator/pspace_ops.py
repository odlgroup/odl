# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Default operators defined on any `ProductSpace`."""

from __future__ import print_function, division, absolute_import
from numbers import Integral
import numpy as np

from odl.operator.operator import Operator
from odl.operator.default_ops import ZeroOperator
from odl.space import ProductSpace


__all__ = ('ProductSpaceOperator',
           'ComponentProjection', 'ComponentProjectionAdjoint',
           'BroadcastOperator', 'ReductionOperator', 'DiagonalOperator')


class ProductSpaceOperator(Operator):

    r"""A "matrix of operators" on product spaces.

    For example a matrix of operators can act on a vector by

        ``ProductSpaceOperator([[A, B], [C, D]])([x, y]) =
        [A(x) + B(y), C(x) + D(y)]``

    Notes
    -----
    This is intended for the case where an operator can be decomposed
    as a linear combination of "sub-operators", e.g.

    .. math::
        \left(
        \begin{array}{ccc}
        A & B & 0 \\
        0 & C & 0 \\
        0 & 0 & D
        \end{array}\right)
        \left(
        \begin{array}{c}
        x \\
        y \\
        z
        \end{array}\right)
        =
        \left(
        \begin{array}{c}
        A(x) + B(y) \\
        C(y) \\
        D(z)
        \end{array}\right)

    Mathematically, a `ProductSpaceOperator` is an operator

    .. math::
        \mathcal{A}: \mathcal{X} \to \mathcal{Y}

    between product spaces
    :math:`\mathcal{X}=\mathcal{X}_1 \times\dots\times \mathcal{X}_m`
    and
    :math:`\mathcal{Y}=\mathcal{Y}_1 \times\dots\times \mathcal{Y}_n`
    which can be written in the form

    .. math::
        \mathcal{A} = (\mathcal{A}_{ij})_{i,j},  \quad
                          i = 1, \dots, n, \ j = 1, \dots, m

    with *component operators*
    :math:`\mathcal{A}_{ij}: \mathcal{X}_j \to \mathcal{Y}_i`.

    Its action on a vector :math:`x = (x_1, \dots, x_m)` is defined as
    the matrix multiplication

    .. math::
        [\mathcal{A}(x)]_i = \sum_{j=1}^m \mathcal{A}_{ij}(x_j).

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
            An array of `Operator`'s, must be 2-dimensional.
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
        >>> pspace = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = pspace.element([[1, 2, 3],
        ...                     [4, 5, 6]])

        Create an operator that sums two inputs:

        >>> prod_op = odl.ProductSpaceOperator([[I, I]])
        >>> prod_op(x)
        ProductSpace(rn(3), 1).element([
            [ 5.,  7.,  9.]
        ])

        Diagonal operator -- 0 or ``None`` means ignore, or the implicit
        zero operator:

        >>> prod_op = odl.ProductSpaceOperator([[I, 0],
        ...                                     [0, I]])
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [ 1.,  2.,  3.],
            [ 4.,  5.,  6.]
        ])

        If a column is empty, the operator domain must be specified. The
        same holds for an empty row and the range of the operator:

        >>> prod_op = odl.ProductSpaceOperator([[I, 0],
        ...                                     [I, 0]], domain=r3 ** 2)
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [ 1.,  2.,  3.],
            [ 1.,  2.,  3.]
        ])
        >>> prod_op = odl.ProductSpaceOperator([[I, I],
        ...                                     [0, 0]], range=r3 ** 2)
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [ 5.,  7.,  9.],
            [ 0.,  0.,  0.]
        ])
        """
        # Lazy import to improve `import odl` time
        import scipy.sparse

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

        if isinstance(operators, scipy.sparse.spmatrix):
            if not all(isinstance(op, Operator) for op in operators.data):
                raise ValueError('sparse matrix `operator` contains non-'
                                 '`Operator` entries')
            self.__ops = operators
        else:
            self.__ops = self._convert_to_spmatrix(operators)

        # Set domain and range (or verify if given)
        if domain is None:
            domains = [None] * self.__ops.shape[1]
        else:
            domains = domain

        if range is None:
            ranges = [None] * self.__ops.shape[0]
        else:
            ranges = range

        for row, col, op in zip(self.__ops.row, self.__ops.col,
                                self.__ops.data):
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
        linear = all(op.is_linear for op in self.__ops.data)

        super(ProductSpaceOperator, self).__init__(
            domain=domain, range=range, linear=linear)

    @staticmethod
    def _convert_to_spmatrix(operators):
        """Convert an array-like object of operators to a sparse matrix."""
        # Lazy import to improve `import odl` time
        import scipy.sparse

        # Convert ops to sparse representation. This is not trivial because
        # operators can be indexable themselves and give the wrong impression
        # of an extra dimension. So we have to infer the shape manually
        # first and extract the indices of nonzero positions.
        nrows = len(operators)
        ncols = None
        irow, icol, data = [], [], []
        for i, row in enumerate(operators):
            try:
                iter(row)
            except TypeError:
                raise ValueError(
                    '`operators` must be a matrix of `Operator` objects, `0` '
                    'or `None`, got {!r} (row {} = {!r} is not iterable)'
                    ''.format(operators, i, row))

            if isinstance(row, Operator):
                raise ValueError(
                    '`operators` must be a matrix of `Operator` objects, `0` '
                    'or `None`, but row {} is an `Operator` {!r}'
                    ''.format(i, row))

            if ncols is None:
                ncols = len(row)
            elif len(row) != ncols:
                raise ValueError(
                    'all rows in `operators` must have the same length, but '
                    'length {} of row {} differs from previous common length '
                    '{}'.format(len(row), i, ncols))

            for j, col in enumerate(row):
                if col is None or col is 0:
                    pass
                elif isinstance(col, Operator):
                    irow.append(i)
                    icol.append(j)
                    data.append(col)
                else:
                    raise ValueError(
                        '`operators` must be a matrix of `Operator` objects, '
                        '`0` or `None`, got entry {!r} at ({}, {})'
                        ''.format(col, i, j))

        # Create object array explicitly, threby avoiding erroneous conversion
        # in `coo_matrix.__init__`
        data_arr = np.empty(len(data), dtype=object)
        data_arr[:] = data
        return scipy.sparse.coo_matrix((data_arr, (irow, icol)),
                                       shape=(nrows, ncols))

    @property
    def ops(self):
        """The sparse operator matrix representing this operator."""
        return self.__ops

    def _call(self, x, out=None):
        """Call the operators on the parts of ``x``."""
        # TODO: add optimization in case an operator appears repeatedly in a
        # row
        if out is None:
            out = self.range.zero()
            for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
                out[i] += op(x[j])
        else:
            has_evaluated_row = np.zeros(len(self.range), dtype=bool)
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
        >>> pspace = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = pspace.element([[1, 2, 3], [4, 5, 6]])

        Example with linear operator (derivative is itself)

        >>> prod_op = ProductSpaceOperator([[0, I], [0, 0]],
        ...                                domain=pspace, range=pspace)
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [ 4.,  5.,  6.],
            [ 0.,  0.,  0.]
        ])
        >>> prod_op.derivative(x)(x)
        ProductSpace(rn(3), 2).element([
            [ 4.,  5.,  6.],
            [ 0.,  0.,  0.]
        ])

        Example with affine operator

        >>> residual_op = I - r3.element([1, 1, 1])
        >>> op = ProductSpaceOperator([[0, residual_op], [0, 0]],
        ...                           domain=pspace, range=pspace)

        Calling operator gives offset by [1, 1, 1]

        >>> op(x)
        ProductSpace(rn(3), 2).element([
            [ 3.,  4.,  5.],
            [ 0.,  0.,  0.]
        ])

        Derivative of affine operator does not have this offset

        >>> op.derivative(x)(x)
        ProductSpace(rn(3), 2).element([
            [ 4.,  5.,  6.],
            [ 0.,  0.,  0.]
        ])
        """
        # Lazy import to improve `import odl` time
        import scipy.sparse

        # Short circuit optimization
        if self.is_linear:
            return self

        deriv_ops = [op.derivative(x[col]) for op, col in zip(self.ops.data,
                                                              self.ops.col)]
        data = np.empty(len(deriv_ops), dtype=object)
        data[:] = deriv_ops
        indices = [self.ops.row, self.ops.col]
        shape = self.ops.shape
        deriv_matrix = scipy.sparse.coo_matrix((data, indices), shape)
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
        >>> pspace = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> x = pspace.element([[1, 2, 3],
        ...                     [4, 5, 6]])

        Matrix is transposed:

        >>> prod_op = ProductSpaceOperator([[0, I], [0, 0]],
        ...                                domain=pspace, range=pspace)
        >>> prod_op(x)
        ProductSpace(rn(3), 2).element([
            [ 4.,  5.,  6.],
            [ 0.,  0.,  0.]
        ])
        >>> prod_op.adjoint(x)
        ProductSpace(rn(3), 2).element([
            [ 0.,  0.,  0.],
            [ 1.,  2.,  3.]
        ])
        """
        # Lazy import to improve `import odl` time
        import scipy.sparse

        adjoint_ops = [op.adjoint for op in self.ops.data]
        data = np.empty(len(adjoint_ops), dtype=object)
        data[:] = adjoint_ops
        indices = [self.ops.col, self.ops.row]  # Swap col/row -> transpose
        shape = (self.ops.shape[1], self.ops.shape[0])
        adj_matrix = scipy.sparse.coo_matrix((data, indices), shape)
        return ProductSpaceOperator(adj_matrix, self.range, self.domain)

    def __getitem__(self, index):
        """Get sub-operator by index.

        Parameters
        ----------
        index : int or tuple of int
            A pair of integers given as (row, col).

        Returns
        -------
        suboperator : `ReductionOperator`, `Operator` or ``0``
            If index is an integer, return the row given by the index.

            If index is a tuple, it must have two elements.
            if there is an operator at ``(row, col)``,  the operator is
            returned, otherwise ``0``.

        Examples
        --------
        >>> r3 = odl.rn(3)
        >>> pspace = odl.ProductSpace(r3, r3)
        >>> I = odl.IdentityOperator(r3)
        >>> prod_op = ProductSpaceOperator([[0, I],
        ...                                 [0, 0]],
        ...                                domain=pspace, range=pspace)
        >>> prod_op[0, 0]
        0
        >>> prod_op[0, 1]
        IdentityOperator(rn(3))
        >>> prod_op[1, 0]
        0
        >>> prod_op[1, 1]
        0

        By accessing single indices, a row is extracted as a
        `ReductionOperator`:

        >>> prod_op[0]
        ReductionOperator(ZeroOperator(rn(3)), IdentityOperator(rn(3)))
        """
        if isinstance(index, tuple):
            row, col = index

            linear_index = np.flatnonzero((self.ops.row == row) &
                                          (self.ops.col == col))
            if linear_index.size == 0:
                return 0
            else:
                return self.ops.data[int(linear_index)]
        else:
            index = int(index)

            ops = [None] * len(self.domain)
            for op, col, row in zip(self.ops.data, self.ops.col, self.ops.row):
                if row == index:
                    ops[col] = op

            for i in range(len(self.domain)):
                if ops[i] is None:
                    ops[i] = ZeroOperator(self.domain[i])

            return ReductionOperator(*ops)

    @property
    def shape(self):
        """Shape of the matrix of operators."""
        return self.ops.shape

    def __len__(self):
        """Return ``len(self)``."""
        return self.shape[0]

    @property
    def size(self):
        """Total size of the matrix of operators."""
        return np.prod(self.shape, dtype='int64')

    def __repr__(self):
        """Return ``repr(self)``."""
        aslist = [[0] * len(self.domain) for _ in range(len(self.range))]
        for i, j, op in zip(self.ops.row, self.ops.col, self.ops.data):
            aslist[i][j] = op
        return '{}({!r})'.format(self.__class__.__name__, aslist)


class ComponentProjection(Operator):

    r"""Projection onto the subspace identified by an index.

    For a product space :math:`\mathcal{X} = \mathcal{X}_1 \times \dots
    \times \mathcal{X}_n`, the component projection

    .. math::
       \mathcal{P}_i: \mathcal{X} \to \mathcal{X}_i

    is given by :math:`\mathcal{P}_i(x) = x_i` for an element
    :math:`x = (x_1, \dots, x_n) \in \mathcal{X}`.

    More generally, for an index set :math:`I \subset \{1, \dots, n\}`,
    the projection operator :math:`\mathcal{P}_I` is defined by
    :math:`\mathcal{P}_I(x) = (x_i)_{i \in I}`.

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
        index : int, slice, or list
            Indices defining the subspace. If ``index`` is not an integer,
            the `Operator.range` of this operator is also a `ProductSpace`.

        Examples
        --------
        >>> r1 = odl.rn(1)
        >>> r2 = odl.rn(2)
        >>> r3 = odl.rn(3)
        >>> pspace = odl.ProductSpace(r1, r2, r3)

        Projection on n-th component:

        >>> proj = odl.ComponentProjection(pspace, 0)
        >>> x = [[1],
        ...      [2, 3],
        ...      [4, 5, 6]]
        >>> proj(x)
        rn(1).element([ 1.])

        Projection on sub-space:

        >>> proj = odl.ComponentProjection(pspace, [0, 2])
        >>> proj(x)
        ProductSpace(rn(1), rn(3)).element([
            [ 1.],
            [ 4.,  5.,  6.]
        ])
        """
        self.__index = index
        super(ComponentProjection, self).__init__(
            space, space[index], linear=True)

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
        """The adjoint operator.

        The adjoint is given by extending along `ComponentProjection.index`,
        and setting zero along the others.

        See Also
        --------
        ComponentProjectionAdjoint
        """
        return ComponentProjectionAdjoint(self.domain, self.index)

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> pspace = odl.ProductSpace(odl.rn(1), odl.rn(2))
        >>> odl.ComponentProjection(pspace, 0)
        ComponentProjection(ProductSpace(rn(1), rn(2)), 0)
        """
        return '{}({!r}, {})'.format(self.__class__.__name__,
                                     self.domain, self.index)


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
        index : int, slice, or list
            Indexes to project from.

        Examples
        --------
        >>> r1 = odl.rn(1)
        >>> r2 = odl.rn(2)
        >>> r3 = odl.rn(3)
        >>> pspace = odl.ProductSpace(r1, r2, r3)
        >>> x = pspace.element([[1],
        ...                     [2, 3],
        ...                     [4, 5, 6]])

        Projection on the 0-th component:

        >>> proj_adj = odl.ComponentProjectionAdjoint(pspace, 0)
        >>> proj_adj(x[0])
        ProductSpace(rn(1), rn(2), rn(3)).element([
            [ 1.],
            [ 0.,  0.],
            [ 0.,  0.,  0.]
        ])

        Projection on a sub-space corresponding to indices 0 and 2:

        >>> proj_adj = odl.ComponentProjectionAdjoint(pspace, [0, 2])
        >>> proj_adj(x[[0, 2]])
        ProductSpace(rn(1), rn(2), rn(3)).element([
            [ 1.],
            [ 0.,  0.],
            [ 4.,  5.,  6.]
        ])
        """
        self.__index = index
        super(ComponentProjectionAdjoint, self).__init__(
            space[index], space, linear=True)

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

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> pspace = odl.ProductSpace(odl.rn(1), odl.rn(2))
        >>> odl.ComponentProjectionAdjoint(pspace, 0)
        ComponentProjectionAdjoint(ProductSpace(rn(1), rn(2)), 0)
        """
        return '{}({!r}, {})'.format(self.__class__.__name__,
                                     self.range, self.index)


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
            [ 1.,  2.,  3.],
            [ 2.,  4.,  6.]
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
        super(BroadcastOperator, self).__init__(
            self.prod_op.domain[0], self.prod_op.range,
            linear=self.prod_op.is_linear)

    @property
    def prod_op(self):
        """`ProductSpaceOperator` implementation."""
        return self.__prod_op

    @property
    def operators(self):
        """Tuple of sub-operators that comprise ``self``."""
        return self.__operators

    def __getitem__(self, index):
        """Return ``self(index)``."""
        return self.operators[index]

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.operators)

    @property
    def size(self):
        """Total number of sub-operators."""
        return len(self)

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
            [ 0.,  1.,  2.],
            [ 0.,  2.,  4.]
        ])

        The derivative of this affine operator does not have an offset:

        >>> op.derivative(x)(x)
        ProductSpace(rn(3), 2).element([
            [ 1.,  2.,  3.],
            [ 2.,  4.,  6.]
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
        rn(3).element([  5.,   8.,  11.])
        """
        return ReductionOperator(*[op.adjoint for op in self.operators])

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> spc = odl.rn(3)
        >>> id = odl.IdentityOperator(spc)
        >>> odl.BroadcastOperator(id, 3)
        BroadcastOperator(IdentityOperator(rn(3)), 3)
        >>> scale = odl.ScalingOperator(spc, 3)
        >>> odl.BroadcastOperator(id, scale)
        BroadcastOperator(IdentityOperator(rn(3)), ScalingOperator(rn(3), 3.0))
        """
        if all(op == self[0] for op in self):
            return '{}({!r}, {})'.format(self.__class__.__name__,
                                         self[0], len(self))
        else:
            op_repr = ', '.join(repr(op) for op in self)
            return '{}({})'.format(self.__class__.__name__, op_repr)


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
    SeparableSum : Corresponding construction for functionals.
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

        Evaluating in a point gives the sum of the evaluation results of
        the individual operators:

        >>> op([[1, 2, 3],
        ...     [4, 6, 8]])
        rn(3).element([  9.,  14.,  19.])

        An ``out`` argument can be given for in-place evaluation:

        >>> out = op.range.element()
        >>> result = op([[1, 2, 3],
        ...              [4, 6, 8]], out=out)
        >>> out
        rn(3).element([  9.,  14.,  19.])
        >>> result is out
        True

        There is a simplified syntax for the case that all operators are
        the same:

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

        super(ReductionOperator, self).__init__(
            self.prod_op.domain, self.prod_op.range[0],
            linear=self.prod_op.is_linear)

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

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.operators)

    @property
    def size(self):
        """Total number of sub-operators."""
        return len(self)

    def _call(self, x, out=None):
        """Apply operators to ``x`` and sum."""
        if out is None:
            return self.prod_op(x)[0]
        else:
            wrapped_out = self.prod_op.range.element([out], cast=False)
            pspace_result = self.prod_op(x, out=wrapped_out)
            return pspace_result[0]

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
        rn(3).element([  9.,  14.,  19.])
        >>> op.derivative([x, y])([x, y])
        rn(3).element([  9.,  14.,  19.])

        Example with affine operator

        >>> residual_op = I - r3.element([1, 1, 1])
        >>> op = ReductionOperator(residual_op, 2 * residual_op)

        Calling operator gives offset by [3, 3, 3]

        >>> op([x, y])
        rn(3).element([  6.,  11.,  16.])

        Derivative of affine operator does not have this offset

        >>> op.derivative([x, y])([x, y])
        rn(3).element([  9.,  14.,  19.])
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
            [ 1.,  2.,  3.],
            [ 2.,  4.,  6.]
        ])
        """
        return BroadcastOperator(*[op.adjoint for op in self.operators])

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> spc = odl.rn(3)
        >>> id = odl.IdentityOperator(spc)
        >>> odl.ReductionOperator(id, 3)
        ReductionOperator(IdentityOperator(rn(3)), 3)
        >>> scale = odl.ScalingOperator(spc, 3)
        >>> odl.ReductionOperator(id, scale)
        ReductionOperator(IdentityOperator(rn(3)), ScalingOperator(rn(3), 3.0))
        """
        if all(op == self[0] for op in self):
            return '{}({!r}, {})'.format(self.__class__.__name__,
                                         self[0], len(self))
        else:
            op_repr = ', '.join(repr(op) for op in self)
            return '{}({})'.format(self.__class__.__name__, op_repr)


class DiagonalOperator(ProductSpaceOperator):
    """Diagonal 'matrix' of operators.

    For example, if ``A`` and ``B`` are operators, the diagonal operator
    can be seen as a matrix of operators::

        [[A, 0],
         [0, B]]

    When evaluated it gives::

         DiagonalOperator(op1, op2)(x) = [op1(x[0]), op2(x[1])]

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
            [ 1.,  2.,  3.],
            [  8.,  10.,  12.]
        ])

        Can also be created using a multiple of a single operator

        >>> op = DiagonalOperator(I, 2)
        >>> op.operators
        (IdentityOperator(rn(3)), IdentityOperator(rn(3)))
        """
        # Lazy import to improve `import odl` time
        import scipy.sparse

        if (len(operators) == 2 and
                isinstance(operators[0], Operator) and
                isinstance(operators[1], Integral)):
            operators = (operators[0],) * operators[1]

        n_ops = len(operators)
        irow = icol = np.arange(n_ops)
        data = np.empty(n_ops, dtype=object)
        data[:] = operators
        shape = (n_ops, n_ops)
        op_matrix = scipy.sparse.coo_matrix((data, (irow, icol)), shape)
        super(DiagonalOperator, self).__init__(op_matrix, **kwargs)

        self.__operators = tuple(operators)

    @property
    def operators(self):
        """Tuple of sub-operators that comprise ``self``."""
        return self.__operators

    def __getitem__(self, index):
        """Return an operator by index."""
        return self.operators[index]

    def __len__(self):
        """Return ``len(self)``."""
        return len(self.operators)

    @property
    def size(self):
        """Total number of sub-operators."""
        return len(self)

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

    def __repr__(self):
        """Return ``repr(self)``.

        Examples
        --------
        >>> spc = odl.rn(3)
        >>> id = odl.IdentityOperator(spc)
        >>> odl.DiagonalOperator(id, 3)
        DiagonalOperator(IdentityOperator(rn(3)), 3)
        >>> scale = odl.ScalingOperator(spc, 3)
        >>> odl.DiagonalOperator(id, scale)
        DiagonalOperator(IdentityOperator(rn(3)), ScalingOperator(rn(3), 3.0))
        """
        if all(op == self[0] for op in self):
            return '{}({!r}, {})'.format(self.__class__.__name__,
                                         self[0], len(self))
        else:
            op_repr = ', '.join(repr(op) for op in self)
            return '{}({})'.format(self.__class__.__name__, op_repr)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
