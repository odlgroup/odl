# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Convenience functions for operators."""

from __future__ import print_function, division, absolute_import
from future.utils import native
import numpy as np

from odl.space.base_tensors import TensorSpace
from odl.space.pspace import ProductSpace, ProductSpaceElement
from odl.space.weighting import ArrayWeighting, ConstWeighting
from odl.util import OptionalArgDecorator


__all__ = ('matrix_representation', 'power_method_opnorm', 'as_scipy_operator',
           'as_scipy_functional', 'as_proximal_lang_operator')


def matrix_representation(op):
    """Return a matrix representation of a linear operator.

    Parameters
    ----------
    op : `Operator`
        The linear operator of which one wants a matrix representation.

    Returns
    -------
    matrix : `numpy.ndarray`
        The matrix representation of the operator.

    Examples
    --------
    Approximate a matrix on its own:

    >>> mat = np.array([[1, 2, 3],
    ...                 [4, 5, 6],
    ...                 [7, 8, 9]])
    >>> op = odl.MatrixOperator(mat)
    >>> matrix_representation(op)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    Works with product spaces:

    >>> prod_ft = odl.DiagonalOperator(op)
    >>> matrix_representation(op)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    Notes
    ----------
    The algorithm works by letting the operator act on all unit vectors, and
    stacking the output as a matrix.
    """

    if not op.is_linear:
        raise ValueError('the operator is not linear')

    if not (isinstance(op.domain, TensorSpace) or
            (isinstance(op.domain, ProductSpace) and
             all(isinstance(spc, TensorSpace) for spc in op.domain))):
        raise TypeError('operator domain {!r} is neither `TensorSpace` '
                        'nor `ProductSpace` with only `TensorSpace` '
                        'components'.format(op.domain))

    if not (isinstance(op.range, TensorSpace) or
            (isinstance(op.range, ProductSpace) and
             all(isinstance(spc, TensorSpace) for spc in op.range))):
        raise TypeError('operator range {!r} is neither `TensorSpace` '
                        'nor `ProductSpace` with only `TensorSpace` '
                        'components'.format(op.range))

    # Get the size of the range, and handle ProductSpace
    # Store for reuse in loop
    op_ran_is_prod_space = isinstance(op.range, ProductSpace)
    if op_ran_is_prod_space:
        num_ran = len(op.range)
        n = [ran.size for ran in op.range]
    else:
        num_ran = 1
        n = [op.range.size]

    # Get the size of the domain, and handle ProductSpace
    # Store for reuse in loop
    op_dom_is_prod_space = isinstance(op.domain, ProductSpace)
    if op_dom_is_prod_space:
        num_dom = len(op.domain)
        m = [dom.size for dom in op.domain]
    else:
        num_dom = 1
        m = [op.domain.size]

    # Generate the matrix
    dtype = np.promote_types(op.domain.dtype, op.range.dtype)
    matrix = np.zeros([np.sum(n), np.sum(m)], dtype=dtype)
    tmp_ran = op.range.element()  # Store for reuse in loop
    tmp_dom = op.domain.zero()  # Store for reuse in loop
    index = 0
    last_i = last_j = 0

    for i in range(num_dom):
        for j in range(m[i]):
            if op_dom_is_prod_space:
                tmp_dom[last_i][last_j] = 0.0
                tmp_dom[i][j] = 1.0
            else:
                tmp_dom[last_j] = 0.0
                tmp_dom[j] = 1.0
            op(tmp_dom, out=tmp_ran)
            if op_ran_is_prod_space:
                tmp_idx = 0
                for k in range(num_ran):
                    matrix[tmp_idx: tmp_idx + op.range[k].size, index] = (
                        (tmp_ran[k]).asarray().ravel())
                    tmp_idx += op.range[k].size
            else:
                matrix[:, index] = tmp_ran.asarray().ravel()
            index += 1
            last_j = j
            last_i = i

    return matrix


def power_method_opnorm(op, xstart=None, maxiter=100, rtol=1e-05, atol=1e-08,
                        callback=None):
    """Estimate the operator norm with the power method.

    Parameters
    ----------
    op : `Operator`
        Operator whose norm is to be estimated. If its `Operator.range`
        range does not coincide with its `Operator.domain`, an
        `Operator.adjoint` must be defined (which implies that the
        operator must be linear).
    xstart : ``op.domain`` `element-like`, optional
        Starting point of the iteration. By default, the ``one``
        element of the `Operator.domain` is used.
    maxiter : positive int, optional
        Number of iterations to perform. If the domain and range of ``op``
        do not match, it needs to be an even number. If ``None`` is given,
        iterate until convergence.
    rtol : float, optional
        Relative tolerance parameter (see Notes).
    atol : float, optional
        Absolute tolerance parameter (see Notes).
    callback : callable, optional
        Function called with the current iterate in each iteration.

    Returns
    -------
    est_opnorm : float
        The estimated operator norm of ``op``.

    Examples
    --------
    Verify that the identity operator has norm 1:

    >>> space = odl.uniform_discr(0, 1, 5)
    >>> id = odl.IdentityOperator(space)
    >>> power_method_opnorm(id)
    1.0

    The operator norm scales as expected:

    >>> power_method_opnorm(3 * id)
    3.0

    Notes
    -----
    The operator norm :math:`||A||` is defined by as the smallest number
    such that

    .. math::
        ||A(x)|| \leq ||A|| ||x||

    for all :math:`x` in the domain of :math:`A`.

    The operator is evaluated until ``maxiter`` operator calls or until the
    relative error is small enough. The error measure is given by

        ``abs(a - b) <= (atol + rtol * abs(b))``,

    where ``a`` and ``b`` are consecutive iterates.
    """
    if maxiter is None:
        maxiter = np.iinfo(int).max

    maxiter, maxiter_in = int(maxiter), maxiter
    if maxiter <= 0:
        raise ValueError('`maxiter` must be positive, got {}'
                         ''.format(maxiter_in))

    if op.domain == op.range:
        use_normal = False
        ncalls = maxiter
    else:
        # Do the power iteration for A*A; the norm of A*A(x_N) is then
        # an estimate of the square of the operator norm
        # We do only half the number of iterations compared to the usual
        # case to have the same number of operator evaluations.
        use_normal = True
        ncalls = maxiter // 2
        if ncalls * 2 != maxiter:
            raise ValueError('``maxiter`` must be an even number for '
                             'non-self-adjoint operator, got {}'
                             ''.format(maxiter_in))

    # Make sure starting point is ok or select initial guess
    if xstart is None:
        try:
            x = op.domain.one()  # TODO: random? better choice?
        except AttributeError:
            raise ValueError('`xstart` must be defined in case the '
                             'operator domain has no `one()`')
    else:
        # copy to ensure xstart is not modified
        x = op.domain.element(xstart).copy()

    # Take first iteration step to normalize input
    x_norm = x.norm()
    if x_norm == 0:
        raise ValueError('``xstart`` must be nonzero')
    x /= x_norm

    # utility to calculate opnorm from xnorm
    def calc_opnorm(x_norm):
        if use_normal:
            return np.sqrt(x_norm)
        else:
            return x_norm

    # initial guess of opnorm
    opnorm = calc_opnorm(x_norm)

    # temporary to improve performance
    tmp = op.range.element()

    # Use the power method to estimate opnorm
    for i in range(ncalls):
        if use_normal:
            op(x, out=tmp)
            op.adjoint(tmp, out=x)
        else:
            op(x, out=tmp)
            x, tmp = tmp, x

        # Calculate x norm and verify it is valid
        x_norm = x.norm()
        if x_norm == 0:
            raise ValueError('reached ``x=0`` after {} iterations'.format(i))
        if not np.isfinite(x_norm):
            raise ValueError('reached nonfinite ``x={}`` after {} iterations'
                             ''.format(x, i))

        # Calculate opnorm
        opnorm, opnorm_old = calc_opnorm(x_norm), opnorm

        # Check if the breaking condition holds, stop. Else rescale and go on.
        if np.isclose(opnorm, opnorm_old, rtol, atol):
            break
        else:
            x /= x_norm

        if callback is not None:
            callback(x)

    return opnorm


def as_scipy_operator(op):
    """Wrap ``op`` as a ``scipy.sparse.linalg.LinearOperator``.

    This is intended to be used with the scipy sparse linear solvers.

    Parameters
    ----------
    op : `Operator`
        A linear operator that should be wrapped

    Returns
    -------
    ``scipy.sparse.linalg.LinearOperator`` : linear_op
        The wrapped operator, has attributes ``matvec`` which calls ``op``,
        and ``rmatvec`` which calls ``op.adjoint``.

    Examples
    --------
    Wrap operator and solve simple problem (here toy problem ``Ix = b``)

    >>> op = odl.IdentityOperator(odl.rn(3))
    >>> scipy_op = as_scipy_operator(op)
    >>> import scipy.sparse.linalg as sl
    >>> result, status = sl.cg(scipy_op, [0, 1, 0])
    >>> result
    array([ 0.,  1.,  0.])

    Notes
    -----
    If the data representation of ``op``'s domain and range is of type
    `NumpyTensorSpace` this incurs no significant overhead. If the space
    type is ``CudaFn`` or some other nonlocal type, the overhead is
    significant.
    """
    # Lazy import to improve `import odl` time
    import scipy.sparse

    if not op.is_linear:
        raise ValueError('`op` needs to be linear')

    dtype = op.domain.dtype
    if op.range.dtype != dtype:
        raise ValueError('dtypes of ``op.domain`` and ``op.range`` needs to '
                         'match')

    shape = (native(op.range.size), native(op.domain.size))

    def matvec(v):
        return (op(v.reshape(op.domain.shape))).asarray().ravel()

    def rmatvec(v):
        return (op.adjoint(v.reshape(op.range.shape))).asarray().ravel()

    return scipy.sparse.linalg.LinearOperator(shape=shape,
                                              matvec=matvec,
                                              rmatvec=rmatvec,
                                              dtype=dtype)


def as_scipy_functional(func, return_gradient=False):
    """Wrap ``op`` as a function operating on linear arrays.

    This is intended to be used with the `scipy solvers
    <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_.

    Parameters
    ----------
    func : `Functional`.
        A functional that should be wrapped
    return_gradient : bool, optional
        ``True`` if the gradient of the functional should also be returned,
        ``False`` otherwise.

    Returns
    -------
    function : ``callable``
        The wrapped functional.
    gradient : ``callable``, optional
        The wrapped gradient. Only returned if ``return_gradient`` is true.

    Examples
    --------
    Wrap functional and solve simple problem
    (here toy problem ``min_x ||x||^2``):

    >>> func = odl.solvers.L2NormSquared(odl.rn(3))
    >>> scipy_func = odl.as_scipy_functional(func)
    >>> from scipy.optimize import minimize
    >>> result = minimize(scipy_func, x0=[0, 1, 0])
    >>> np.allclose(result.x, [0, 0, 0])
    True

    The gradient (jacobian) can also be provided:

    >>> func = odl.solvers.L2NormSquared(odl.rn(3))
    >>> scipy_func, scipy_grad = odl.as_scipy_functional(func, True)
    >>> from scipy.optimize import minimize
    >>> result = minimize(scipy_func, x0=[0, 1, 0], jac=scipy_grad)
    >>> np.allclose(result.x, [0, 0, 0])
    True

    Notes
    -----
    If the data representation of ``op``'s domain is of type `NumpyFn` this
    incurs no significant overhead. If the space type is ``CudaFn`` or some
    other nonlocal type, the overhead is significant.
    """
    def func_call(arr):
        return func(np.asarray(arr).reshape(func.domain.shape))

    if return_gradient:
        def func_gradient_call(arr):
            return np.asarray(
                func.gradient(np.asarray(arr).reshape(func.domain.shape)))

        return func_call, func_gradient_call
    else:
        return func_call


def as_proximal_lang_operator(op, norm_bound=None):
    """Wrap ``op`` as a ``proximal.BlackBox``.

    This is intended to be used with the `ProxImaL language solvers.
    <https://github.com/comp-imaging/proximal>`_

    For documentation on the proximal language (ProxImaL) see [Hei+2016].

    Parameters
    ----------
    op : `Operator`
        Linear operator to be wrapped. Its domain and range must implement
        ``shape``, and elements in these need to implement ``asarray``.
    norm_bound : float, optional
        An upper bound on the spectral norm of the operator. Note that this is
        the norm as defined by ProxImaL, and hence use the unweighted spaces.

    Returns
    -------
    ``proximal.BlackBox`` : proximal_lang_operator
        The wrapped operator.

    Notes
    -----
    If the data representation of ``op``'s domain and range is of type
    `NumpyTensorSpace` this incurs no significant overhead. If the data
    space is implemented with CUDA or some other non-local representation,
    the overhead is significant.

    References
    ----------
    [Hei+2016] Heide, F et al. *ProxImaL: Efficient Image Optimization using
    Proximal Algorithms*. ACM Transactions on Graphics (TOG), 2016.
    """
    # TODO: use out parameter once "as editable array" is added

    def forward(inp, out):
        out[:] = op(inp).asarray()

    def adjoint(inp, out):
        out[:] = op.adjoint(inp).asarray()

    import proximal
    return proximal.LinOpFactory(input_shape=op.domain.shape,
                                 output_shape=op.range.shape,
                                 forward=forward,
                                 adjoint=adjoint,
                                 norm_bound=norm_bound)


class auto_adjoint_weighting(OptionalArgDecorator):

    """Make an unweighted adjoint automatically account for weightings.

    This decorator can be used in `Operator` subclasses to automatically
    perform weighting of the adjoint operator based on the space
    weightings. It is used as ::

        class MyOperator(odl.Operator):

            @property
            @auto_adjoint_weighting  # MUST come after @property
            def adjoint(self):
                return MyOperatorUnweightedAdjoint(...)

    In this context, the term "unweighted adjoint" of an operator ``A``
    refers to the the adjoint of ``A`` defined between **unweighted
    Euclidean spaces**. In particular, discretized ``L^p`` type spaces are
    interpreted as weighted variants of ``R^n`` or ``C^n`` spaces.

    For instance, an (m x n) real matrix ``M`` is also an operator
    ``M: R^n -> R^m``, and its unweighted adjoint is defined by the
    transposed matrix ``M^T: R^m -> R^n``.

    Similarly, when an integration operator ``A: L^2(I) -> R`` on an
    interval ``I`` is discretized using ``n`` grid points, it can be seen
    as a summation operator ``S: R^n -> R, S(x) = dx * sum(x)`` scaled
    by the subinterval size ``dx``.
    Here, the unweighted adjoint is the adjoint ``S^*: R -> R^n`` of ``S``,
    given by ``S^*(c) = dx * (c, ..., c)``.
    The weighting to correctly define the *weighted* adjoint
    ``A^*: R -> L^2(I)`` is taken care of by this decorator.

    Parameters
    ----------
    unweighted_adjoint : function
        Method on an `Operator` class that returns the unweighted variant
        of the adjoint. It will be patched with a new ``_call()`` method.
        The weightings of ``domain`` and ``range`` of the operator
        must be `ArrayWeighting` or `ConstWeighting`.
    optimize : bool, optional
        If ``True``, merge and move around constant weightings for
        highest expected efficiency.

        **Note:** Merging of a constant weight and an array weight will
        result in a copy of the array, doubling the amount of required
        memory.

    Notes
    -----
    **Mathematical background:**

    Consider a linear operator :math:`A: X_w \\to Y_v` between spaces with
    weights :math:`w` and :math:`v`, respectively, along with the same
    operator :math:`B: X \\to Y` defined between the unweighted variants of
    the spaces. (This means that :math:`B f = A f` for all
    :math:`f \\in X \cong X_w`).

    Then, the adjoint of :math:`A` is related to the adjoint of :math:`B`
    as follows:

    .. math::
        \\langle Af, g \\rangle_{Y_v} =
        \\langle Bf, v \cdot g \\rangle_Y =
        \\langle f, B^*(v \cdot g) \\rangle_X =
        \\langle f, w^{-1}\, B^*(v \cdot g) \\rangle_{X_w}.

    Thus, from the existing unweighted adjoint :math:`B^*` one can compute
    the weighted one as :math:`A^* = w^{-1}\, B^*(v\, \cdot)`.

    **Example:**

    Consider the integration operator

    .. math::
        A: L^2(I) \\to \mathbb{R},\quad
        A(f) = \int_I f(x)\, \mathrm{d}x

    discretized as

    .. math::
        A_h: L^2_h(I) \\to \mathbb{R},\quad
        A_h(f) = h \sum_{i=1}^n f(x_i).

    We can interpret :math:`L^2_h(I)` as weighted space
    :math:`\mathbb{R}^n_w` with :math:`w = h`, and the operator as
    :math:`A_h(y) = h \sum_{i=1}^n y_i`.

    Now the unweighted adjoint is the adjoint of the same operator defined
    on unweighted spaces, i.e.,

    .. math::
        &B: \mathbb{R}^n \\to \mathbb{R},\quad
        B(y) = h \sum_{i=1}^n y_i,

        &B^*: \mathbb{R} \\to \mathbb{R}^n,\quad
        B^*(c) = h\, (c, \dots, c).

    The weighted adjoint is given by the formula

    .. math::
        A_h^*(c) = w^{-1} B^*(c) = (c, \dots, c)

    as expected.

    **Rules for weight simplification:**

    Depending on the weightings, the correction is achieved by composing
    the unweighted operator with either `ScalingOperator` or
    `ConstantOperator`. The following rules are applied for the domain
    weighting ``w``, the range weighting ``v`` and the provided unweighted
    adjoint ``B^*``:

    - If both ``w`` and ``v`` are arrays, return ::

        (1 / w) * (B^*) * v

    - If ``w`` is an array and ``v`` a constant, return ::

        (v / w) * (B^*)

    - If ``w`` is a constant and ``v`` an array, return ::

        (B^*) * (w / v)

    - If both ``w`` and ``v`` are constants, return ::

        (B^*) * (v / w)

      if ``B.range.size < B.domain.size``, otherwise ::

        (v / w) * (B^*)

    - Ignore constants 1.0.

    To avoid the inconvenience of dealing with `OperatorComp` objects,
    the given operator is monkey-patched instead of composed.
    """

    @staticmethod
    def _wrapper(unweighted_adjoint, optimize=True):
        """Return the weighted variant of the unweighted adjoint."""
        # Support decorating the `adjoint` property directly
        import inspect
        from functools import wraps
        from odl.operator.operator import Operator

        if (inspect.isfunction(unweighted_adjoint) and
                unweighted_adjoint.__name__ == 'adjoint'):

            # We need this level of indirection since `self` needs to be
            # filled in with the instance, but we decorate at class level
            @wraps(unweighted_adjoint)
            def weighted_adjoint(self):
                adj = unweighted_adjoint(self)
                if not isinstance(adj, Operator):
                    raise TypeError('`adjoint` did not return an `Operator`')
                if adj is self:
                    raise TypeError(
                        'returning `self` in an `adjoint` property using '
                        '`auto_adjoint_weighting` is not allowed')

                # This is for cached adjoints: don't double-wrap
                if hasattr(adj, '_call_unweighted'):
                    return adj
                else:
                    return auto_adjoint_weighting._instance_wrapper(
                        adj, optimize)

            return weighted_adjoint

        else:
            raise TypeError(
                "`auto_adjoint_weighting` can only be applied to 'adjoint' "
                'methods (as @auto_adjoint_weighting decorator); '
                'make sure that the (textual) order of decorators is '
                '`@property`, then `@auto_adjoint_weighting`, not the other '
                'way around')

    @staticmethod
    def _instance_wrapper(unweighted_adjoint, optimize=True):
        """Wrapper for `Operator` instances."""
        # Use notions of the original operator, not the adjoint
        dom_weighting = unweighted_adjoint.range.weighting
        ran_weighting = unweighted_adjoint.domain.weighting

        if isinstance(dom_weighting, ArrayWeighting):
            dom_w_type = 'array'
            dom_w = dom_weighting.array
        elif isinstance(dom_weighting, ConstWeighting):
            dom_w_type = 'const'
            dom_w = dom_weighting.const
        else:
            raise TypeError(
                'weighting of `unweighted_adjoint.range` must be of '
                'type `ArrayWeighting` or `ConstWeighting`, got {}'
                ''.format(type(dom_weighting)))

        if isinstance(ran_weighting, ArrayWeighting):
            ran_w_type = 'array'
            ran_w = ran_weighting.array
        elif isinstance(ran_weighting, ConstWeighting):
            ran_w_type = 'const'
            ran_w = ran_weighting.const
        else:
            raise TypeError(
                'weighting of `unweighted_adjoint.domain` must be of '
                'type `ArrayWeighting` or `ConstWeighting`, got {}'
                ''.format(type(ran_weighting)))

        # Compute the effective weights and mark constants 1.0 as to be
        # skipped
        if not optimize:
            new_dom_w, new_ran_w = dom_w, ran_w
            skip_dom = (dom_w_type == 'const' and dom_w == 1.0)
            skip_ran = (ran_w_type == 'const' and ran_w == 1.0)
        elif dom_w_type == 'array' and ran_w_type == 'array':
            new_dom_w, new_ran_w = dom_w, ran_w
            skip_dom = skip_ran = False
        elif dom_w_type == 'array' and ran_w_type == 'const':
            new_dom_w = dom_w / ran_w
            new_ran_w = 1.0
            skip_dom = False
            skip_ran = True
        elif dom_w_type == 'const' and ran_w_type == 'array':
            new_dom_w = 1.0
            new_ran_w = ran_w / dom_w
            skip_dom = True
            skip_ran = False
        elif dom_w_type == 'const' and ran_w_type == 'const':
            if unweighted_adjoint.domain.size < unweighted_adjoint.range.size:
                new_dom_w = 1.0
                new_ran_w = ran_w / dom_w
                skip_dom = True
                skip_ran = False
            else:
                new_dom_w = dom_w / ran_w
                new_ran_w = 1.0
                skip_dom = False
                skip_ran = True

        # Define the new `_call` depending on original signature
        self = unweighted_adjoint

        def mul_weight(x, w):
            """Multiplication with weights that works in product spaces."""
            if not isinstance(x, ProductSpaceElement) or np.isscalar(w):
                return w * x
            else:
                # Product space, array weight
                return x.space.element([mul_weight(xi, wi)
                                        for xi, wi in zip(x, w)])

        def idiv_weight(x, w):
            """In-place division by weights that works in product spaces."""
            if not isinstance(x, ProductSpaceElement) or np.isscalar(w):
                x /= w
            else:
                # Product space, array weight
                for xi, wi in zip(x, w):
                    idiv_weight(xi, wi)

        # Monkey-patching starts here
        if self._call_has_out and self._call_out_optional:
            def _call(x, out=None):
                if not skip_ran:
                    x = mul_weight(x, new_ran_w)
                out = self._call_unweighted(x, out=out)
                if not skip_dom:
                    idiv_weight(out, new_dom_w)
                return out

            self._call_unweighted = self._call_in_place
            self._call_in_place = self._call_out_of_place = _call

        elif self._call_has_out and not self._call_out_optional:
            def _call(x, out):
                if not skip_ran:
                    x = mul_weight(x, new_ran_w)
                self._call_unweighted(x, out=out)
                if not skip_dom:
                    idiv_weight(out, new_dom_w)
                return out

            self._call_unweighted = self._call_in_place
            self._call_in_place = _call

        else:
            def _call(x):
                if not skip_ran:
                    x = mul_weight(x, new_ran_w)
                out = self._call_unweighted(x)
                if not skip_dom:
                    idiv_weight(out, new_dom_w)
                return out

            self._call_unweighted = self._call_out_of_place
            self._call_out_of_place = _call

        return self


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
