# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Convenience functions for operators."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from future.utils import raise_from, native
standard_library.install_aliases()

import numpy as np

from odl.space.base_ntuples import FnBase
from odl.space import ProductSpace
from odl.util import as_flat_array

__all__ = ('matrix_representation', 'power_method_opnorm', 'as_scipy_operator',
           'as_scipy_functional', 'as_proximal_lang_operator')


def matrix_representation(op):
    """Return a matrix representation of a linear operator.

    Parameters
    ----------
    op : `Operator`
        The linear operator of which one wants a matrix representation.

    Returns
    ----------
    matrix : `numpy.ndarray`
        The matrix representation of the operator.

    Notes
    ----------
    The algorithm works by letting the operator act on all unit vectors, and
    stacking the output as a matrix.
    """

    if not op.is_linear:
        raise ValueError('the operator is not linear')

    if not (isinstance(op.domain, FnBase) or
            (isinstance(op.domain, ProductSpace) and
             all(isinstance(spc, FnBase) for spc in op.domain))):
        raise TypeError('operator domain {!r} is not FnBase, nor ProductSpace '
                        'with only FnBase components'.format(op.domain))

    if not (isinstance(op.range, FnBase) or
            (isinstance(op.range, ProductSpace) and
             all(isinstance(spc, FnBase) for spc in op.range))):
        raise TypeError('operator range {!r} is not FnBase, nor ProductSpace '
                        'with only FnBase components'.format(op.range))

    # Get the size of the range, and handle ProductSpace
    # Store for reuse in loop
    op_ran_is_prod_space = isinstance(op.range, ProductSpace)
    if op_ran_is_prod_space:
        num_ran = op.range.size
        n = [ran.size for ran in op.range]
    else:
        num_ran = 1
        n = [op.range.size]

    # Get the size of the domain, and handle ProductSpace
    # Store for reuse in loop
    op_dom_is_prod_space = isinstance(op.domain, ProductSpace)
    if op_dom_is_prod_space:
        num_dom = op.domain.size
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
                        as_flat_array(tmp_ran[k]))
                    tmp_idx += op.range[k].size
            else:
                matrix[:, index] = as_flat_array(tmp_ran)
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
        except AttributeError as exc:
            raise_from(
                ValueError('`xstart` must be defined in case the '
                           'operator domain has no `one()`'), exc)
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
    `NumpyFn` this incurs no significant overhead. If the space type is
    ``CudaFn`` or some other nonlocal type, the overhead is significant.
    """
    if not op.is_linear:
        raise ValueError('`op` needs to be linear')

    dtype = op.domain.dtype
    if op.range.dtype != dtype:
        raise ValueError('dtypes of ``op.domain`` and ``op.range`` needs to '
                         'match')

    def as_flat_array(arr):
        if hasattr(arr, 'order'):
            return arr.asarray().ravel(arr.order)
        else:
            return arr.asarray()

    shape = (native(op.range.size), native(op.domain.size))

    def matvec(v):
        return as_flat_array(op(v))

    def rmatvec(v):
        return as_flat_array(op.adjoint(v))

    import scipy.sparse.linalg
    return scipy.sparse.linalg.LinearOperator(shape=shape,
                                              matvec=matvec,
                                              rmatvec=rmatvec,
                                              dtype=dtype)


def as_scipy_functional(func, return_gradient=False):
    """Wrap ``op`` as a function operating on linear arrays.

    This is intended to be used with the scipy solvers.

    Parameters
    ----------
    func : `Functional`
        A functional that should be wrapped
    return_gradient : bool, optional
        If the gradient of the functional should also be returned

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
    >>> scipy_func = as_scipy_functional(func)
    >>> from scipy.optimize import minimize
    >>> result = minimize(scipy_func, [0, 1, 0])
    >>> np.allclose(result.x, [0, 0, 0])
    True

    Can also provide gradient:

    >>> func = odl.solvers.L2NormSquared(odl.rn(3))
    >>> scipy_func, scipy_grad = as_scipy_functional(func, True)
    >>> from scipy.optimize import minimize
    >>> result = minimize(scipy_func, [0, 1, 0], jac=scipy_grad)
    >>> np.allclose(result.x, [0, 0, 0])
    True

    Notes
    -----
    If the data representation of ``op``'s domain and range is of type
    `NumpyFn` this incurs no significant overhead. If the space type is
    ``CudaFn`` or some other nonlocal type, the overhead is significant.
    """
    def as_shaped_array(arr):
        if hasattr(func.domain, 'order'):
            return np.asarray(arr).reshape(func.domain.order)
        else:
            return np.asarray(arr)

    def as_flat_array(vec):
        if hasattr(vec, 'order'):
            return np.asarray(vec).ravel(vec.order)
        else:
            return np.asarray(vec)

    def func_call(arr):
        return func(as_shaped_array(arr))

    if return_gradient:
        def func_gradient_call(arr):
            return as_flat_array(func.gradient(as_shaped_array(arr)))

        return func_call, func_gradient_call
    else:
        return func_call


def as_proximal_lang_operator(op, norm_bound=None):
    """Wrap ``op`` as a ``proximal.BlackBox``.

    This is intended to be used with the `ProxImaL language solvers.
    <https://github.com/comp-imaging/proximal>`_

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
    `NumpyFn` this incurs no significant overhead. If the data space is
    ``CudaFn`` or some other nonlocal type, the overhead is significant.

    References
    ----------
    For documentation on the proximal language (ProxImaL) see [Hei+2016]_.
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

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
