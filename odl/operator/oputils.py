# Copyright 2014-2019 The ODL contributors
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
from odl.space import ProductSpace
from odl.util import nd_iterator
from odl.util.testutils import noise_element

__all__ = ('matrix_representation', 'power_method_opnorm', 'as_scipy_operator',
           'as_scipy_functional', 'as_proximal_lang_operator')


def matrix_representation(op):
    """Return a matrix representation of a linear operator.

    Parameters
    ----------
    op : `Operator`
        The linear operator of which one wants a matrix representation.
        If the domain or range is a `ProductSpace`, it must be a power-space.

    Returns
    -------
    matrix : `numpy.ndarray`
        The matrix representation of the operator.
        The shape will be ``op.domain.shape + op.range.shape`` and the dtype
        is the promoted (greatest) dtype of the domain and range.

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

    It also works with `ProductSpace`'s and higher dimensional `TensorSpace`'s.
    In this case, the returned "matrix" will also be higher dimensional:

    >>> space = odl.uniform_discr([0, 0], [2, 2], (2, 2))
    >>> grad = odl.Gradient(space)
    >>> tensor = odl.matrix_representation(grad)
    >>> tensor.shape == (2, 2, 2, 2, 2)
    True

    Since the "matrix" is now higher dimensional, we need to use e.g.
    `numpy.tensordot` if we want to compute with the matrix representation:

    >>> x = space.element(lambda x: x[0] ** 2 + 2 * x[1] ** 2)
    >>> grad(x)
    ProductSpace(uniform_discr([ 0.,  0.], [ 2.,  2.], (2, 2)), 2).element([
    <BLANKLINE>
            [[ 2.  ,  2.  ],
             [-2.75, -6.75]],
    <BLANKLINE>
            [[ 4.  , -4.75],
             [ 4.  , -6.75]]
    ])
    >>> np.tensordot(tensor, x, axes=grad.domain.ndim)
    array([[[ 2.  ,  2.  ],
            [-2.75, -6.75]],
    <BLANKLINE>
           [[ 4.  , -4.75],
            [ 4.  , -6.75]]])

    Notes
    ----------
    The algorithm works by letting the operator act on all unit vectors, and
    stacking the output as a matrix.
    """

    if not op.is_linear:
        raise ValueError('the operator is not linear')

    if not (isinstance(op.domain, TensorSpace) or
            (isinstance(op.domain, ProductSpace) and
             op.domain.is_power_space and
             all(isinstance(spc, TensorSpace) for spc in op.domain))):
        raise TypeError('operator domain {!r} is neither `TensorSpace` '
                        'nor `ProductSpace` with only equal `TensorSpace` '
                        'components'.format(op.domain))

    if not (isinstance(op.range, TensorSpace) or
            (isinstance(op.range, ProductSpace) and
             op.range.is_power_space and
             all(isinstance(spc, TensorSpace) for spc in op.range))):
        raise TypeError('operator range {!r} is neither `TensorSpace` '
                        'nor `ProductSpace` with only equal `TensorSpace` '
                        'components'.format(op.range))

    # Generate the matrix
    dtype = np.promote_types(op.domain.dtype, op.range.dtype)
    matrix = np.zeros(op.range.shape + op.domain.shape, dtype=dtype)
    tmp_ran = op.range.element()  # Store for reuse in loop
    tmp_dom = op.domain.zero()  # Store for reuse in loop

    for j in nd_iterator(op.domain.shape):
        tmp_dom[j] = 1.0

        op(tmp_dom, out=tmp_ran)
        matrix[(Ellipsis,) + j] = tmp_ran.asarray()

        tmp_dom[j] = 0.0

    return matrix


def power_method_opnorm(op, xstart=None, maxiter=100, rtol=1e-05, atol=1e-08,
                        callback=None):
    r"""Estimate the operator norm with the power method.

    Parameters
    ----------
    op : `Operator`
        Operator whose norm is to be estimated. If its `Operator.range`
        range does not coincide with its `Operator.domain`, an
        `Operator.adjoint` must be defined (which implies that the
        operator must be linear).
    xstart : ``op.domain`` `element-like`, optional
        Starting point of the iteration. By default an `Operator.domain`
        element containing noise is used.
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
    Verify that the identity operator has norm close to 1:

    >>> space = odl.uniform_discr(0, 1, 5)
    >>> id = odl.IdentityOperator(space)
    >>> estimation = power_method_opnorm(id)
    >>> round(estimation, ndigits=3)
    1.0

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
        x = noise_element(op.domain)
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

        # If the breaking condition holds, stop. Else rescale and go on.
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
    >>> import scipy.sparse.linalg as scipy_solvers
    >>> result, status = scipy_solvers.cg(scipy_op, [0, 1, 0])
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
    If the data representation of ``op``'s domain is of type
    `NumpyTensorSpace`, this incurs no significant overhead. If the space type
    is ``CudaFn`` or some other nonlocal type, the overhead is significant.
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


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
