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

"""Convenience functions for operators."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from future.utils import raise_from
standard_library.install_aliases()

from math import ceil, sqrt
import numpy as np

from odl.space.base_ntuples import FnBase
from odl.space.pspace import ProductSpace

__all__ = ('matrix_representation', 'power_method_opnorm', 'as_scipy_operator')


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
    matrix = np.zeros([np.sum(n), np.sum(m)])
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
                        tmp_ran[k])
                    tmp_idx += op.range[k].size
            else:
                matrix[:, index] = tmp_ran.asarray()
            index += 1
            last_j = j
            last_i = i

    return matrix


def power_method_opnorm(op, niter, xstart=None):
    """Estimate the operator norm with the power method.

    Parameters
    ----------
    op : `Operator`
        Operator whose norm is to be estimated. If its `Operator.range`
        range does not coincide with its `Operator.domain`, an
        `Operator.adjoint` must be defined (which implies that the
        operator must be linear).
    niter : positive `int`
        Number of iterations to perform
    xstart : `Operator.domain` `element`, optional
        Starting point of the iteration. By default, the ``one``
        element of the `Operator.domain` is used.

    Returns
    -------
    est_norm : `float`
        The estimated operator norm
    """
    if op.domain == op.range:
        use_normal = False
    else:
        use_normal = True

    if xstart is None:
        try:
            x = op.domain.one()  # TODO: random? better choice?
        except AttributeError as exc:
            raise_from(
                ValueError('a starting element must be defined in case the '
                           'operator domain has no `one()`'), exc)
    else:
        x = xstart.copy()

    tmp = op.range.element()

    if use_normal:
        # Do the power iteration for A*A; the norm of A*A(x_N) is then
        # an estimate of the square of the operator norm
        # We do only half the number of iterations compared to the usual
        # case to have the same number of operator evaluations.
        niter = ceil(niter / 2)
        for _ in range(niter):
            op(x, out=tmp)
            op.adjoint(tmp, out=x)
            x /= x.norm()

        op(x, out=tmp)
        op.adjoint(tmp, out=x)
        return sqrt(x.norm())

    else:
        for _ in range(niter):
            op(x, out=tmp)
            x, tmp = tmp, x
            x /= x.norm()

        op(x, out=tmp)
        return tmp.norm()


def as_scipy_operator(op):
    """Wrap ``op`` as a ``scipy.sparse.linalg.LinearOperator.``

    This is intended to be used with the scipy sparse linear solvers.

    Parameters
    ----------
    op : `Operator`
        A linear operator that should be wrapped

    Returns
    -------
    ``scipy.sparse.linalg.LinearOperator`` : linear_op
        The wrapped operator, has attributes ``matvec`` which calls ``op``, and
        ``rmatvec`` which calls ``op.adjoint``.

    Examples
    --------
    Wrap operator and solve simple problem (here toy problem ``Ix = b``)

    >>> import odl
    >>> op = op = odl.IdentityOperator(odl.Rn(3))
    >>> scipy_op = as_scipy_operator(op)
    >>> import scipy.sparse.linalg as sl
    >>> result, status = sl.cg(scipy_op, [0, 1, 0])
    >>> result
    array([ 0.,  1.,  0.])

    Notes
    -----
    If the data representation of ``op``'s domain and range is of type `Fn`
    this incurs no significant overhead. If the data type is `CudaFn` or other
    nonlocal type, the overhead is significant.
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

    shape = (op.range.size, op.domain.size)

    def matvec(v):
        return as_flat_array(op(v))

    def rmatvec(v):
        return as_flat_array(op.adjoint(v))

    import scipy.sparse.linalg
    return scipy.sparse.linalg.LinearOperator(shape=shape,
                                              matvec=matvec,
                                              rmatvec=rmatvec,
                                              dtype=dtype)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
