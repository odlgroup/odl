# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Maximum Likelihood Expectation Maximization algorithm."""

from __future__ import print_function, division, absolute_import
import numpy as np

__all__ = ('mlem', 'osmlem', 'poisson_log_likelihood')


def mlem(op, x, data, niter, callback=None, **kwargs):

    """Maximum Likelihood Expectation Maximation algorithm.

    Attempts to solve::

        max_x L(x | data)

    where ``L(x | data)`` is the Poisson likelihood of ``x`` given ``data``.
    The likelihood depends on the forward operator ``op`` such that
    (approximately)::

        op(x) = data


    Parameters
    ----------
    op : `Operator`
        Forward operator in the inverse problem.
    x : ``op.domain`` element
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
        The initial value of ``x`` should be non-negative.
    data : ``op.range`` `element-like`
        Right-hand side of the equation defining the inverse problem.
    niter : int
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    sensitivities : float or ``op.domain`` `element-like`, optional
        The algorithm contains a ``A^T 1``
        term, if this parameter is given, it is replaced by it.
        Default: ``op.adjoint(op.range.one())``

    Notes
    -----
    Given a forward model :math:`A` and data :math:`g`,
    the algorithm attempts to find an :math:`x` that maximizes:

    .. math::
        P(g | g \text{ is } Poisson(A(x)) \text{ distributed}).

    The algorithm is explicitly given by:

    .. math::
       x_{n+1} = \frac{x_n}{A^* 1} A^* (g / A(x_n))

    See Also
    --------
    osmlem : Ordered subsets MLEM
    loglikelihood : Function for calculating the logarithm of the likelihood
    """
    osmlem([op], x, [data], niter=niter, callback=callback,
           **kwargs)


def osmlem(op, x, data, niter, callback=None, **kwargs):
    r"""Ordered Subsets Maximum Likelihood Expectation Maximation algorithm.

    This solver attempts to solve::

        max_x L(x | data)

    where ``L(x, | data)`` is the likelihood of ``x`` given ``data``. The
    likelihood depends on the forward operators ``op[0], ..., op[n-1]`` such
    that (approximately)::

        op[i](x) = data[i]


    Parameters
    ----------
    op : sequence of `Operator`
        Forward operators in the inverse problem.
    x : ``op.domain`` element
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
        The initial value of ``x`` should be non-negative.
      data : sequence of ``op.range`` `element-like`
        Right-hand sides of the equation defining the inverse problem.
    niter : int
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    sensitivities : float or ``op.domain`` `element-like`, optional
        The algorithm contains an ``A^T 1``
        term, if this parameter is given, it is replaced by it.
        Default: ``op[i].adjoint(op[i].range.one())``

    Notes
    -----
    Given forward models :math:`A_i`, and data :math:`g_i`,
    :math:`i = 1, ..., M`,
    the algorithm attempts to find an :math:`x` that
    maximizes:

    .. math::
        \prod_{i=1}^M P(g_i | g_i \text{ is }
        Poisson(A_i(x)) \text{ distributed}).

    The algorithm is explicitly given by partial updates:

    .. math::
       x_{n + m/M} =
       \frac{x_{n + (m - 1)/M}}{A_i^* 1} A_i^* (g_i / A_i(x_{n + (m - 1)/M}))

    for :math:`m = 1, ..., M` and :math:`x_{n+1} = x_{n + M/M}`.

    The algorithm is not guaranteed to converge, but works for many practical
    problems.

    References
    ----------
    Natterer, F. Mathematical Methods in Image Reconstruction, section 5.3.2.

    See Also
    --------
    mlem : Ordinary MLEM algorithm without subsets.
    loglikelihood : Function for calculating the logarithm of the likelihood
    """
    n_ops = len(op)
    if len(data) != n_ops:
        raise ValueError('number of data ({}) does not match number of '
                         'operators ({})'.format(len(data), n_ops))
    if not all(x in opi.domain for opi in op):
        raise ValueError('`x` not an element in the domains of all operators')

    # Convert data to range elements
    data = [op[i].range.element(data[i]) for i in range(len(op))]

    # Parameter used to enforce positivity.
    # TODO: let users give this.
    eps = 1e-8

    if np.any(np.less(x, 0)):
        raise ValueError('`x` must be non-negative')

    # Extract the sensitivites parameter
    sensitivities = kwargs.pop('sensitivities', None)
    if sensitivities is None:
        sensitivities = [np.maximum(opi.adjoint(opi.range.one()), eps)
                         for opi in op]
    else:
        # Make sure the sensitivities is a list of the correct size.
        try:
            list(sensitivities)
        except TypeError:
            sensitivities = [sensitivities] * n_ops

    tmp_dom = op[0].domain.element()
    tmp_ran = [opi.range.element() for opi in op]

    for _ in range(niter):
        for i in range(n_ops):
            op[i](x, out=tmp_ran[i])
            tmp_ran[i].ufuncs.maximum(eps, out=tmp_ran[i])
            data[i].divide(tmp_ran[i], out=tmp_ran[i])

            op[i].adjoint(tmp_ran[i], out=tmp_dom)
            tmp_dom /= sensitivities[i]

            x *= tmp_dom

            if callback is not None:
                callback(x)


def poisson_log_likelihood(x, data):
    """Poisson log-likelihood of ``data`` given noise parametrized by ``x``.

    Parameters
    ----------
    x : ``op.domain`` element
        Value to condition the log-likelihood on.
    data : ``op.range`` element
        Data whose log-likelihood given ``x`` shall be calculated.
    """
    if np.any(np.less(x, 0)):
        raise ValueError('`x` must be non-negative')

    return np.sum(data * np.log(x + 1e-8) - x)
