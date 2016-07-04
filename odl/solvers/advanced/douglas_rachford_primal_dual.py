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

""" Implementation of the Douglas-Rachford splitting algorithm """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


from odl.operator import Operator


__all__ = ('douglas_rachford_pd',)


def douglas_rachford_pd(x, prox_f, prox_cc_g, L, tau, sigma, lam, niter,
                        callback=None, **kwargs):
    """Douglas-Rachford primal-dual splitting algorithm.

    Minimizes the sum of several convex functions composed with linear
    operators

        min_x f(x) + \sum_i g_i(L_i x)

    Where f, g_i are convex functions, L_i are linear operators.

    Can also be used to solve the more general problem

        min_x f(x) + \sum_i (g_i @ l_i)(L_i x)

    Where l_i are convex functions and  @ is the infimal convolution:

        (f @ g)(x) = inf_y { f(x-y) + g(y) }

    Parameters
    ----------
    x : `LinearSpaceVector`
        Initial point, updated in place.
    prox_f : `callable`
        Funciton returning an `Operator` when called with `tau`.
        The Operator should be the proximal of `tau * f`.
    prox_cc_g : `sequence` of `callable`'s
        Sequence of functions returning an operator when called with stepsize.
        The `Operator` should be the proximal of ``g_i^*``.
    L : `sequence` of `Operator`'s
        A sequence with as many elements as ``prox_cc_gs`` of operators
        ``L_i``.
    tau : `float`
        Stepsize of ``f``.
    sigma : `sequence` of  `float`
        Stepsize of the ``g_i``'s.
    lam : `float`
        Step size.
    niter : `int`
        Number of iterations.
    callback : `Callback`, optional
        Show partial results.

    Other Parameters
    ----------------
    prox_cc_l : `sequence` of `callable`'s
        Sequence of functions returning an operator when called with stepsize.
        The `Operator` should be the proximal of ``l_i^*``.

    Notes
    -----
    To guarantee convergence, the parameters ``tau``, ``sigma`` and ``L`` need
    to satisfy

    .. math::

       \\tau \sum_{i=1}^n \\sigma_i ||L_i||^2 < 4

    References
    ----------
    For references on the Forward-Backward algorithm, see [BH2013]_.

    For more on convex analysis including convex conjugates and
    resolvent operators see [Roc1970]_.

    For more on proximal operators and algorithms see [PB2014]_.
    """

    # Problem size
    m = len(L)
    if not all(isinstance(op, Operator) for op in L):
        raise ValueError('`L` not a sequence of operators')
    if not all(op.is_linear for op in L):
        raise ValueError('Not all operators are linear')
    if not all(x in op.domain for op in L):
        raise ValueError('`x` not in the domain of all operators')
    if len(sigma) != m:
        raise ValueError('`sigma` not same length as `L`')
    if len(prox_cc_g) != m:
        raise ValueError('`prox_cc_g` not same length as `L`')

    # Get parameters
    prox_cc_l = kwargs.pop('prox_cc_l', None)
    if prox_cc_l is not None and len(prox_cc_l) != m:
        raise ValueError('`prox_cc_l` not same length as `L`')

    # Check for unused parameters
    if kwargs:
        raise TypeError('Unexpected **kwargs: {}'.format(kwargs))

    # Pre-allocate values
    v = [Li.range.zero() for Li in L]

    p1 = x.space.zero()
    p2 = [Li.range.zero() for Li in L]

    z1 = x.space.zero()
    z2 = [Li.domain.zero() for Li in L]

    w1 = x.space.zero()
    w2 = [Li.range.zero() for Li in L]

    # Iteratively solve
    for k in range(niter):
        tmp_1 = sum(Li.adjoint(vi) for Li, vi in zip(L, v))
        p1 = prox_f(tau)(x - (tau / 2.0) * tmp_1)
        w1 = 2.0 * p1 - x

        for i in range(m):
            p2[i] = prox_cc_g[i](sigma[i])(v[i] + (sigma[i] / 2.0) * L[i](w1))
            w2[i] = 2.0 * p2[i] - v[i]

        tmp_2 = sum(Li.adjoint(wi) for Li, wi in zip(L, w2))
        z1 = w1 - (tau / 2.0) * tmp_2
        x += lam * (z1 - p1)

        for i in range(m):
            tmp = w2[i] + (sigma[i] / 2.0) * L[i](2.0 * z1 - w1)
            if prox_cc_l is not None:
                z2[i] = prox_cc_l[i](sigma[i])(tmp)
            else:
                z2[i] = tmp
            v[i] += lam * (z2[i] - p2[i])

        if callback is not None:
            callback(p1)

    x.assign(p1)
