# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Douglas-Rachford splitting algorithm for convex optimization."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator import Operator


__all__ = ('douglas_rachford_pd',)


def douglas_rachford_pd(x, f, g, L, tau, sigma, niter,
                        callback=None, **kwargs):
    """Douglas-Rachford primal-dual splitting algorithm.

    Minimizes the sum of several convex functions composed with linear
    operators::

        min_x f(x) + sum_i g_i(L_i x)

    where ``f``, ``g_i`` are convex functions, ``L_i`` are linear `Operator`'s.

    Can also be used to solve the more general problem::

        min_x f(x) + sum_i (g_i @ l_i)(L_i x)

    where ``l_i`` are convex functions and ``@`` is the infimal convolution::

        (g @ l)(x) = inf_y g(y) + l(x - y)

    For references on the algorithm, see algorithm 3.1 in [BH2013].

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    f : `Functional`
        `proximal factory` for the function ``f``.
    g : sequence of `Functional`'s
        Sequence of of the functions ``g_i``. Needs to have
        ``g[i].convex_conj.proximal``.
    L : sequence of `Operator`'s
        Sequence of `Opeartor`s` with as many elements as ``g``.
    tau : float
        Step size parameter for ``f``.
    sigma : sequence of floats
        Step size parameters for the ``g_i``s.
    niter : int
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    l : sequence of `Functionals`'s, optional
        Sequence of of the functions ``l_i``. Needs to have
        ``l[i].convex_conj.proximal``.
        If omitted, the simpler problem without ``l_i``  will be considered.
    lam : float or callable, optional
        Overrelaxation step size. If callable, it should take an index
        (starting at zero) and return the corresponding step size.

    Notes
    -----
    The mathematical problem to solve is

    .. math::
       \min_x f(x) + \sum_{i=0}^n (g_i \Box l_i)(L_i x),

    where :math:`f`, :math:`g_i`, :math:`l_i` are proper, convex and lower
    semicontinuous and :math:`L_i` are linear operators. The infimal
    convolution :math:`g \Box l` is defined by

    .. math::
       (g \Box l)(x) = \inf_y g(y) + l(x - y).

    The simplified problem,

    .. math::
        \min_x f(x) + \sum_{i=0}^n g_i(L_i x),

    can be obtained by setting

    .. math::
        l(x) = 0 \\text{ if } x = 0, \infty \\text{ else.}

    To guarantee convergence, the parameters :math:`\\tau`, :math:`\\sigma_i`
    and :math:`L_i` need to satisfy

    .. math::
       \\tau \\sum_{i=1}^n \\sigma_i ||L_i||^2 < 4

    The parameter :math:`\\lambda` needs to satisfy :math:`0 < \\lambda < 2`
    and if it is given as a function it needs to satisfy

    .. math::
        \\sum_{n=1}^\infty \\lambda_n (2 - \\lambda_n) = +\infty.

    See Also
    --------
    odl.solvers.nonsmooth.chambolle_pock.chambolle_pock_solver :
        Solver for similar problems.
    odl.solvers.nonsmooth.forward_backward.forward_backward_pd :
        Solver for similar problems which can additionaly handle a
        differentiable term.

    References
    ----------
    [BH2013] Bot, R I, and Hendrich, C. *A Douglas-Rachford type
    primal-dual method for solving inclusions with mixtures of
    composite and parallel-sum type monotone operators*. SIAM Journal
    on Optimization, 23.4 (2013), pp 2541--2565.
    """
    # Problem size
    m = len(L)

    # Validate input
    if not all(isinstance(op, Operator) for op in L):
        raise ValueError('`L` not a sequence of operators')
    if not all(op.is_linear for op in L):
        raise ValueError('not all operators in `L` are linear')
    if not all(x in op.domain for op in L):
        raise ValueError('`x` not in the domain of all operators')
    if len(sigma) != m:
        raise ValueError('len(sigma) != len(L)')
    if len(g) != m:
        raise ValueError('len(prox_cc_g) != len(L)')

    prox_cc_g = [gi.convex_conj.proximal for gi in g]

    # Get parameters from kwargs
    l = kwargs.pop('l', None)
    if l is not None and len(l) != m:
        raise ValueError('`l` does not have the same number of '
                         'elements as `L`')
    if l is not None:
        prox_cc_l = [li.convex_conj.proximal for li in l]

    lam_in = kwargs.pop('lam', 1.0)
    if not callable(lam_in) and not (0 < lam_in < 2):
        raise ValueError('`lam` must callable or a number between 0 and 2')
    lam = lam_in if callable(lam_in) else lambda _: lam_in

    # Check for unused parameters
    if kwargs:
        raise TypeError('unexpected keyword argument: {}'.format(kwargs))

    # Pre-allocate values
    v = [Li.range.zero() for Li in L]
    p1 = x.space.zero()
    p2 = [Li.range.zero() for Li in L]
    z1 = x.space.zero()
    z2 = [Li.range.zero() for Li in L]
    w1 = x.space.zero()
    w2 = [Li.range.zero() for Li in L]

    # Temporaries (not in original article)
    tmp_domain = x.space.zero()

    for k in range(niter):
        lam_k = lam(k)

        if len(L) > 0:
            # Compute tmp_domain = sum(Li.adjoint(vi) for Li, vi in zip(L, v))
            L[0].adjoint(v[0], out=tmp_domain)
            for Li, vi in zip(L[1:], v[1:]):
                Li.adjoint(vi, out=p1)
                tmp_domain += p1

            tmp_domain.lincomb(1, x, -tau / 2, tmp_domain)
        else:
            tmp_domain.assign(x)

        f.proximal(tau)(tmp_domain, out=p1)
        w1.lincomb(2, p1, -1, x)

        for i in range(m):
            tmp = v[i] + (sigma[i] / 2.0) * L[i](w1)
            prox_cc_g[i](sigma[i])(tmp, out=p2[i])
            w2[i].lincomb(2.0, p2[i], -1, v[i])

        if len(L) > 0:
            # Compute:
            # tmp_domain = sum(Li.adjoint(w2i) for Li, w2i in zip(L, w2))
            L[0].adjoint(w2[0], out=tmp_domain)
            for Li, w2i in zip(L[1:], w2[1:]):
                Li.adjoint(w2i, out=z1)
                tmp_domain += z1
        else:
            tmp_domain.set_zero()

        z1.lincomb(1.0, w1, - (tau / 2.0), tmp_domain)

        # Compute x += lam(k) * (z1 - p1)
        x.lincomb(1, x, lam_k, z1)
        x.lincomb(1, x, -lam_k, p1)

        tmp_domain.lincomb(2, z1, -1, w1)
        for i in range(m):
            if l is not None:
                # In this case the infimal convolution is used.
                tmp = w2[i] + (sigma[i] / 2.0) * L[i](tmp_domain)
                prox_cc_l[i](sigma[i])(tmp, out=z2[i])
            else:
                # If the infimal convolution is not given, prox_cc_l is the
                # identity and hence omitted. For more details, see the
                # documentation.
                z2[i].lincomb(1, w2[i], sigma[i] / 2.0, L[i](tmp_domain))

            # Compute v[i] += lam(k) * (z2[i] - p2[i])
            v[i].lincomb(1, v[i], lam_k, z2[i])
            v[i].lincomb(1, v[i], -lam_k, p2[i])

        if callback is not None:
            callback(p1)

    # The final result is actually in p1 according to the algorithm, so we need
    # to assign here.
    x.assign(p1)
