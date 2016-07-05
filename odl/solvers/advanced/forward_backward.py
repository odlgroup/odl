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

""" Implementation of the Forward-Backward splitting algorithm """


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator import Operator


__all__ = ('forward_backward_pd',)


def forward_backward_pd(x, prox_f, prox_cc_g, L, grad_h, tau, sigma, niter,
                        callback=None, **kwargs):
    """ The forward-backward primal-dual splitting algorithm

    Minimizes the sum of several convex functions composed with linear
    operators

        ``min_x f(x) + sum_i g_i(L_i x) + h(x)``

    Where f, g_i, h are convex functions, L_i are linear `Operator`'s, and h is
    also differentiable.

    Can also be used to solve the more general problem

        ``min_x f(x) + sum_i (g_i @ l_i)(L_i x)``

    where l_i are convex functions and @ is the infimal convolution:

        ``(f @ g)(x) = inf_y { f(x-y) + g(y) }``

    Parameters
    ----------
    x : `Vector`
        Initial point
    prox_f : `callable`
        Funciton returning an `Operator` when called with `tau`.
        The Operator should be the proximal of `tau * f`.
    prox_cc_g : `sequence` of `callable`'s
        Sequence of functions returning an operator when called with stepsize
        `sigma`.
        The Operator should be the proximal of `sigma_i * g_i^*`.
    grad_h : `Operator`
        Operators representing the gradient of  `h`.
    L : `sequence` of `Operator`'s
        A sequence with as many elements as ``prox__cc_gs`` of operators
        ``L_i``
    tau : `float`
        Stepsize of ``f``
    sigma : `sequence` of  `float`
        Stepsize of the ``g_i``'s
    niter : `int`
        Number of iterations
    callback : `Callback`, optional
        Show partial results

    Other Parameters
    ----------------
    grad_cc_l : `sequence` of `Operator`'s, optional
        Sequence of operators representing the gradient of  `l_i^*`.
        If omitted, the simpler problem will be considered, which corresponds
        to the convex conjugate functionals of the l_i:s being zero functionals
        and hence the gradient being the zero-operator.

    Notes
    -----
    Strictly, we have the following conditions on the functions involved: f and
    g_i are proper, convex and lower semicontinuous, and h is convex and
    differentialbe with :math:`\\eta`-Lipschitz continuous gradient.

    For the optional input l_i we need it to be proper, convex, lower
    semicontinuous, and :math:`\\nu_i^{-1}`-strongly convex. The fact that l_i
    is :math:`\\nu_i^{-1}`-strongly convex implies that the convex conjugate
    functional of l_i is differentiable with :math:`\\nu_i`-Lipschitz
    continuous gradient.

    To guarantee convergence, the parameters ``tau``, ``sigma`` and
    ``L`` need to satisfy

    .. math::

       2 * \min \{ \\frac{1}{\\tau}, \\frac{1}{\sigma_1}, \\ldots,
       \\frac{1}{\sigma_m} \} * \min\{ \\eta, \\nu_1, \\ldots, \\nu_m  \} *
       \\sqrt{1 - \\tau \\sum_{i=1}^n \\sigma_i ||L_i||^2} > 1

    where, if the simpler problem is considered, all :math:`\\nu_i` can be
    considered to be :math:`\\infty`.

    For references on the Forward-Backward algorithm, see [BC2015]_.

    For more on convex analysis including convex conjugates and
    resolvent operators see [Roc1970]_.

    For more on proximal operators and algorithms see [PB2014]_.
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
    if len(prox_cc_g) != m:
        raise ValueError('len(prox_cc_g) != len(L)')

    # Get parameters from kwargs
    grad_cc_l = kwargs.pop('prox_cc_l', None)
    if grad_cc_l is not None and len(grad_cc_l) != m:
        raise ValueError('`grad_cc_l` not same length as `L`')

    # Check for unused parameters
    if kwargs:
        raise TypeError('unexpected keyword argument: {}'.format(kwargs))

    # Pre-allocate values
    v = [Li.range.zero() for Li in L]
    y = x.space.zero()

    # Iteratively solve
    for k in range(niter):
        x_old = x

        tmp_1 = grad_h(x) + sum(Li.adjoint(vi) for Li, vi in zip(L, v))
        prox_f(tau)(x - tau * tmp_1, out=x)
        y.lincomb(2.0, x, -1, x_old)

        for i in range(m):
            # In this case gradients were given
            if grad_cc_l is not None:
                tmp_2 = sigma[i] * (L[i](y) + grad_cc_l[i](v[i]))

            # In this case there were not. "Applying" zero-operator
            else:
                tmp_2 = sigma[i] * L[i](y)

            v[i] = prox_cc_g[i](sigma[i])(v[i] + tmp_2)

        if callback is not None:
            callback(x)
