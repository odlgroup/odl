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

""" Implementation of the forward-backward primal-dual splitting algorithm for
optimization.
"""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator import Operator


__all__ = ('forward_backward_pd',)


def forward_backward_pd(x, prox_f, prox_cc_g, L, grad_h, tau, sigma, niter,
                        callback=None, **kwargs):
    """ The forward-backward primal-dual splitting algorithm.

    The algorithm minimizes the sum of several convex functionals composed with
    linear operators,

        ``min_x f(x) + sum_i g_i(L_i x) + h(x)``,

    where ``f``, ``g_i`` are convex functionals, ``L_i`` are linear
    `Operator`'s, and ``h`` is a convex and differentiable functional.

    The method can also be used to solve the more general problem

        ``min_x f(x) + sum_i (g_i @ l_i)(L_i x)``,

    where ``l_i`` are convex functionals and @ is the infimal convolution:

        ``(f @ g)(x) = inf_y { f(x-y) + g(y) }``

    Parameters
    ----------
    x : `LinearSpaceVector`
        Initial point, updated in place.
    prox_f : `callable`
        `Proximal factory` for the functional ``f``.
    prox_cc_g : `sequence` of `callable`'s
        Sequence of `proximal factories` for the convex conjuates of the
        functionals ``g_i``.
    grad_h : `Operator`
        Operators representing the gradient of  `h`.
    L : `sequence` of `Operator`'s
        A sequence with as many elements as ``prox__cc_gs`` of operators
        ``L_i``
    tau : `float`
        Step size-like parameter for ``prox_f``
    sigma : `sequence` of  `float`
        Sequence of step size-like parameter for the sequence ``prox_cc_g``
    niter : `int`
        Number of iterations
    callback : `Callback`, optional
        Show partial results

    Other Parameters
    ----------------
    grad_cc_l : `sequence` of `Operator`'s, optional
        Sequence of operators representing the gradient of  `l_i^*`.
        If omitted, the simpler problem will be considered.

    Notes
    -----
    The exact conditions on the involved functionals are as follows: :math:`f`
    and :math:`g_i` are proper, convex and lower semicontinuous, and :math:`h`
    is convex and differentialbe with :math:`\\eta`-Lipschitz continuous
    gradient.

    The optional input :math:`\\nabla l_i^*` need to be :math:`\\nu_i`
    -Lipschitz continuous. Note that in the reference [BC2015]_, the condition
    is formulated as :math:`l_i` being proper, lower semicontinuous, and
    :math:`\\nu_i^{-1}`-strongly convex, which implies that :math:`l_i^*` have
    :math:`\\nu_i`-Lipschitz continuous gradient.

    If the optional input :math:`\\nabla l_i^*` is omitted, the simpler problem
    will be considered. Mathematically, this is done by taking :math:`l_i` to
    be the functional that is zero only in the zero element and :math:`\\infty`
    otherwise. This gives that :math:`l_i^*` is the zero functional, and hence
    the corresponding gradient is the zero operator.

    To guarantee convergence, the parameters ``tau``, ``sigma`` and
    ``L`` need to satisfy

    .. math::

       2 \min \{ \\frac{1}{\\tau}, \\frac{1}{\sigma_1}, \\ldots,
       \\frac{1}{\sigma_m} \} \cdot \min\{ \\eta, \\nu_1, \\ldots, \\nu_m  \}
       \cdot \\sqrt{1 - \\tau \\sum_{i=1}^n \\sigma_i ||L_i||^2} > 1,

    where, if the simpler problem is considered, all :math:`\\nu_i` can be
    considered to be :math:`\\infty`.

    For reference on the forward-backward primal-dual algorithm, see [BC2015]_.

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

    grad_cc_l = kwargs.pop('grad_cc_l', None)
    if grad_cc_l is not None and len(grad_cc_l) != m:
        raise ValueError('`grad_cc_l` not same length as `L`')

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
            if grad_cc_l is not None:
                # In this case gradients were given.
                tmp_2 = sigma[i] * (L[i](y) + grad_cc_l[i](v[i]))
            else:
                # In this case gradients were not given. Therefore the gradient
                # step is omitted. For more details, see the documentation.
                tmp_2 = sigma[i] * L[i](y)

            prox_cc_g[i](sigma[i])(v[i] + tmp_2, out=v[i])

        if callback is not None:
            callback(x)
