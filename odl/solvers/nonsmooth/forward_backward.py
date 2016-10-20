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

"""Optimization methods based on a forward-backward splitting scheme."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from odl.operator import Operator


__all__ = ('forward_backward_pd',)


def forward_backward_pd(x, f, g, L, h, tau, sigma, niter,
                        callback=None, **kwargs):
    """The forward-backward primal-dual splitting algorithm.

    The algorithm minimizes the sum of several convex functionals composed with
    linear operators,::

        min_x f(x) + sum_i g_i(L_i x) + h(x)

    where ``f``, ``g_i`` are convex functionals, ``L_i`` are linear
    operator's, and ``h`` is a convex and differentiable functional.

    The method can also be used to solve the more general problem::

        min_x f(x) + sum_i (g_i @ l_i)(L_i x) + h(x)

    where ``l_i`` are strongly convex functionals and @ is the infimal
    convolution::

        (g @ l)(x) = inf_y { g(y) + l(x-y) }

    Note that the strong convexity of ``l_i`` makes the convex conjugate
    ``l_i^*`` differentialbe; see the Notes section for more information on
    this.

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    f : `Functional`
        The functional ``f``. Needs to have ``f.proximal``.
    g : sequence of `Functional`'s
        The functionals ``g_i``. Needs to have ``g_i.convex_conj.proximal``.
    L : sequence of `Operator`'s'
        Sequence of linear operators ``L_i``, with as many elements as
        ``prox_cc_gs``.
    h : `Functional`
        The functional ``h``. Needs to have ``h.gradient``.
    tau : float
        Step size-like parameter for ``prox_f``.
    sigma : sequence of floats
        Sequence of step size-like parameters for the sequence ``prox_cc_g``.
    niter : int
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    l : sequence of `Functional`'s, optional
        The functionals ``l_i``. Needs to have ``g_i.convex_conj.gradient``.
        If omitted, the simpler problem without ``l_i``  will be considered.

    Notes
    -----
    The mathematical problem to solve is

     .. math::

        \min_x f(x) + \sum_{i=0}^n (g_i \Box l_i)(L_i x) + h(x),

    where :math:`f`, :math:`g_i`, :math:`l_i` and :math:`h` are functionals and
    :math:`L_i` are linear operators. The infimal convolution :math:`g \Box l`
    is defined by

     .. math::

        (g \Box l)(x) = \inf_y g(y) + l(x - y).

    The exact conditions on the involved functionals are as follows: :math:`f`
    and :math:`g_i` are proper, convex and lower semicontinuous, and :math:`h`
    is convex and differentiable with :math:`\\eta^{-1}`-Lipschitz continuous
    gradient, :math:`\\eta > 0`.

    The optional operators :math:`\\nabla l_i^*` need to be
    :math:`\\nu_i`-Lipschitz continuous. Note that in the reference
    [BC2015]_, the condition is formulated as :math:`l_i` being proper, lower
    semicontinuous, and :math:`\\nu_i^{-1}`-strongly convex, which implies that
    :math:`l_i^*` have :math:`\\nu_i`-Lipschitz continuous gradients.

    If the optional operators :math:`\\nabla l_i^*` are omitted, the simpler
    problem without :math:`l_i` will be considered. Mathematically, this is
    done by taking :math:`l_i` to be the functionals that are zero only in the
    zero element and :math:`\\infty` otherwise. This gives that :math:`l_i^*`
    are the zero functionals, and hence the corresponding gradients are the
    zero operators.

    To guarantee convergence, the parameters :math:`\\tau`, :math:`\\sigma` and
    :math:`L_i` need to satisfy

    .. math::

       2 \min \{ \\frac{1}{\\tau}, \\frac{1}{\sigma_1}, \\ldots,
       \\frac{1}{\sigma_m} \} \cdot \min\{ \\eta, \\nu_1, \\ldots, \\nu_m  \}
       \cdot \\sqrt{1 - \\tau \\sum_{i=1}^n \\sigma_i ||L_i||^2} > 1,

    where, if the simpler problem is considered, all :math:`\\nu_i` can be
    considered to be :math:`\\infty`.

    See Also
    --------
    odl.solvers.nonsmooth.chambolle_pock.chambolle_pock_solver :
        Solver for similar problems without differentiability in any
        of the terms.
    odl.solvers.nonsmooth.douglas_rachford.douglas_rachford_pd :
        Solver for similar problems without differentiability in any
        of the terms.

    References
    ----------
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
        raise ValueError('`x` not in the domain of all operators in `L`')
    if len(sigma) != m:
        raise ValueError('len(sigma) != len(L)')
    if len(g) != m:
        raise ValueError('len(prox_cc_g) != len(L)')

    # Extract operators
    prox_cc_g = [gi.convex_conj.proximal for gi in g]
    grad_h = h.gradient
    prox_f = f.proximal

    l = kwargs.pop('l', None)
    if l is not None:
        if len(l) != m:
            raise ValueError('`grad_cc_l` not same length as `L`')
        grad_cc_l = [li.convex_conj.gradient for li in l]

    if kwargs:
        raise TypeError('unexpected keyword argument: {}'.format(kwargs))

    # Pre-allocate values
    v = [Li.range.zero() for Li in L]
    y = x.space.zero()

    for k in range(niter):
        x_old = x

        tmp_1 = grad_h(x) + sum(Li.adjoint(vi) for Li, vi in zip(L, v))
        prox_f(tau)(x - tau * tmp_1, out=x)
        y.lincomb(2.0, x, -1, x_old)

        for i in range(m):
            if l is not None:
                # In this case gradients were given.
                tmp_2 = sigma[i] * (L[i](y) - grad_cc_l[i](v[i]))
            else:
                # In this case gradients were not given. Therefore the gradient
                # step is omitted. For more details, see the documentation.
                tmp_2 = sigma[i] * L[i](y)

            prox_cc_g[i](sigma[i])(v[i] + tmp_2, out=v[i])

        if callback is not None:
            callback(x)
