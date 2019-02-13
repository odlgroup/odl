# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Alternating dual (AD) update algorithm studied by McGaffin and Fessler.

The alternating dual upgrade method solves structured convex optimization
problems by successively updating dual variables which are associated with
each of the components.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

__all__ = ('adupdates',)


def adupdates(x, g, L, stepsize, inner_stepsizes, niter, random=False,
              callback=None, callback_loop='outer'):
    r"""Alternating Dual updates method.

    The Alternating Dual (AD) updates method of McGaffin and Fessler `[MF2015]
    <http://ieeexplore.ieee.org/document/7271047/>`_ is designed to solve an
    optimization problem of the form ::

        min_x [ sum_i g_i(L_i x) ]

    where ``g_i`` are proper, convex and lower semicontinuous functions and
    ``L_i`` are linear `Operator` s.

    Parameters
    ----------
    g : sequence of `Functional` s
        All functions need to provide a `Functional.convex_conj` with a
        `Functional.proximal` factory.
    L : sequence of `Operator` s
        Length of ``L`` must equal the length of ``g``.
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    stepsize : positive float
        The stepsize for the outer (proximal point) iteration. The theory
        guarantees convergence for any positive real number, but the
        performance might depend on the choice of a good stepsize.
    inner_stepsizes : sequence of stepsizes
        Parameters determining the stepsizes for the inner iterations. Must be
        matched with the norms of ``L``, and convergence is guaranteed if the
        ``inner_stepsizes`` are small enough. See the Notes section for
        details.
    niter : int
        Number of (outer) iterations.
    random : bool, optional
        If `True`, the order of the dual upgdates is chosen randomly,
        otherwise the order provided by the lists ``g``, ``L`` and
        ``inner_stepsizes`` is used.
    callback : callable, optional
        Function called with the current iterate after each iteration.
    callback_loop : {'inner', 'outer'}, optional
       If 'inner', the ``callback`` function is called after each inner
       iteration, i.e., after each dual update. If 'outer', the ``callback``
       function is called after each outer iteration, i.e., after each primal
       update.

    Notes
    -----
    The algorithm as implemented here is described in the article [MF2015],
    where it is applied to a tomography problem. It solves the problem

    .. math::
        \min_x \sum_{i=1}^m g_i(L_i x),

    where :math:`g_i` are proper, convex and lower semicontinuous functions
    and :math:`L_i` are linear, continuous operators for
    :math:`i = 1, \ldots, m`. In an outer iteration, the solution is found
    iteratively by an iteration

    .. math::
        x_{n+1} = \mathrm{arg\,min}_x \sum_{i=1}^m g_i(L_i x)
            + \frac{\mu}{2} \|x - x_n\|^2

    with some ``stepsize`` parameter :math:`\mu > 0` according to the proximal
    point algorithm. In the inner iteration, dual variables are introduced for
    each of the components of the sum. The Lagrangian of the problem is given
    by

    .. math::
        S_n(x; v_1, \ldots, v_m) = \sum_{i=1}^m (\langle v_i, L_i x \rangle
            - g_i^*(v_i)) + \frac{\mu}{2} \|x - x_n\|^2.

    Given the dual variables, the new primal variable :math:`x_{n+1}`
    can be calculated by directly minimizing :math:`S_n` with respect to
    :math:`x`. This corresponds to the formula

    .. math::
         x_{n+1} = x_n - \frac{1}{\mu} \sum_{i=1}^m L_i^* v_i.

    The dual updates are executed according to the following rule:

    .. math::
        v_i^+ = \mathrm{Prox}^{\mu M_i^{-1}}_{g_i^*}
            (v_i + \mu M_i^{-1} L_i x_{n+1}),

    where :math:`x_{n+1}` is given by the formula above and  :math:`M_i` is a
    diagonal matrix with positive diagonal entries such that
    :math:`M_i - L_i L_i^*` is positive semidefinite. The variable
    ``inner_stepsizes`` is chosen as a stepsize to the `Functional.proximal` to
    the `Functional.convex_conj` of each of the ``g`` s after multiplying with
    ``stepsize``. The ``inner_stepsizes`` contain the elements of
    :math:`M_i^{-1}` in one of the following ways:

    * Setting ``inner_stepsizes[i]`` a positive float :math:`\gamma`
      corresponds to the choice :math:`M_i^{-1} = \gamma \mathrm{Id}`.
    * Assume that ``g_i`` is a `SeparableSum`, then setting
      ``inner_stepsizes[i]`` a list :math:`(\gamma_1, \ldots, \gamma_k)` of
      positive floats corresponds to the choice of a block-diagonal matrix
      :math:`M_i^{-1}`, where each block corresponds to one of the space
      components and equals :math:`\gamma_i \mathrm{Id}`.
    * Assume that ``g_i`` is an `L1Norm` or an `L2NormSquared`, then setting
      ``inner_stepsizes[i]`` a ``g_i.domain.element`` :math:`z` corresponds to
      the choice :math:`M_i^{-1} = \mathrm{diag}(z)`.

    References
    ----------
    [MF2015] McGaffin, M G, and Fessler, J A. *Alternating dual updates
    algorithm for X-ray CT reconstruction on the GPU*. IEEE Transactions
    on Computational Imaging, 1.3 (2015), pp 186--199.
    """
    # Check the lenghts of the lists (= number of dual variables)
    length = len(g)
    if len(L) != length:
        raise ValueError('`len(L)` should equal `len(g)`, but {} != {}'
                         ''.format(len(L), length))

    if len(inner_stepsizes) != length:
        raise ValueError('len(`inner_stepsizes`) should equal `len(g)`, '
                         ' but {} != {}'.format(len(inner_stepsizes), length))

    # Check if operators have a common domain
    # (the space of the primal variable):
    domain = L[0].domain
    if any(opi.domain != domain for opi in L):
        raise ValueError('domains of `L` are not all equal')

    # Check if range of the operators equals domain of the functionals
    ranges = [opi.range for opi in L]
    if any(L[i].range != g[i].domain for i in range(length)):
        raise ValueError('L[i].range` should equal `g.domain`')

    # Normalize string
    callback_loop, callback_loop_in = str(callback_loop).lower(), callback_loop
    if callback_loop not in ('inner', 'outer'):
        raise ValueError('`callback_loop` {!r} not understood'
                         ''.format(callback_loop_in))

    # Initialization of the dual variables
    duals = [space.zero() for space in ranges]

    # Reusable elements in the ranges, one per type of space
    unique_ranges = set(ranges)
    tmp_rans = {ran: ran.element() for ran in unique_ranges}

    # Prepare the proximal operators. Since the stepsize does not vary over
    # the iterations, we always use the same proximal operator.
    proxs = [func.convex_conj.proximal(stepsize * inner_ss
                                       if np.isscalar(inner_ss)
                                       else stepsize * np.asarray(inner_ss))
             for (func, inner_ss) in zip(g, inner_stepsizes)]

    # Iteratively find a solution
    for _ in range(niter):
        # Update x = x - 1/stepsize * sum([ops[i].adjoint(duals[i])
        # for i in range(length)])
        for i in range(length):
            x -= (1.0 / stepsize) * L[i].adjoint(duals[i])

        if random:
            rng = np.random.permutation(range(length))
        else:
            rng = range(length)

        for j in rng:
            step = (stepsize * inner_stepsizes[j]
                    if np.isscalar(inner_stepsizes[j])
                    else stepsize * np.asarray(inner_stepsizes[j]))
            arg = duals[j] + step * L[j](x)
            tmp_ran = tmp_rans[L[j].range]
            proxs[j](arg, out=tmp_ran)
            x -= 1.0 / stepsize * L[j].adjoint(tmp_ran - duals[j])
            duals[j].assign(tmp_ran)

            if callback is not None and callback_loop == 'inner':
                callback(x)
        if callback is not None and callback_loop == 'outer':
            callback(x)


def adupdates_simple(x, g, L, stepsize, inner_stepsizes, niter,
                     random=False):
    """Non-optimized version of ``adupdates``.
    This function is intended for debugging. It makes a lot of copies and
    performs no error checking.
    """
    # Initializations
    length = len(g)
    ranges = [Li.range for Li in L]
    duals = [space.zero() for space in ranges]

    # Iteratively find a solution
    for _ in range(niter):
        # Update x = x - 1/stepsize * sum([ops[i].adjoint(duals[i])
        # for i in range(length)])
        for i in range(length):
            x -= (1.0 / stepsize) * L[i].adjoint(duals[i])

        rng = np.random.permutation(range(length)) if random else range(length)

        for j in rng:
            dual_tmp = ranges[j].element()
            dual_tmp = (g[j].convex_conj.proximal
                        (stepsize * inner_stepsizes[j]
                         if np.isscalar(inner_stepsizes[j])
                         else stepsize * np.asarray(inner_stepsizes[j]))
                        (duals[j] + stepsize * inner_stepsizes[j] * L[j](x)
                         if np.isscalar(inner_stepsizes[j])
                         else duals[j] + stepsize *
                         np.asarray(inner_stepsizes[j]) * L[j](x)))
            x -= 1.0 / stepsize * L[j].adjoint(dual_tmp - duals[j])
            duals[j].assign(dual_tmp)
