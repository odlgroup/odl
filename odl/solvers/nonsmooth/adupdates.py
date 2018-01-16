# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Alternating dual (AD) update algorithm studied by McGaffin and Fessler.

The alternating dual upgrade method is reported to be a flexible method which
allows an acceleration by permitting efficient calculations on a GPU.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

__all__ = ('adupdates',)


def adupdates(funcs, ops, x, stepsize, majs, niter, random=False,
              callback=None, callback_loop='outer'):
    """Alternating Dual updates method.

    The Alternating Dual (AD) updates method of McGaffin and Fessler `[MF2015]
    <http://ieeexplore.ieee.org/document/7271047/>`_ is designed to solve an
    optimization problem of the form

        min_x sum_i f_i(L_i x)

    where ``f_i`` are proper, convex and lower semicontinuous functions and
    ``L_i`` are linear `Operator`s.

    Parameters
    ----------
    funcs : sequence of `Functional`s
        All functions need to provide a `convex_conj` with a
        `proximal` factory.
    ops : sequence of `Operator`s
        Needs to provide as many elements as `funcs`.
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    stepsize : positive float
        The stepsize for the outer (proximal point) iteration. The theory
        guarantees convergence for any positive real number, but the
        performance might depend on the choice of a good stepsize.
    majs : sequence of majorizers
        Parameters determining the stepsizes for the inner iterations. Must be
        matched with the norms of `ops`, and convergence is guaranteed if
        `majs` are large enough. See the Notes section for details.
    niter : int
        Number of (outer) iterations.
    random : bool, optional
        If `True`, the order of the dual upgdates is chosen randomly,
        otherwise the order provided by the lists `funcs`, `ops` and `majs`
        is used.
    callback : callable, optional
        Function called with the current iterate after each iteration.
    callback_loop : {'inner', 'outer'}, optional
       If 'inner', the `callback` function is called after each inner
       iteration, i.e., after each dual update. If 'outer', the `callback`
       function is called after each outer iteration, i.e., after each primal
       update.

    Notes
    -----
    The algorithm as implemented here is described in the article [MF2015],
    where it is applied to a tomography problem. It solves the problem

    .. math::
        \min_x \sum_{i=1}^m f_i(L_i x),

    where :math:`f_i` are proper, convex and lower semicontinuous functions
    and :math:`L_i` are linear, continuous operators for
    :math:`i = 1, \ldots, m`. In an outer iteration, the solution is found
    iteratively by an iteration

    .. math::
        x_{n+1} = \mathrm{arg\,min}_x \sum_{i=1}^m f_i(L_i x)
            + \frac{\mu}{2} \|x - x_n\|^2

    according to the proximal point algorithm. In the inner iteration, dual
    variables are introduces for each of the components of the sum. The
    Lagrangian of the problem is given by

    .. math::
        S_n(x; v_1, \ldots, v_m) = \sum_{i=1}^m (\langle v_i, L_i x \rangle
            - f_i^*(v_i)) + \frac{\mu}{2} \|x - x_n\|^2.

    Given the dual variables, the new primal variable :math:`\tilde x_{n+1}`
    can be calculated by directly minimizing :math:`S_n` with respect to
    :math:`x`. This corresponds to the formula

    .. math::
         \tilde x_{n+1} = x_n - \frac{1}{\mu} \sum_{i=1}^m L_i^* v_i.

    The dual updates are executed according to the following rule:

    .. math::
        v_j^+ = \mathrm{Prox}_{\mu M_j^{-1}}
            (v_j + \mu M_j^{-1} L_j \tilde x_{n+1}),

    where :math:`M_j` is a diagonal matrix with positive diagonal entries such
    that :math:`M_j - L_j L_j^*` is positive semidefinite. The matrix
    :math:`M_j` is incorporated into this function via the variable `majs` in
    one of the following ways:

    * Setting `majs[i]` a positive float :math:`\gamma` corresponds to the
    choice :math:`M_i = \gamma \mathrm{Id}`.

    * Assume that `f_j` is a `SeparableSum`, then setting `majs[i]` a list
    :math:`(\gamma_1, \ldots, \gamma_k)`of positive floats corresponds to the
    choice of a block-diagonal matrix :math:`M_j`, where each block corresponds
    to one of the space components and equals :math:`\gamma_i \mathrm{Id}`.

    * Assume that `f_j` is a `L1_norm` or a `L2_norm_squared`, then setting
    `majs[i]` a `f_j.domain.element` :math:`z` corresponds to the choice
    :math:`\mathrm{diag}(z)`.

    References
    ----------
    [MF2015] McGaffin, M G, and Fessler, J A. *Alternating dual updates
    algorithm for X-ray CT reconstruction on the GPU*. IEEE Transactions
    on Computational Imaging, 1.3 (2015), pp 186--199.
    """
    # Check the lenghts of the lists (= number of dual variables)
    length = len(funcs)
    if not len(funcs) == len(ops):
        raise ValueError('Number of `ops` {} and number '
                         'of `funcs` {} do not agree.'
                         ''.format(len(ops), len(funcs)))

    # Check if operators have a common domain
    # (the space of the primal variable):
    domain = ops[0].domain
    if any(domain != opi.domain for opi in ops):
        raise ValueError('Domains of `ops` are not all equal.')

    # Check if range of the operators equals domain of the functionals
    ranges = [opi.range for opi in ops]
    if any(ops[i].range != funcs[i].domain for i in range(length)):
        raise ValueError('Ranges of operators and domains'
                         'of functionals do not agree.')

    # Initialization of the dual variables
    duals = [space.zero() for space in ranges]

    # Reusable elements in the ranges, one per type of space
    unique_ranges = set(ranges)
    tmp_rans = {ran: ran.element() for ran in unique_ranges}

    # Prepare the proximal operators. Since the stepsize does not vary over
    # the iterations, we always use the same proximal operator.
    proxs = [func.convex_conj.proximal(stepsize / maj)
             for (func, maj) in zip(funcs, majs)]

    # Iteratively find a solution
    for _ in range(niter):
        # Update x = x - 1/stepsize * sum([ops[i].adjoint(duals[i])
        # for i in range(length)])
        for i in range(length):
            x -= (1.0 / stepsize) * ops[i].adjoint(duals[i])

        if random:
            rng = np.random.permutation(range(len(ops)))
        else:
            rng = range(len(ops))

        for j in rng:
            step = stepsize / majs[j]
            arg = duals[j] + step * ops[j](x)
            tmp_ran = tmp_rans[ops[j].range]
            proxs[j](arg, out=tmp_ran)
            x -= 1.0 / stepsize * ops[j].adjoint(tmp_ran - duals[j])
            duals[j].assign(tmp_ran)

            if callback is not None and callback_loop == 'inner':
                callback(x)
        if callback is not None and callback_loop == 'outer':
            callback(x)
