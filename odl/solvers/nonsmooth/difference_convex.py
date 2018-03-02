# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Solvers for the optimization of the difference of convex functions.

Collection of DCA and related methods which make use of structured optimization
if the objective function can be written as a difference of two convex
functions.
"""

from __future__ import print_function, division, absolute_import

__all__ = ('dca', 'prox_dca', 'doubleprox_dc')


def dca(x, g, h, niter, callback=None):
    r"""Subgradient DCA of Tao and An.

    This algorithm solves a problem of the form::

        min_x g(x) - h(x),

    where ``g`` and ``h`` are proper, convex and lower semicontinuous
    functions.

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    g : `Functional`
        Convex part. Needs to provide a `Functional.convex_conj` with a
        `Functional.gradient` method.
    h : `Functional`
        Negative of the concave part. Needs to provide a
        `Functional.gradient` method.
    niter : int
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    The algorithm is described in Section 3 and in particular in Theorem 3 of
    `[TA1997] <http://journals.math.ac.vn/acta/pdf/9701289.pdf>`_. The problem

    .. math::
        \min g(x) - h(x)

    has the first-order optimality condition :math:`0 \in \partial g(x) -
    \partial h(x)`, i.e., aims at finding an :math:`x` so that there exists a
    common element

    .. math::
        y \in \partial g(x) \cap \partial h(x).

    The element :math:`y` can be seen as a solution of the Toland dual problem

    .. math::
        \min h^*(y) - g^*(y)

    and the iteration is given by

    .. math::
        y_n \in \partial h(x_n), \qquad x_{n+1} \in \partial g^*(y_n),

    for :math:`n\geq 0`. Here, a subgradient is found by evaluating the
    gradient method of the respective functionals.

    References
    ----------
    [TA1997] Tao, P D, and An, L T H. *Convex analysis approach to d.c.
    programming: Theory, algorithms and applications*. Acta Mathematica
    Vietnamica, 22.1 (1997), pp 289--355.

    See also
    --------
    """
#    `prox_dca`, `doubleprox_dc`
#    """
    space = g.domain
    if h.domain != space:
        raise ValueError('`g.domain` and `h.domain` need to be equal, but '
                         '{} != {}'.format(space, h.domain))
    for _ in range(niter):
        g.convex_conj.gradient(h.gradient(x), out=x)

        if callback is not None:
            callback(x)


def prox_dca(x, g, h, niter, gamma, callback=None):
    """Proximal DCA of Sun, Sampaio and Candido.

    This algorithm solves a problem of the form ::

        min_x g(x) - h(x)

    where ``g`` and ``h`` are two proper, convex and lower semicontinuous
    functions.

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    g : `Functional`
        Convex part. Needs to provide a `Functional.convex_conj` with a
        `Functional.proximal` factory.
    h : `Functional`
        The negative of the concave part. Needs to provide a
        `Functional.proximal` factory.
    niter : int
        Number of iterations.
    gamma : positive float
        Stepsize of the proximal steps.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    The algorithm was proposed as Algorithm 2.3 in
    `[SSC2003] <http://www.global-sci.org/jcm/readabs.php?vol=21&\
no=4&page=451&year=2003&ppage=462>`_. It solves the problem

    .. math ::
        \\min g(x) - h(x)

    by involving subgradients of :math:`h` and proximal points of :math:`g^*`.
    The iteration is given by

    .. math ::
        y_n \\in \\partial h(x_n), \\qquad x_{n+1}
            = \\mathrm{Prox}^{\\gamma}_g(x_n + \\gamma y_n).

    In contrast to `dca`, `prox_dca` uses proximal steps with respect to the
    convex part ``g``. Both algorithms use subgradients of the concave part
    ``h``.

    References
    ----------
    [SSC2003] Sun, W, Sampaio R J B, and Candido M A B. *Proximal point
    algorithm for minimization of DC function*. Journal of Computational
    Mathematics, 21.4 (2003), pp 451--462.
    """
    space = g.domain
    if h.domain != space:
        raise ValueError('`g.domain` and `h.domain` need to be equal, but '
                         '{} != {}'.format(space, h.domain))
    for _ in range(niter):
        g.proximal(gamma)(x + gamma * h.gradient(x), out=x)

        if callback is not None:
            callback(x)


def doubleprox_dc(x, y, g, h, phi, K, niter, gamma, mu, callback=None):
    r"""Double-proxmial gradient d.c. algorithm of Banert and Bot.

    This algorithm solves a problem of the form::

        min_x g(x) + phi(x) - h(Kx).

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial primal guess, updated in-place.
    y : `LinearSpaceElement`
        Initial dual guess, updated in-place.
    g : `Functional`
        The nonsmooth part of the convex part of the problem. Needs to provide
        a `Functional.proximal` factory.
    h : `Functional`
        The functional involved in the concave part of the problem. Needs to
        provide a `Functional.proximal` factory.
    phi : `Functional`
        The smooth part of the convex part of the problem. Needs to provide a
        `Functional.gradient`, and convergence can be guaranteed if the
        gradient is Lipschitz continuous.
    K : `Operator`
        The operator involved in the concave part of the problem.
    niter : int
        Number of iterations.
    gamma : positive float
        Stepsize in the primal updates.
    mu : positive float
        Stepsize in the dual updates.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    This algorithm is proposed in`[BB2016]
    <https://arxiv.org/abs/1610.06538>`_ and solves the d.c. problem

    .. math ::
        \min_x g(x) + \varphi(x) - h(Kx)

    together with its Toland dual

    .. math ::
        \min_y h^*(y) - (g + \varphi)^*(K^* y).

    The iterations are given by

    .. math ::
        x_{n+1} &= \mathrm{Prox}_{\gamma}^g (x_n + \gamma K^* y_n
                   - \gamma \nabla \varphi(x_n)), \\
        y_{n+1} &= \mathrm{Prox}_{\mu}^{h^*} (y_n + \mu K x_{n+1}).

    To guarantee convergence, the parameter :math:`\gamma` must satisfy
    :math:`0 < \gamma < 2/L` where :math:`L` is the Lipschitz constant of
    :math:`\nabla \varphi`.

    References
    ----------
    [BB2016] Banert, S, and Bot, R I. *A general double-proximal gradient
    algorithm for d.c. programming*. arXiv:1610.06538 [math.OC] (2016).
    """
    primal_space = g.domain
    dual_space = h.domain

    if phi.domain != primal_space:
        raise ValueError('`g.domain` and `phi.domain` need to be equal, but '
                         '{} != {}'.format(primal_space, phi.domain))
    if K.domain != primal_space:
        raise ValueError('`g.domain` and `K.domain` need to be equal, but '
                         '{} != {}'.format(primal_space, K.domain))
    if K.range != dual_space:
        raise ValueError('`h.domain` and `K.range` need to be equal, but '
                         '{} != {}'.format(dual_space, K.range))

    for _ in range(niter):
        g.proximal(gamma)(x + gamma * K.adjoint(y) -
                          gamma * phi.gradient(x), out=x)
        h.convex_conj.proximal(mu)(y + mu * K(x), out=y)

        if callback is not None:
            callback(x)
