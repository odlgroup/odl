# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Solvers for the optimization of the difference of convex functions.

Collection of DCA (d.c. algorithms) and related methods which make use of
structured optimization if the objective function can be written as a
difference of two convex functions.
"""

from __future__ import print_function, division, absolute_import

__all__ = ('dca', 'prox_dca', 'doubleprox_dc')


def dca(x, f, g, niter, callback=None):
    r"""Subgradient DCA of Tao and An.

    This algorithm solves a problem of the form ::

        min_x f(x) - g(x),

    where ``f`` and ``g`` are proper, convex and lower semicontinuous
    functions.

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    f : `Functional`
        Convex functional. Needs to implement ``f.convex_conj.gradient``.
    g : `Functional`
        Convex functional. Needs to implement ``g.gradient``.
    niter : int
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    The algorithm is described in Section 3 and in particular in Theorem 3 of
    `[TA1997] <http://journals.math.ac.vn/acta/pdf/9701289.pdf>`_. The problem

    .. math::
        \min f(x) - g(x)

    has the first-order optimality condition :math:`0 \in \partial f(x) -
    \partial g(x)`, i.e., aims at finding an :math:`x` so that there exists a
    common element

    .. math::
        y \in \partial f(x) \cap \partial g(x).

    The element :math:`y` can be seen as a solution of the Toland dual problem

    .. math::
        \min g^*(y) - f^*(y)

    and the iteration is given by

    .. math::
        y_n \in \partial g(x_n), \qquad x_{n+1} \in \partial f^*(y_n),

    for :math:`n\geq 0`. Here, a subgradient is found by evaluating the
    gradient method of the respective functionals.

    References
    ----------
    [TA1997] Tao, P D, and An, L T H. *Convex analysis approach to d.c.
    programming: Theory, algorithms and applications*. Acta Mathematica
    Vietnamica, 22.1 (1997), pp 289--355.

    See also
    --------
    prox_dca :
        Solver with a proximal step for ``f`` and a subgradient step for ``g``.
    doubleprox_dc :
        Solver with proximal steps for all the nonsmooth convex functionals
        and a gradient step for a smooth functional.
    """
    space = f.domain
    if g.domain != space:
        raise ValueError('`f.domain` and `g.domain` need to be equal, but '
                         '{} != {}'.format(space, g.domain))
    f_convex_conj = f.convex_conj
    for _ in range(niter):
        f_convex_conj.gradient(g.gradient(x), out=x)

        if callback is not None:
            callback(x)


def prox_dca(x, f, g, niter, gamma, callback=None):
    r"""Proximal DCA of Sun, Sampaio and Candido.

    This algorithm solves a problem of the form ::

        min_x f(x) - g(x)

    where ``f`` and ``g`` are two proper, convex and lower semicontinuous
    functions.

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial point, updated in-place.
    f : `Functional`
        Convex functional. Needs to implement ``f.proximal``.
    g : `Functional`
        Convex functional. Needs to implement ``g.gradient``.
    niter : int
        Number of iterations.
    gamma : positive float
        Stepsize in the primal updates.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    The algorithm was proposed as Algorithm 2.3 in
    `[SSC2003]
    <http://www.global-sci.org/jcm/readabs.php?vol=21&no=4&page=451&year=2003&ppage=462>`_.
    It solves the problem

    .. math ::
        \min f(x) - g(x)

    by using subgradients of :math:`g` and proximal points of :math:`f`.
    The iteration is given by

    .. math ::
        y_n \in \partial g(x_n), \qquad x_{n+1}
            = \mathrm{Prox}_{\gamma f}(x_n + \gamma y_n).

    In contrast to `dca`, `prox_dca` uses proximal steps with respect to the
    convex part ``f``. Both algorithms use subgradients of the concave part
    ``g``.

    References
    ----------
    [SSC2003] Sun, W, Sampaio R J B, and Candido M A B. *Proximal point
    algorithm for minimization of DC function*. Journal of Computational
    Mathematics, 21.4 (2003), pp 451--462.

    See also
    --------
    dca :
        Solver with subgradinet steps for all the functionals.
    doubleprox_dc :
        Solver with proximal steps for all the nonsmooth convex functionals
        and a gradient step for a smooth functional.
    """
    space = f.domain
    if g.domain != space:
        raise ValueError('`f.domain` and `g.domain` need to be equal, but '
                         '{} != {}'.format(space, g.domain))
    for _ in range(niter):
        f.proximal(gamma)(x.lincomb(1, x, gamma, g.gradient(x)), out=x)

        if callback is not None:
            callback(x)


def doubleprox_dc(x, y, f, phi, g, K, niter, gamma, mu, callback=None):
    r"""Double-proxmial gradient d.c. algorithm of Banert and Bot.

    This algorithm solves a problem of the form ::

        min_x f(x) + phi(x) - g(Kx).

    Parameters
    ----------
    x : `LinearSpaceElement`
        Initial primal guess, updated in-place.
    y : `LinearSpaceElement`
        Initial dual guess, updated in-place.
    f : `Functional`
        Convex functional. Needs to implement ``g.proximal``.
    phi : `Functional`
        Convex functional. Needs to implement ``phi.gradient``.
        Convergence can be guaranteed if the gradient is Lipschitz continuous.
    g : `Functional`
        Convex functional. Needs to implement ``h.convex_conj.proximal``.
    K : `Operator`
        Linear operator. Needs to implement ``K.adjoint``
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
    This algorithm is proposed in `[BB2016]
    <https://arxiv.org/abs/1610.06538>`_ and solves the d.c. problem

    .. math ::
        \min_x f(x) + \varphi(x) - g(Kx)

    together with its Toland dual

    .. math ::
        \min_y g^*(y) - (f + \varphi)^*(K^* y).

    The iterations are given by

    .. math ::
        x_{n+1} &= \mathrm{Prox}_{\gamma f} (x_n + \gamma (K^* y_n
                   - \nabla \varphi(x_n))), \\
        y_{n+1} &= \mathrm{Prox}_{\mu g^*} (y_n + \mu K x_{n+1}).

    To guarantee convergence, the parameter :math:`\gamma` must satisfy
    :math:`0 < \gamma < 2/L` where :math:`L` is the Lipschitz constant of
    :math:`\nabla \varphi`.

    References
    ----------
    [BB2016] Banert, S, and Bot, R I. *A general double-proximal gradient
    algorithm for d.c. programming*. arXiv:1610.06538 [math.OC] (2016).

    See also
    --------
    dca :
        Solver with subgradient steps for all the functionals.
    prox_dca :
        Solver with a proximal step for ``f`` and a subgradient step for ``g``.
    """
    primal_space = f.domain
    dual_space = g.domain

    if phi.domain != primal_space:
        raise ValueError('`f.domain` and `phi.domain` need to be equal, but '
                         '{} != {}'.format(primal_space, phi.domain))
    if K.domain != primal_space:
        raise ValueError('`f.domain` and `K.domain` need to be equal, but '
                         '{} != {}'.format(primal_space, K.domain))
    if K.range != dual_space:
        raise ValueError('`g.domain` and `K.range` need to be equal, but '
                         '{} != {}'.format(dual_space, K.range))

    g_convex_conj = g.convex_conj
    for _ in range(niter):
        f.proximal(gamma)(x.lincomb(1, x,
                                    gamma, K.adjoint(y) - phi.gradient(x)),
                          out=x)
        g_convex_conj.proximal(mu)(y.lincomb(1, y, mu, K(x)), out=y)

        if callback is not None:
            callback(x)


def doubleprox_dc_simple(x, y, f, phi, g, K, niter, gamma, mu):
    """Non-optimized version of ``doubleprox_dc``.
    This function is intended for debugging. It makes a lot of copies and
    performs no error checking.
    """
    for _ in range(niter):
        f.proximal(gamma)(x + gamma * K.adjoint(y) -
                          gamma * phi.gradient(x), out=x)
        g.convex_conj.proximal(mu)(y + mu * K(x), out=y)
