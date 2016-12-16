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

"""(Fast) Iterative shrinkage-thresholding algorithm."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()


import numpy as np


__all__ = ('proximal_gradient', 'accelerated_proximal_gradient')


def proximal_gradient(x, f, g, gamma, niter, callback=None, **kwargs):
    """(Accelerated) proximal gradient algorithm for convex optimization.

    Also known as "Iterative Soft-Thresholding Algorithm" (ISTA).
    See Beck2009_ for more information.

    Solves the convex optimization problem::

        min_{x in X} f(x) + g(x)

    where the proximal operator of ``f`` is known and ``g`` is differentiable.

    Parameters
    ----------
    x : ``f.domain`` element
        Starting point of the iteration, updated in-place.
    f : `Functional`
        The function ``f`` in the problem definition. Needs to have
        ``f.proximal``.
    g : `Functional`
        The function ``g`` in the problem definition. Needs to have
        ``g.gradient``.
    gamma : positive float
        Step size parameter.
    niter : non-negative int, optional
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Other Parameters
    ----------------
    lam : float or callable, optional
        Overrelaxation step size. If callable, should take an index (zero
        indexed) and return the corresponding step size.
        Default: 1.0

    Notes
    -----
    The problem of interest is

    .. math::

        \\min_{x \\in X} f(x) + g(x),

    where the technical conditions are that
    :math:`f : X \\rightarrow \mathbb{R}` is proper, convex and
    lower-semicontinuous, and :math:`g : X \\rightarrow \mathbb{R}` is
    differentiable and :math:`\\nabla g` is :math:`1 / \\beta`-Lipschitz
    continuous.

    Convergence is only guaranteed if the step length :math:`\\gamma` satisfies

    .. math::

       0 < \\gamma < 2 \\beta

    and the parameter :math:`\\lambda` (``lam``) satisfies

    .. math::

       \\sum_{k=0}^\\infty \\lambda_k (\\delta - \\lambda_k) = + \\infty

    where :math:`\\delta = \\min \{1, \\beta / \\gamma\}`.

    References
    ----------
    .. _Beck2009: http://epubs.siam.org/doi/abs/10.1137/080716542
    """
    # Starting point
    if x not in f.domain:
        raise TypeError('`x` {!r} is not in the domain of `f` {!r}'
                        ''.format(x, f.domain))
    if x not in g.domain:
        raise TypeError('`x` {!r} is not in the domain of `g` {!r}'
                        ''.format(x, g.domain))

    # Step size parameter
    gamma, gamma_in = float(gamma), gamma
    if gamma <= 0:
        raise ValueError('`gamma` must be positive, got {}'.format(gamma_in))

    # Number of iterations
    if not isinstance(niter, int) or niter < 0:
        raise ValueError('`niter` {} not understood'
                         ''.format(niter))

    lam_in = kwargs.pop('lam', 1.0)
    lam = lam_in if callable(lam_in) else lambda _: float(lam_in)

    # Get the proximals
    f_prox = f.proximal(gamma)

    # Create temporary
    tmp = x.space.element()

    for k in range(niter):
        lam_k = lam(k)

        # x - tau grad_g (x)
        tmp.lincomb(1, x, -gamma, g.gradient(x))

        # Update x
        x.lincomb(1 - lam_k, x, lam_k, f_prox(tmp))

        if callback is not None:
            callback(x)


def accelerated_proximal_gradient(x, f, g, gamma, niter, callback=None,
                                  **kwargs):
    """Accelerated proximal gradient algorithm for convex optimization.

    The method is known as "Fast Iterative Soft-Thresholding Algorithm"
    (FISTA). See Beck2009_ for more information.

    Solves the convex optimization problem::

        min_{x in X} f(x) + g(x)

    where the proximal operator of ``f`` is known and ``g`` is differentiable.

    Parameters
    ----------
    x : ``f.domain`` element
        Starting point of the iteration, updated in-place.
    f : `Functional`
        The function ``f`` in the problem definition. Needs to have
        ``f.proximal``.
    g : `Functional`
        The function ``g`` in the problem definition. Needs to have
        ``g.gradient``.
    gamma : positive float
        Step size parameter.
    niter : non-negative int, optional
        Number of iterations.
    callback : callable, optional
        Function called with the current iterate after each iteration.

    Notes
    -----
    The problem of interest is

    .. math::

        \\min_{x \\in X} f(x) + g(x),

    where the technical conditions are that
    :math:`f : X \\rightarrow \mathbb{R}` is proper, convex and
    lower-semicontinuous, and :math:`g : X \\rightarrow \mathbb{R}` is
    differentiable and :math:`\\nabla g` is :math:`1 / \\beta`-Lipschitz
    continuous.

    Convergence is only guaranteed if the step length :math:`\\gamma` satisfies

    .. math::

       0 < \\gamma < 2 \\beta.

    References
    ----------
    .. _Beck2009: http://epubs.siam.org/doi/abs/10.1137/080716542
    """
    # Starting point
    if x not in f.domain:
        raise TypeError('`x` {!r} is not in the domain of `f` {!r}'
                        ''.format(x, f.domain))
    if x not in g.domain:
        raise TypeError('`x` {!r} is not in the domain of `g` {!r}'
                        ''.format(x, g.domain))

    # Step size parameter
    gamma, gamma_in = float(gamma), gamma
    if gamma <= 0:
        raise ValueError('`gamma` must be positive, got {}'.format(gamma_in))

    # Number of iterations
    if not isinstance(niter, int) or niter < 0:
        raise ValueError('`niter` {} not understood'
                         ''.format(niter))

    # Get the proximals
    f_prox = f.proximal(gamma)

    # Create temporary
    tmp = x.space.element()
    y = x.copy()
    t = 1

    for k in range(niter):
        # Update t
        t, t_old = (1 + np.sqrt(1 + 4 * t ** 2)) / 2, t
        alpha = (t_old - 1) / t

        # x - tau grad_g (x)
        tmp.lincomb(1, y, -gamma, g.gradient(y))

        # Store old x value in y
        y.assign(x)

        # Update x
        f_prox(tmp, out=x)

        # Update y
        y.lincomb(1 + alpha, x, -alpha, y)

        if callback is not None:
            callback(x)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
