# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""First-order primal-dual algorithm developed by Chambolle and Pock.

The Chambolle-Pock algorithm is a flexible method well suited for
non-smooth convex optimization problems in imaging.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.operator import Operator


__all__ = ('chambolle_pock_solver',)


# TODO: add dual gap as convergence measure
# TODO: diagonal preconditioning

def chambolle_pock_solver(x, f, g, L, tau, sigma, niter, **kwargs):
    """Chambolle-Pock algorithm for non-smooth convex optimization problems.

    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is::

        min_{x in X} f(L x) + g(x)

    where ``L`` is an operator and ``F`` and ``G`` are functionals.

    The Chambolle-Pock algorithm is a primal-dual algorithm, and basically
    consists of alternating a gradient ascent in the dual variable and a
    gradient descent in the primal variable. The proximal operator is used to
    generate a ascent direction for the convex conjugate of F and descent
    direction for G. Additionally an over-relaxation in the primal variable is
    performed.

    Parameters
    ----------
    x : ``L.domain`` element
        Starting point of the iteration, updated in-place.
    f : `Functional`
        The function ``f`` in the problem definition. Needs to have
        ``f.convex_conj.proximal``.
    g : `Functional`
        The function ``g`` in the problem definition. Needs to have
        ``g.proximal``.
    L : linear `Operator`
        The linear operator that should be applied before ``f``. Its range must
        match the domain of ``f`` and its domain must match the domain of
        ``g``.
    tau : positive float
        Step size parameter for the update of the primal (``g``) variable.
    sigma : positive float
        Step size parameter for the update of the dual (``f``) variable.
    niter : non-negative int
        Number of iterations.

    Other Parameters
    ----------------
    callback : callable, optional
        Function called with the current iterate after each iteration.
    theta : float, optional
        Relaxation parameter, required to fulfill ``0 <= theta <= 1``.
        Default: 1
    gamma : non-negative float, optional
        Acceleration parameter. If not ``None``, it overrides ``theta`` and
        causes variable relaxation parameter and step sizes to be used,
        with ``tau`` and ``sigma`` as initial values. Requires ``G`` or
        ``F^*`` to be uniformly convex.
        Default: ``None``
    x_relax : ``op.domain`` element, optional
        Required to resume iteration. For ``None``, a copy of the primal
        variable ``x`` is used.
        Default: ``None``
    y : ``op.range`` element, optional
        Required to resume iteration. For ``None``, ``op.range.zero()``
        is used.
        Default: ``None``

    Notes
    -----
    The problem of interest is

    .. math::
        \\min_{x \\in X} F(K x) + G(x),

    where the formal conditions are that :math:`K` is an operator
    between Hilbert spaces :math:`X` and :math:`Y`.
    Further, :math:`G : X \\rightarrow [0, +\\infty]` and
    :math:`F : Y \\rightarrow [0, +\\infty]` are proper, convex,
    lower-semicontinuous functionals.

    Convergence is only guaranteed if :math:`K` is linear, :math:`X, Y`
    are finite dimensional and the step lengths :math:`\\sigma` and
    :math:`\\tau` satisfy

    .. math::
       \\tau \\sigma \|K\| < 1

    where :math:`\|K\|` is the operator norm of :math:`K`.

    It is often of interest to study problems that involve several operators,
    for example the classical TV regularized problem

    .. math::
        \\min_x \|Ax - b\|_2^2 + \|\\nabla x\|_1.

    Here it is tempting to let :math:`K=A`, :math:`F(y)=||y||_2^2` and
    :math:`G(x)=\|\\nabla x\|_1`. This is however not feasible since the
    proximal of :math:`||\\nabla x||_1` has no closed form expression.

    Instead, the problem can be formulated :math:`K(x) = (A(x), \\nabla x)`,
    :math:`F((x_1, x_2)) = \|x_1\|_2^2 + \|x_2\|_1`, :math:`G(x)=0`. See the
    examples folder for more information on how to do this.

    For a more detailed documentation see :ref:`chambolle_pock`.

    References on the algorithm can be found in [CP2011a] and [CP2011b].

    This implementation of the CP algorithm is along the lines of
    [Sid+2012].

    The non-linear case is analyzed in [Val2014].

    See Also
    --------
    odl.solvers.nonsmooth.douglas_rachford.douglas_rachford_pd :
        Solver for similar problems which can additionaly handle infimal
        convolutions and multiple forward operators.
    odl.solvers.nonsmooth.forward_backward.forward_backward_pd :
        Solver for similar problems which can additionaly handle infimal
        convolutions, multiple forward operators and a differentiable term.

    References
    ----------
    [CP2011a] Chambolle, A and Pock, T. *A First-Order
    Primal-Dual Algorithm for Convex Problems with Applications to
    Imaging*. Journal of Mathematical Imaging and Vision, 40 (2011),
    pp 120-145.

    [CP2011b] Chambolle, A and Pock, T. *Diagonal
    preconditioning for first order primal-dual algorithms in convex
    optimization*. 2011 IEEE International Conference on Computer Vision
    (ICCV), 2011, pp 1762-1769.

    [Sid+2012] Sidky, E Y, Jorgensen, J H, and Pan, X.
    *Convex optimization problem prototyping for image reconstruction in
    computed tomography with the Chambolle-Pock algorithm*. Physics in
    Medicine and Biology, 57 (2012), pp 3065-3091.

    [Val2014] Valkonen, T.
    *A primal-dual hybrid gradient method for non-linear operators with
    applications to MRI*. Inverse Problems, 30 (2014).
    """
    # Forward operator
    if not isinstance(L, Operator):
        raise TypeError('`op` {!r} is not an `Operator` instance'
                        ''.format(L))

    # Starting point
    if x not in L.domain:
        raise TypeError('`x` {!r} is not in the domain of `op` {!r}'
                        ''.format(x, L.domain))

    # Step size parameter
    tau, tau_in = float(tau), tau
    if tau <= 0:
        raise ValueError('`tau` must be positive, got {}'.format(tau_in))

    # Step size parameter
    sigma, sigma_in = float(sigma), sigma
    if sigma <= 0:
        raise ValueError('`sigma` must be positive, got {}'.format(sigma_in))

    # Number of iterations
    if not isinstance(niter, int) or niter < 0:
        raise ValueError('`niter` {} not understood'
                         ''.format(niter))

    # Relaxation parameter
    theta = kwargs.pop('theta', 1)
    theta, theta_in = float(theta), theta
    if not 0 <= theta <= 1:
        raise ValueError('`theta` {} not in [0, 1]'
                         ''.format(theta_in))

    # Acceleration parameter
    gamma = kwargs.pop('gamma', None)
    if gamma is not None:
        gamma, gamma_in = float(gamma), gamma
        if gamma < 0:
            raise ValueError('`gamma` must be non-negative, got {}'
                             ''.format(gamma_in))

    # Callback object
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'
                        ''.format(callback))

    # Initialize the relaxation variable
    x_relax = kwargs.pop('x_relax', None)
    if x_relax is None:
        x_relax = x.copy()
    elif x_relax not in L.domain:
        raise TypeError('`x_relax` {} is not in the domain of '
                        '`L` {}'.format(x_relax.space, L.domain))

    # Initialize the dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = L.range.zero()
    elif y not in L.range:
        raise TypeError('`y` {} is not in the range of `L` '
                        '{}'.format(y.space, L.range))

    # Get the proximals
    proximal_dual = f.convex_conj.proximal
    proximal_primal = g.proximal
    proximal_constant = (gamma is None)
    if proximal_constant:
        # Pre-compute proximals for efficiency
        proximal_dual_sigma = proximal_dual(sigma)
        proximal_primal_tau = proximal_primal(tau)

    # Temporary copy to store previous iterate
    x_old = x.space.element()

    # Temporaries
    dual_tmp = L.range.element()
    primal_tmp = L.domain.element()

    for _ in range(niter):
        # Copy required for relaxation
        x_old.assign(x)

        # Gradient ascent in the dual variable y
        # Compute dual_tmp = y + sigma * L(x_relax)
        L(x_relax, out=dual_tmp)
        dual_tmp.lincomb(1, y, sigma, dual_tmp)

        # Apply the dual proximal
        if not proximal_constant:
            proximal_dual_sigma = proximal_dual(sigma)
        proximal_dual_sigma(dual_tmp, out=y)

        # Gradient descent in the primal variable x
        # Compute primal_tmp = x + (- tau) * L.derivative(x).adjoint(y)
        L.derivative(x).adjoint(y, out=primal_tmp)
        primal_tmp.lincomb(1, x, -tau, primal_tmp)

        # Apply the primal proximal
        if not proximal_constant:
            proximal_primal_tau = proximal_primal(tau)
        proximal_primal_tau(primal_tmp, out=x)

        # Acceleration
        if gamma is not None:
            theta = float(1 / np.sqrt(1 + 2 * gamma * tau))
            tau *= theta
            sigma /= theta

        # Over-relaxation in the primal variable x
        x_relax.lincomb(1 + theta, x, -theta, x_old)

        if callback is not None:
            callback(x)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
