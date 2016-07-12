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

"""First-order primal-dual algorithm developed by Chambolle and Pock.

The Chambolle-Pock algorithm is a flexible method well suited for
non-smooth convex optimization problems in imaging. It was first
proposed in [CP2011a]_.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

from odl.operator.operator import Operator


__all__ = ('chambolle_pock_solver',)


# TODO: add dual gap as convergence measure
# TODO: diagonal preconditioning

def chambolle_pock_solver(op, x, tau, sigma, proximal_primal, proximal_dual,
                          niter=1, **kwargs):
    """Chambolle-Pock algorithm for non-smooth convex optimization problems.

    First order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure. The
    primal formulation of the general problem is::

        min_{x in X} F(K x) + G(x)

    where ``K`` is an operator and ``F`` and ``G`` are functionals.

    The Chambolle-Pock algorithm is a primal-dual algorithm, and basically
    consists of alternating a gradient ascent in the dual variable and a
    gradient descent in the primal variable. The proximal operator is used to
    generate a ascent direction for the convex conjugate of F and descent
    direction for G. Additionally an over-relaxation in the primal variable is
    performed.

    Parameters
    ----------
    op : `Operator`
        Forward operator, the operator ``K`` in the problem formulation.
    x : element in the domain of ``op``
        Starting point of the iteration, updated in place.
    tau : positive `float`
        Step size parameter for the update of the primal variable.
        Controls the extent to which ``proximal_primal`` maps points
        towards the minimum of G.
    sigma : positive `float`
        Step size parameter for the update of the dual variable. Controls
        the extent to which ``proximal_dual`` maps points towards the
        minimum of ``F^*``.
    proximal_primal : `callable`
        `proximal factory` for the functional ``G``.
    proximal_dual : `callable`
        `proximal factory` for the functional ``F^*``.
    niter : non-negative `int`, optional
        Number of iterations.

    Other Parameters
    ----------------
    callback : `callable`, optional
        Function called with the current iterate after each iteration.
    theta : `float` in [0, 1], optional
        Relaxation parameter. Default: 1
    gamma : non-negative `float`, optional
        Acceleration parameter. If not `None`, overwrites ``theta`` and uses
        variable relaxation parameter and step sizes with ``tau`` and
        ``sigma`` as initial values. Requires G or F^* to be uniformly
        convex. Default: `None`
    x_relax : element in the domain of ``op``, optional
        Required to resume iteration. If `None` it is a copy of the primal
        variable x. Default: `None`
    y : element in the range of ``op``, optional
        Required to resume iteration. If `None` it is set to a zero element
        in Y which is the range of ``op``. Default: `None`

    Notes
    -----
    The problem of interest is

    .. math::

        \\min_{x \\in X} F(K x) + G(x),

    where the technical conditions are that :math:`K` is an operator
    between Hilbert spaces :math:`X` and :math:`Y`, where convergence is only
    guaranteed if :math:`K` is linear and :math:`X, Y` are finite dimensional.
    Further, :math:`G : X -> [0, +\\infty]` and :math:`F : Y -> [0, +\\infty]`
    are proper, convex, lower-semicontinuous functionals.

    It is often of interest to study problems that involve several operators,
    for example the classical TV regularized problem

    .. math::

        \\min_x ||Ax - b||_2^2 + ||\\nabla x||_1.

    Here it is tempting to let :math:`K=A`, :math:`F(y)=||y||_2^2` and
    :math:`G(x)=||\\nabla x||_1`. This is however not feasible since the
    proximal of :math:`||\\nabla x||_1` has no closed form expression.

    Instead, the problem can be formulated :math:`K(x) = (A(x), \\nabla x)`,
    :math:`F((x_1, x_2)) = ||x_1||_2^2 + ||x_2||_1`, :math:`G(x)=0`. See the
    examples folder for more information on how to do this.

    See Also
    --------
    douglas_rachford_pd : Solver for similar problems.

    References
    ----------
    For a more detailed documentation see :ref:`chambolle_pock`.

    References on the Chambolle-Pock algorithm can be found in [CP2011a]_ and
    [CP2011b]_.

    This implementation of the CP algorithm is along the lines of
    [Sid+2012]_.

    For more on convex analysis including convex conjugates and
    resolvent operators see [Roc1970]_.

    For more on proximal operators and algorithms see [PB2014]_.

    The non-linear case is analyzed in [Val2014]_.
    """
    # Forward operator
    if not isinstance(op, Operator):
        raise TypeError('`op` {} is not an instance of {}'
                        ''.format(op, Operator))

    # Starting point
    if x not in op.domain:
        raise TypeError('`x` {} is not in the domain of `op` {}'
                        ''.format(x.space, op.domain))

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
        raise TypeError('`callback` {} is not `callable`'
                        ''.format(callback))

    # Initialize the relaxation variable
    x_relax = kwargs.pop('x_relax', None)
    if x_relax is None:
        x_relax = x.copy()
    elif x_relax not in op.domain:
        raise TypeError('`x_relax` {} is not in the domain of '
                        '`op` {}'.format(x_relax.space, op.domain))

    # Initialize the dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = op.range.zero()
    elif y not in op.range:
        raise TypeError('`y` {} is not in the range of `op` '
                        '{}'.format(y.space, op.range))

    # Temporary copy to store previous iterate
    x_old = x.space.element()

    for _ in range(niter):
        # Copy required for relaxation
        x_old.assign(x)

        # Gradient ascent in the dual variable y
        dual_tmp = y + sigma * op(x_relax)
        proximal_dual(sigma)(dual_tmp, out=y)

        # Gradient descent in the primal variable x
        primal_tmp = x + (- tau) * op.derivative(x).adjoint(y)
        proximal_primal(tau)(primal_tmp, out=x)

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
