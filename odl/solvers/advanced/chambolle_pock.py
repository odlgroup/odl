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

The method is proposed in [CP2011a]_ and augmented with diagonal
preconditioners in [CP2011b]_. The algorithm is flexible and well apt suited
for non-smooth, convex optimization problems in imaging. This implementation
is along the lines of [Sid+2012]_.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

# Internal
from odl.operator.operator import Operator
from odl.solvers.util import Partial

__all__ = ('chambolle_pock_solver',)


# TODO: add dual gap as convergence measure
# TODO: diagonal preconditioning

def chambolle_pock_solver(op, x, tau, sigma, proximal_primal, proximal_dual,
                          niter=1, **kwargs):
    """Chambolle-Pock algorithm for non-smooth convex optimization problems.

    The Chambolle-Pock (CP) algorithm, as proposed in [CP2011a]_, is a first
    order primal-dual hybrid-gradient method for non-smooth convex
    optimization problems with known saddle-point structure

        min_{x in X} max_{y in Y} <K x, y>_Y + G(x) - F_cc(y)

    where X and Y are finite-dimensional Hilbert spaces with inner product
    <.,.> and norm ||.||_2 = <.,.>^(1/2), K is a continuous linear operator
    K : X -> Y. G : X -> [0, +infinity] and F_cc : Y -> [0, +infinity] are
    proper, lower-semicontinuous functionals, and F_cc is the convex (or
    Fenchel) conjugate of F, see below.

    The saddle-point problem is a primal-dual formulation of the following
    primal minimization problem

        min_{x in X} G(x) + F(Kx)

    The corresponding dual maximization problem is

        max_{y in Y} G(-K_adj x) - F_cc(y)

    with K_adj being the adjoint of K.

    The convex conjugate is a mapping from a normed vector space X to its dual
    space X_dual and defined by

        F_cc(x_dual) = sup_{x in X} <x_dual, x> - F(x)

    with x_dual in X_dual and dual pairing <.,.>. For Hilbert spaces,
    which are self-dual, we have X = X_dual and <.,.> is the inner product.
    The convex conjugate is always convex, and if F is convex, proper,
    and lower semi-continuous we have F = (F_cc)_cc. For more details
    see [Roc1970]_.


    Algorithm

    The CP algorithm basically consists of alternating a gradient ascend in
    the dual variable y and a gradient descent in the primal variable x.
    Additionally an over-relaxation in the primal variable is performed.

    Initialization:

        choose tau > 0, sigma > 0, theta in [0,1], x_0 in X, y_0 in Y,
        xr_0 = x_0

    Iteration: for n > 0 update x_n, y_n, and xr_n as follows

        y_{n+1} = prox_sigma[F_cc](y_n + sigma K xr_n)

        x_{n+1} = prox_tau[G](x_n - tau  K_adj y_{n+1})

        xr_{n+1} = x_{n+1} + theta (x_{n+1} - x_n)

    The proximal operator, prox_tau[f](x), of the functional H with step
    size parameter tau is defined as

        prox_tau[H](x) = arg min_y f(y) + 1 / (2 tau) ||x - y||_2^2

    A simple choice of step size parameters is tau = sigma < 1 / ||K|| with
    the induced operator norm

        ||K|| = max{||K x|| : x in X, ||x|| < 1}

    For ||K||^2 sigma tau < 1 the algorithms should converge.

    If G or F_cc is uniformly convex, convergence can be accelerated using
    variable step sizes: replace tau -> tau_n, sigma -> sigma_n, and theta
    -> theta_n with t_0 sigma_0 ||K||^2 < 1 and gamma > 0. After the update
    of the primal variable (x_{n+1}) and before the update of the relaxation
    variable (xr_{n+1}) update the parameters as

        theta_n = 1 / sqrt(1 + 2 gamma tau_n)

        tau_{n+1} = theta_n tau_n

        sigma_{n+1} = sigma_n / theta_n

    Instead of choosing step size parameters preconditioning techniques can
    be employed, see [CP2011b]_. In this case the steps tau and sigma are
    replaced by symmetric and positive definite matrices tau -> T, sigma ->
    Sigma and convergence should hold for ||Sigma^(1/2) K T^(1/2)||^2 < 1.

    For more on proximal operators and algorithms see [PB2014]_. The
    following implementation of the CP algorithm is along the lines of
    [Sid+2012]_.

    Parameters
    ----------
    op : `Operator`
        A (product space) operator between Hilbert spaces with domain X
        and range Y
    x : element in the domain of ``op``
        Starting point of the iteration with x in X
    tau : positive `float`
        Step size parameter for the update of the primal variable x.
        Controls the extent to which ``proximal_primal`` maps points
        towards the minimum of G.
    sigma : positive `float`
        Step size parameter for the update of the dual variable y. Controls
        the extent to which ``proximal_dual`` maps points towards the
        minimum of F_cc.
    proximal_primal : `callable`
        Evaluated at ``tau``, the function returns the proximal operator,
        prox_tau[G](x), of the functional G. The domain of G and its
        proximal operator instance are the space, X, of the primal variable
        x  i.e. the domain of ``op``.
    proximal_dual : `callable`
        Evaluated at ``sigma``, the function returns the proximal operator,
        prox_sigma[F_cc](x), of the convex conjugate, F_cc, of the function
        F. The domain of F_cc and its proximal operator instance are the
        space, Y, of the dual variable y i.e. the range of ``op``.
    niter : non-negative `int`, optional
        Number of iterations

    Other Parameters
    ----------------
    theta : `float` in [0, 1], optional
        Relaxation parameter. Default: `None`
    gamma : non-negative `float`, optional
        Acceleration parameter. If not `None` overwrites ``theta`` and uses
        variable relaxation parameter and step sizes with ``tau`` and
        ``sigma`` as initial values. Requires G or F_cc to be uniformly
        convex. Default: `None`
    partial : `Partial`, optional
        If not `None` the `Partial` instance(s) are executed in each
        iteration, e.g. plotting each iterate. Default: `None`
    x_relax : element in the domain of ``op``, optional
        Required to resume iteration. If `None` it is a copy of the primal
        variable x. Default: `None`
    y : element in the range of ``op``, optional
        Required to resume iteration. If `None` it is set to a zero element
        in Y which is the range of ``op``. Default: `None`

    References
    ----------
    .. [CP2011a] `Chambolle, Antonin and Pock, Thomas. *A First-Order
       Primal-Dual Algorithm for Convex Problems with Applications to
       Imaging*. Journal of Mathematical Imaging and Vision, 40 (2011), pp 120.

    .. [CP2011b] `Chambolle, Antonin and Pock, Thomas. *Diagonal
       preconditioning for first order primal-dual algorithms in convex
       optimization*. 2011 IEEE International Conference on Computer Vision
       (ICCV), 2011, pp 1762-1769.

    .. [Sid+2012] `Sidky, Emil Y, Jorgensen, Jakob H, and Pan, Xiaochuan.
       *Convex optimization problem prototyping for image reconstruction in
       computed tomography with the Chambolle-Pock algorithm*. Physics in
       Medicine and Biology, 57 (2012), pp 3065-3091.

    .. [PB2014] Parikh, Neal and Boyd, Stephen. *Proximal Algorithms*.
       Foundations and Trends in Optimization, 1 (2014), pp 127-239.

    .. [Roc1970] Rockafellar, R. Tyrrell. *Convex analysis*. Princeton
       University Press, 1970.
    """
    # Forward operator
    if not isinstance(op, Operator):
        raise TypeError('operator {} is not an instance of {}'
                        ''.format(op, Operator))

    # Starting point
    if x.space != op.domain:
        raise TypeError('starting point {} is not in the domain of `op` {}'
                        ''.format(x.space, op.domain))

    # Step size parameter
    if tau <= 0:
        raise ValueError('step size parameter {} not positive.'.format(tau))
    else:
        tau = float(tau)

    # Step size parameter
    if sigma <= 0:
        raise ValueError('step size parameter {} not positive.'.format(sigma))
    else:
        sigma = float(sigma)

    # Number of iterations
    if not isinstance(niter, int) or niter < 0:
        raise ValueError('number of iterations {} not valid.'
                         ''.format(niter))

    # Relaxation parameter
    theta = kwargs.pop('theta', 1)
    if not 0 <= theta <= 1:
        raise ValueError('relaxation parameter {} not in [0, 1].'
                         ''.format(theta))
    else:
        theta = float(theta)

    # Acceleration parameter
    gamma = kwargs.pop('gamma', None)
    if gamma is not None:
        if gamma < 0:
            raise ValueError('acceleration parameter {} not '
                             'non-negative'.format(gamma))
        else:
            gamma = float(gamma)

    # Partial object
    partial = kwargs.pop('partial', None)
    if not (partial is None or isinstance(partial, Partial)):
        raise TypeError('partial {} is not an instance of {}'
                        ''.format(op, Partial))

    # Initialize the relaxation variable
    x_relax = kwargs.pop('x_relax', None)
    if x_relax is None:
        x_relax = x.copy()
    else:
        if x_relax.space != op.domain:
            raise TypeError('relaxation variable {} is not in the domain of '
                            '`op` {}'.format(x_relax.space, op.domain))

    # Initialize the dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = op.range.zero()
    else:
        if y.space != op.range:
            raise TypeError('variable {} is not in the range of `op` '
                            '{}'.format(y.space, op.range))

    # Temporal copy to store previous iterate
    x_old = x.space.element()

    # Adjoint of the (product space) operator
    op_adjoint = op.adjoint

    for _ in range(niter):
        # Copy required for relaxation
        x_old.assign(x)

        # Gradient ascent in the dual variable y
        proximal_dual(sigma)(y + sigma * op(x_relax), out=y)

        # Gradient descent in the primal variable x
        proximal_primal(tau)(x + (- tau) * op_adjoint(y), out=x)

        # Acceleration
        if gamma is not None:
            theta = float(1 / np.sqrt(1 + 2 * gamma * tau))
            tau *= theta
            sigma /= theta

        # Over-relaxation in the primal variable x
        x_relax.lincomb(1 + theta, x, -theta, x_old)

        if partial is not None:
            partial(x)


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
