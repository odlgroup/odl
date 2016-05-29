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
from odl.solvers.util import Partial


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

    where ``X`` and ``Y`` are finite-dimensional Hilbert spaces, ``K``
    is a linear operator ``K : X -> Y``.  and ``G : X -> [0, +inf]``
    and ``F : Y -> [0, +inf]`` are proper, convex, lower-semicontinuous
    functionals.

    The Chambolle-Pock algorithm basically consists of alternating a
    gradient ascent in the dual variable y and a gradient descent in the
    primal variable x. The proximal operator is used to generate a ascent
    direction for the convex conjugate of F and descent direction for G.
    Additionally an over-relaxation in the primal variable is performed.

    Parameters
    ----------
    op : `Operator`
        A (product space) operator between Hilbert spaces with domain X
        and range Y
    x : element in the domain of ``op``
        Starting point of the iteration
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
        Relaxation parameter. Default: 1
    gamma : non-negative `float`, optional
        Acceleration parameter. If not `None` overwrites ``theta`` and uses
        variable relaxation parameter and step sizes with ``tau`` and
        ``sigma`` as initial values. Requires G or F_cc to be uniformly
        convex. Default: `None`
    partial : `Partial`, optional
        If not `None` the `Partial` instance(s) are executed in each
        iteration, e.g. plotting each iterate. Default: `None`
    precond_primal : `Operator`, optional
        Preconditioning operator for the primal variable, needs an
        `Operator.inverse` method. The operator needs to be symmetric and
        positive definite. Default: No preconditioning
    precond_dual : `Operator`, optional
        Preconditioning operator for the dual variable, needs an
        `Operator.inverse` method. The operator needs to be symmetric and
        positive definite. Default: No preconditioning
    x_relax : element in the domain of ``op``, optional
        Required to resume iteration. If `None` it is a copy of the primal
        variable x. Default: `None`
    y : element in the range of ``op``, optional
        Required to resume iteration. If `None` it is set to a zero element
        in Y which is the range of ``op``. Default: `None`

    Notes
    -----
    For a more detailed documentation see :ref:`chambolle_pock`.

    For references on the Chambolle-Pock algorithm see [CP2011a]_ and
    [CP2011b]_.

    This implementation of the CP algorithm is along the lines of
    [Sid+2012]_.

    For more on convex analysis including convex conjugates and
    resolvent operators see [Roc1970]_.

    For more on proximal operators and algorithms see [PB2014]_.
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
    if partial is not None and not callable(partial):
        raise TypeError('partial {} is not an instance of {}'
                        ''.format(op, Partial))

    # preconditioners
    precond_primal = kwargs.pop('precond_primal', None)
    if precond_primal is not None:
        if not isinstance(precond_primal, Operator):
            raise TypeError('precond_primal {} is not a `Operator`'
                            ''.format(precond_primal))
        if not precond_primal.is_linear:
            raise TypeError('precond_primal {} is not linear'
                            ''.format(precond_primal))
        if not precond_primal.domain == op.range:
            raise TypeError('precond_primal.domain {} is op.range {}'
                            ''.format(precond_primal.domain, op.range))
        if not precond_primal.range == op.range:
            raise TypeError('precond_primal.range {} is op.range {}'
                            ''.format(precond_primal.range, op.range))

    precond_dual = kwargs.pop('precond_dual', None)
    if precond_dual is not None:
        if not isinstance(precond_dual, Operator):
            raise TypeError('precond_dual {} is not a `Operator`'
                            ''.format(precond_dual))
        if not precond_dual.is_linear:
            raise TypeError('precond_dual {} is not linear'
                            ''.format(precond_dual))
        if not precond_dual.domain == op.range:
            raise TypeError('precond_dual.domain {} is op.range {}'
                            ''.format(precond_dual.domain, op.domain))
        if not precond_dual.range == op.range:
            raise TypeError('precond_dual.range {} is op.range {}'
                            ''.format(precond_dual.range, op.domain))

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
        if precond_dual is None:
            dual_tmp = y + sigma * op(x_relax)
        else:
            dual_tmp = y + sigma * precond_dual(op(x_relax))
        proximal_dual(sigma)(dual_tmp, out=y)

        # Gradient descent in the primal variable x
        if precond_primal is None:
            primal_tmp = x + (- tau) * op_adjoint(y)
        else:
            primal_tmp = x + (- tau) * precond_primal(op_adjoint(y))
        proximal_primal(tau)(primal_tmp, out=x)

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
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
