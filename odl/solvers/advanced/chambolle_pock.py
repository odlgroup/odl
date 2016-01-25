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

The method is proposed in `[1]_` and augmented with diagonal preconditioners
in [2]_. The algorithm is flexible and particularly suited for non-smooth,
convex optimization problems in imaging. This implementation is along the
lines of `[3]`_.

.. [1]: http://dx.doi.org/10.1007/s10851-010-0251-1
.. [2]: http://dx.doi.org/10.1109/ICCV.2011.6126441
.. [3]: http://stacks.iop.org/0031-9155/57/i=10/a=3065
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External

# Internal
from odl.operator.operator import Operator
from odl.operator.default_ops import IdentityOperator
from odl.solvers.util import Partial

__all__ = ('chambolle_pock_solver', 'f_cc_prox_l2_tv', 'g_prox_none')


# TODO: add dual gap as convergence measure
# TODO: diagonal preconditioning
# TODO: add positivity constraint
# TODO: improve doc including reference
# TODO: split f_cc_prox_l2_tv
# TODO: add checks

def chambolle_pock_solver(op, x, tau, sigma, proximal_primal, proximal_dual,
                          theta=1, niter=1, partial=None, x_relaxation=None,
                          y=None):
    """Chambolle-Pock algorithm for non-smooth convex optimization problems.

    First order primal-dual hybrid-gradient (PDHG) method for non-smooth
    convex optimization problems with known saddle-point structure
    developed by Chambolle and Pock (CP) [1]_.

    Generic saddle-point optimization problem:

        min_{x in X} max_{y in Y} <Kx,y>_Y + G(x) - F^*(y)

    where X and Y are finite-dimensional Hilbert spaces with inner product
    <.,.>, K is a continuous linear operator K : X -> Y. G : X -> [0,+infty]
    and F^* : Y -> [0,+infty] are proper, lower-semicontinuous functions with
    F^* being the convex (or Fenchel) conjugate of

    Corresponding primal minimization problem

        min_x G(x) + F(Kx)

    Corresponding dual maximization:

        max_y G(-K^*x) - F^*(y)

    Convergence proven for ``||K||_2^2 * sigma * tau < 1``

    For the use of CP with simple and easy to use preconditioning techniques
    see [2]_. This implementation is based on the
    article on convex optimization problem prototyping for image
    reconstruction in computed tomography by [3]_. For more on
    proximal operators and algorithms see [4]_.

    Parameters
    ----------
    op : `Operator`
        A (product space) operator between Hilbert spaces with domain ``X``
        and range ``Y``
    x : element in the domain of ``op``
        Starting point of the iteration with ``x`` in ``X``
    tau : positive `float`
        Parameter similar to a step size for the update of the primal
        variable ``x``. Controls the extent to which ``proximal_primal``
        maps points towards the minimum of ``G``.
    sigma : positive `float`
        Parameter similar to a step size for the update of the dual
        variable ``y``. Controls the extent to which ``proximal_dual``
        maps points towards the minimum of ``F^*``.
    proximal_primal : callable `function`
        Evaluated at ``tau``, the function returns the proximal operator,
        ``prox_tau[G](x)``, of the function ``G``. The domain of ``G`` and
        the returned proximal operator is the space, ``X``, of the primal
        variable ``x``  i.e. domain of ``op``.
    proximal_dual : callable `function`
        Evaluated at ``sigma``, the function returns the proximal operator,
        ``prox_sigma[F^*](x)``, of the convex conjugate, ``F^*``, of the
        function ``F``. The domain of ``F^*`` and the returned proximal
        operator is the space, ``Y``, of the dual variable ``y = op(x)``
          i.e. the range of ``op``.
    theta : `float` in [0, 1], optional
        Relaxation parameter
    niter : non-negative `int`, optional
        Number of iterations
    partial : `Partial`, optional
        If not `None` the object passed executes code per iteration,
        e.g. plotting each iterate
    x_relaxation : element in the domain of ``op``
        Required to resume iteration. If `None` it is equal to ``x``.
    y : element in the range of ``op``
        Required to resume iteration. If `None` it is set to zero.

    References
    ----------
    .. [1]: Chambolle, Antonin, and Pock, Thomas. *A First-Order Primal-Dual
    Algorithm for Convex Problems with Applications to Imaging*. J. Math.
    Imaging Vis. 40 120, 2011.
    http://dx.doi.org/10.1007/s10851-010-0251-1

    .. [2]: Chambolle, Antonin, and Pock, Thomas. *Diagonal preconditioning
    for first order primal-dual algorithms in convex optimization*. 2011
    IEEE International Conference on Computer Vision (ICCV), 1762, 2011.
     http://dx.doi.org/10.1109/ICCV.2011.6126441

    .. [3]: Emil Y, Sidky, Jakob H, Jorgensen, and Xiaochuan, Pan. *Convex
    optimization problem prototyping for image reconstruction in computed
    tomography with the Chambolle-Pock algorithm*, Phys Med Biol, 57 3065,
    2012. http://stacks.iop.org/0031-9155/57/i=10/a=3065

    .. [4]: Parikh Neal, and Boyd Stephen. *Proximal Algorithms* .
    Foundations and Trends in Optimization 1 127, 2014.
    http://dx.doi.org/10.1561/2400000003
    """
    if not isinstance(op, Operator):
        raise TypeError('operator ({}) is not an instance of {}'
                        ''.format(op, Operator))

    if x.space != op.domain:
        raise TypeError('starting point ({}) is not in the domain of `op` '
                        '({})'.format(x.space, op.domain))
    if tau <= 0:
        raise ValueError('update parameter ({0}) not positive.'.format(tau))
    else:
        tau = float(tau)
    if sigma <= 0:
        raise ValueError('update parameter ({0})  not positive.'.format(sigma))
    else:
        sigma = float(sigma)
    if not 0 <= theta <= 1:
        raise ValueError('relaxation parameter ({0}) not in [0, 1].'
                         ''.format(theta))
    else:
        theta = float(theta)
    if not isinstance(niter, int) or niter < 0:
        raise ValueError('number of iterations ({0}) not valid.'
                         ''.format(niter))
    if not (partial is None or isinstance(partial, Partial)):
        raise TypeError('partial ({}) is not an instance of {}'
                        ''.format(op, Partial))

    # Relaxation variable
    if x_relaxation is None:
            x_relaxation = x.copy()
    else:
        if x_relaxation.space != op.domain:
            raise TypeError('relaxation variable {} is not in the domain of '
                            '`op` ({})'.format(x_relaxation.space, op.domain))

    # Initialize the dual variable
    if y is None:
        y = op.range.zero()
    else:
        if y.space != op.domain:
            raise TypeError('variable {} is not in the range of `op` '
                            '({})'.format(x_relaxation.space, op.domain))

    # Temporal copy of previous iterate
    x_old = x.space.element()

    # Initialize the proximal operator of the convex conjugate of functional F
    f_cc_prox_sigma = proximal_dual(sigma)
    # Initialize the proximal operator of functional G
    g_prox_tau = proximal_primal(tau)

    # Adjoint of the product space operator
    op_adjoint = op.adjoint

    for _ in range(niter):
        # Copy required for relaxation
        x_old.assign(x)

        # Gradient descent in the dual variable y
        f_cc_prox_sigma(y + sigma * op(x_relaxation), out=y)

        # Gradient descent in the primal variable x
        g_prox_tau(x + (- tau) * op_adjoint(y), out=x)

        # Over-relaxation in the primal variable x
        x_relaxation.lincomb(1 + theta, x, -theta, x_old)

        if partial is not None:
            partial.send(x)


def f_cc_prox_l2_tv(space, g, lam):
    """Function for the proximal operator with l2-data plus TV-regularization.

    Factory function which provides a function to initialize the proximal
    operator of the convex conjugate of the functional ``F`` given by the
    L2-data term and the isotropic total variation semi-norm of in the primal
    minimization problem:

    F(y,z)= 1/2 * ||y - g||_2^2 + lambda * ||(|grad u|)||_1

    where y = Au and z = grad u. The operators ``A`` and ``grad`` are
    combined in a matrix operator ``K`` as:

        K = (A, grad)^T

    The proximal operator is mapping from a vector space X to its dual space
    X^*. We assume X to be Hilbert spaces which is self-dual. Here, the domain
    and range of the proximal operator are given by the range of ``K``.

    Parameters
    ----------
    space : `ProductSpace`
        Product space of the range of forward operator ``A`` and of the range
        of the gradient operator i.e. a product space of the image space
    g : `DiscreteLpVector`
        Element in the range of the forward operator ``A``
    lam : positive `float`
        Regularization parameter

    Returns
    -------
    make_prox : `function`
        Function initialize the proximal operator with a given step
        length parameter ``sigma``
    """
    lam = float(lam)

    def make_prox(sigma):
        """Returns an instance of the proximal operator.

        Parameters
        ----------
        sigma : positive `float`
            Step length parameter

        Returns
        -------
        prox_op : `Operator`
            Proximal operator initialized with ``sigma``
        """

        class _ProxOp(Operator):

            """The proximal operator."""

            def __init__(self, sigma):
                """Initialize the proximal operator.

                Parameters
                ----------
                sigma : positive `float`
                """
                self.sigma = float(sigma)
                super().__init__(domain=space, range=space, linear=False)

            def _call(self, x, out):
                """Apply the operator to ``x`` and stores the result in
                ``out``"""

                y = x[0]
                z = x[1]

                # First component: (y - sig*g) / (1 + sig)

                sig = self.sigma
                out[0].lincomb(1 / (1 + sig), y, -sig / (1 + sig), g)

                # Second component: lam * z / (max(lam, |z|))

                # Calculate |z| = pointwise 2-norm of z
                tmp = z[0] ** 2
                sq_tmp = z[0].space.element()
                for zi in z[1:]:
                    sq_tmp.multiply(zi, zi)
                    tmp += sq_tmp
                tmp.ufunc.sqrt(out=tmp)

                # Pointwise maximum of |z| and lambda
                tmp.ufunc.maximum(lam, out=tmp)
                tmp /= lam

                for oi, zi in zip(out[1], z):
                    oi.divide(zi, tmp)

        return _ProxOp(sigma)

    return make_prox


def g_prox_none(space):
    """Function to create the proximal operator of the constraining functional.

    Factory function which provides a function to initialize the proximal
    operator of the functional ``G`` which accounts for constraints on the
    primal variable ``x``.

    Parameters
    ----------
    space : domain of ``op``
        Space of the domain of the (combined) operator ``op`` in which the
        primal variable ``x`` resides in

    Returns
    -------
    make_prox : `function`
        Function to initialize the proximal operator
    """

    def make_prox(tau):
        """Return an instance of the proximal operator.

        Parameters
        ----------
        tau : positive `float`
            Step length parameter. Unused here but introduced to provide a
            common interface

        Returns
        -------
        id : `IdentityOperator`
            The proximal operator for an unconstrained the primal variable
            is the identity map
        """

        return IdentityOperator(space)

    return make_prox


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
