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

"""First-order primal-dual algorithm developed by Chambolle and Pock."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External

# Internal
from odl.operator.operator import Operator

__all__ = ('chambolle_pock_solver', 'f_dual_prox_l2_tv', 'g_prox_none')


# TODO: add dual gap as convergence measure
# TODO: diagonal preconditioning

def chambolle_pock_solver(K, f_dual_prox, g_prox, tau, sigma, theta=1,
                          niter=100, partial=None):
    """Chambolle-Pock aglgorithms for convex optimization problems.

    First order primal-dual hybrid-gradient (PDHG) method for non-smooth
    convex optimization probelms with known saddle-point structure developed
    by `Chambolle and Pock`_ (CP).


    Parameters
    ----------
    niter : `int`, optional
        Number of iterations
    partial : `Partial`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Returns
    -------
    x : `DiscreteLpVector`

    References
    ----------
    Original paper by `Chambolle and Pock`_. Also see `diagonal
    preconditioning`_ techniques. This implementation is based on the
    article on convex optimization problem prototyping for image
    reconstruction in computed tomography by `Sidky et al.`_.

    .. _Chambolle and Pock: http://dx.doi.org/10.1007/s10851-010-0251-1/
    .. _diagonal preconditioning: http://dx.doi.org/10.1109/ICCV.2011.6126441
    .. _Sidky et al.: http://stacks.iop.org/0031-9155/57/i=10/a=3065
    """
    x = K.domain.one()
    xbar = x.copy()
    y = K.range.zero()

    f_dual_prox_sigma = f_dual_prox(sigma)
    g_prox_tau = g_prox(tau)
    Kadjoint = K.adjoint

    for _ in range(niter):
        xold = x.copy()
        f_dual_prox_sigma(y + sigma * K(xbar), out=y)
        g_prox_tau(x - tau * Kadjoint(y), out=x)
        xbar = x + theta * (x - xold)

        # TODO: decide on what to send
        if partial is not None:
            partial.send((x[0], _))

    return x


def f_dual_prox_l2_tv(space, g, lam):
    """Function for the proximal operator with l2-data plus TV-regularization.

    Factory function which provides a function to initialize the proximal
    operator of the convex conjugate of the functional ``F`` given by the
    L2-data term and the isotropic total variation semi-norm as in the primal
    minimization problem:

    F(y,z)= 1/2 * ||y - g||_2^2 + lambda * ||(|grad u|)||_1

    with y = Au and z = grad u. The operators ``A`` and ``grad`` are
    combined to a matrix operator ``K`` as:

        K = (A, grad)^T

    The domain and range of the proximal operator are given by the range of
    ``K``.

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

    def make_prox(sigma):
        """Returns an instance of the proximal operator.

        Parameters
        ----------
        sigma : positive `float'
            Step length parameter

        Returns
        -------
        prox_op : `Operator`
            Proximal operator initialized with ``sigma``
        """

        class _prox_op(Operator):

            """The proximal operator."""

            def __init__(self, sigma):
                """Initialize the proximal operator.

                Parameters
                ----------
                sigma

                Returns
                -------

                """
                self.sigma = sigma
                super().__init__(domain=space, range=space)

            def _call(self, x, out=None):
                """Apply the operator to ``x`` and stores the result in
                ``out``"""

                y = x[0]
                z = x[1]

                # First component: (y - sig*g) / (1 + sig)

                sig = self.sigma
                out[0][...] = y / sig
                out[0] -= g
                out[0] *= sig / (1 + sig)

                # Second component: lam * z / (max(lam, |z|))

                # Calculate |z| = pointwise 2-norm of z
                tmp = z[0].space.zero()
                for zi in z:
                    tmp += zi ** 2
                tmp.ufunc.sqrt(out=tmp)

                # Pointwise maximum of |z| and lambda
                tmp.ufunc.maximum(lam, out=tmp)
                tmp /= lam

                out[1] = z.copy()
                for zi in out[1]:
                    zi /= tmp

        return _prox_op(sigma)

    return make_prox


def g_prox_none(space):
    """Function to create the proximal operator of the constraining functional.

    This factory function which provides a function to initialize the
    proximal operator for the functional which accounts for constraints on
    the image ``x``.


    Parameters
    ----------
    space : `DiscreteLp`
        Image domain

    Returns
    -------
    make_prox : `function`
        Function to initialize the proximal operator
    """

    def make_prox(tau):
        """Returns an instance of the proximal operator.

        Parameters
        ----------
        tau : positive `float'
            Step length parameter

        Returns
        -------
        prox_op : `Operator`
            Proximal operator initialized with ``sigma``

        """

        class _prox_op(Operator):

            """The proximal operator. """

            def __init__(self, tau):
                self.tau = tau
                super().__init__(domain=space, range=space)

            def _call(self, x, out=None):
                """TODO"""

                out.assign(x)

        return _prox_op(tau)

    return make_prox


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
