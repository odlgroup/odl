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
# along with ODL. If not, see <http://www.gnu.org/licenses/>.

"""Factory functions for creating proximal operators.

For more details see :ref:`proximal_operators` and references therein. For
more details on proximal operators including how to evaluate the proximal
operator of a variety of functions see [PB2014]_. """

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import scipy as sp

from odl.operator.operator import Operator
from odl.operator.default_ops import IdentityOperator
from odl.operator.pspace_ops import ProductSpaceOperator
from odl.set.pspace import ProductSpace


__all__ = ('combine_proximals', 'proximal_zero', 'proximal_nonnegativity',
           'proximal_convexconjugate_l1', 'proximal_convexconjugate_l2',
           'proximal_convexconjugate_kl')


# TODO: remove diagonal op once available on master
def combine_proximals(factory_list):
    """Combine proximal operators into a diagonal product space operator.

    This assumes the functional to be separable across variables in order to
    make use of the separable sum property of proximal operators.

        prox_tau[f(x) + g(y)](x, y) = (prox_tau[f](x), prox_tau[g](y))

    Parameters
    ----------
    factory_list : list of `Operator`
        A list containing proximal operators which are created by the
        corresponding factory functions

    Returns
    -------
    diag_op : `Operator`
        Returns a diagonal product space operator to be initialized with
        the same step size parameter
    """

    def diagonal_operator(operators, dom=None, ran=None):
        """Broadcast argument to set of operators.

        Parameters
        ----------
        operators : array-like
            An array of `Operator`s
        dom : `ProductSpace`, optional
            Domain of the operator. If not provided, it is tried to be
            inferred from the operators. This requires each **column**
            to contain at least one operator.
        ran : `ProductSpace`, optional
            Range of the operator. If not provided, it is tried to be
            inferred from the operators. This requires each **row**
            to contain at least one operator.
        """

        indices = [range(len(operators)), range(len(operators))]
        shape = (len(operators), len(operators))
        op_matrix = sp.sparse.coo_matrix((operators, indices), shape)

        return ProductSpaceOperator(op_matrix, dom=dom, ran=ran)

    def make_diag(step_size):
        """Diagonal matrix of operators

        Parameters
        ----------
        step_size : positive `float`
            Step size parameter

        Returns
        -------
        diag_op : `Operator`
        """
        return diagonal_operator(
            [factory(step_size) for factory in factory_list])

    return make_diag


def proximal_zero(space):
    """Function to create the proximal operator of G(x) = 0.

    Function to initialize the proximal operator of G(x) = 0 where x is an
    element in ``space``. The proximal operator of this functional is the
    identity operator

        prox_tau[G](x) = x

    It is independent of tau.

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp` spaces
        Domain of the functional G

    Returns
    -------
    prox : `Operator`
        Returns the proximal operator to be initialized
    """

    def make_prox(tau):
        """Return an instance of the proximal operator.

        Parameters
        ----------
        tau : positive `float`
            Unused step size parameter. Introduced to provide a unified
            interface

        Returns
        -------
        id : `IdentityOperator`
            The proximal operator instance of G(x) = 0 which is the
            identity operator
        """

        return IdentityOperator(space)

    return make_prox


def proximal_nonnegativity(space):
    """Function to create the proximal operator of G(x) = ind(x > 0).

    Function for the proximal operator of the functional G(x)=ind(x > 0) to be
    initialized.

    If P is the set of non-negative elements, the indicator function of
    which is defined as

        ind(x > 0) = {0 if x in P, infinity if x is not in P}

    with x being an element in ``space``.

    The proximal operator of G is the point-wise non-negativity thresholding
    of x

         prox_tau[G](x) = {x if x > 0, 0 if <= 0}

    It is independent of tau and invariant under a positive rescaling of G
    which leaves the indicator function as it stands.

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp`
        Domain of the functional G(x)

    Returns
    -------
    prox : `Operator`
        Returns the proximal operator to be initialized
    """

    class _ProxOpNonNegative(Operator):

        """The proximal operator."""

        def __init__(self, tau):
            """Initialize the proximal operator.

            Parameters
            ----------
            tau : positive `float`
                Unused step size parameter. Introduced to provide a unified
                interface
            """
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and store the result in ``out``."""

            # Point-wise non-negativity thresholding: x if x > 0, else 0
            x.ufunc.maximum(0.0, out=out)

    return _ProxOpNonNegative


def proximal_convexconjugate_l2(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of the l2-norm.

    Function for the proximal operator of the convex conjugate of the
    functional F where F is the l2-norm

        F(x) =  lam 1/2 ||x - g||_2^2

    with x and g elements in ``space``, scaling factor lam, and given data g.

    The convex conjugate, F_cc, of F is given by

        F_cc(y) = 1/lam (1/2 ||y/lam||_2^2 + <y/lam,g>)

    The proximal operator of F_cc is given by

        prox_sigma[F_cc](y) = (y - sigma g) / (1 + sigma/lam)

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp`
        Domain of F(x)
    g : `DiscreteLpVector`
        An element in ``space``
    lam : positive `float`
        Scaling factor or regularization parameter

    Returns
    -------
    prox : `Operator`
        Returns the proximal operator to be initialized
    """
    lam = float(lam)

    if g is None:
        g = space.zero()
    else:
        if g not in space:
            raise TypeError('{} is not an element of {}'.format(g, space))

    class _ProximalConvConjL2(Operator):

        """The proximal operator."""

        def __init__(self, sigma):
            """Initialize the proximal operator.

            Parameters
            ----------
            sigma : positive `float`
                Step size parameter
            """
            self.sigma = float(sigma)
            super().__init__(domain=space, range=space, linear=False)

        def _call(self, x, out):
            """Apply the operator to ``x`` and stores the result in
            ``out``"""

            # (x - sig*g) / (1 + sig/lam)

            sig = self.sigma
            out.lincomb(1 / (1 + sig / lam), x, -sig / (1 + sig / lam), g)

    return _ProximalConvConjL2


def proximal_convexconjugate_l1(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of the l1-semi-norm.

    Function for the proximal operator of the convex conjugate of the
    functional F where F is an l1-semi-norm

        F(x) = lam || ||x-g||_p ||_1

    with x and g elements in ``space``, scaling factor lam, and point-wise
    magnitude ||x||_p of x. If x is vector-valued, ||x||_p is the point-wise
    l2-norm across the vector components.

    The convex conjugate, F_cc, of F is given by the indicator function of
    the set box(lam)

        F_cc(y) = lam ind_{box(lam)}(||y / lam||_p + <y / lam, g>)

    where box(lam) is a hypercube centered at the origin with width 2 lam.

    The proximal operator of F_cc is

        prox_sigma[F_cc](y) = lam (y - sigma g) / (max(lam 1_{||y||_p},
        ||y - sigma g||_p)

    where max(.,.) thresholds the lower bound of ||y||_p point-wise and
    1_{||y||_p} is a vector in the space of ||y||_p with all components set
    to 1.

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp` spaces
        Domain of the functional F
    g : `DiscreteLpVector`
        An element in ``space``
    lam : positive `float`
        Scaling factor or regularization parameter

    Returns
    -------
    prox : `Operator`
        Returns the proximal operator to be initialized
    """
    lam = float(lam)

    if g is None:
        g = space.zero()
    else:
        if g not in space:
            raise TypeError('{} is not an element of {}'.format(g, space))

    class _ProximalConvConjL1(Operator):

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
            """Apply the operator to ``x`` and stores the result in ``out``."""
            sig = self.sigma

            # lam * (x - sig * g) / max(lam, |x - sig * g|)

            diff = x - sig * g

            if isinstance(x.space, ProductSpace):
                # Calculate |x| = pointwise 2-norm of x

                tmp = diff[0] ** 2
                sq_tmp = x[0].space.element()
                for x_i in diff[1:]:
                    sq_tmp.multiply(x_i, x_i)
                    tmp += sq_tmp
                tmp.ufunc.sqrt(out=tmp)

                # Pointwise maximum of |x| and lambda
                tmp.ufunc.maximum(lam, out=tmp)

                # Global scaling
                tmp /= lam

                # Pointwise division
                for out_i, x_i in zip(out, diff):
                    out_i.divide(x_i, tmp)

            else:
                # Calculate |x| = pointwise 2-norm of x
                diff.ufunc.absolute(out=out)

                # Pointwise maximum of |x| and lambda
                out.ufunc.maximum(lam, out=out)

                # Global scaling
                out /= lam

                # Pointwise division
                out.divide(diff, out)

    return _ProximalConvConjL1


# TODO: move notes to ODL doc
def proximal_convexconjugate_kl(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of the KL divergence.

    Function returning the proximal operator of the convex conjugate of the
    functional F where F is the entropy-type Kullback-Leibler (KL) divergence

        F(x) = sum_i (x - g + g ln(g) - g ln(pos(x)))_i + ind_P(x)

    with x and g in X and g non-negative. The indicator function ind_P(x)
    for the positive elements of x is used to restrict the domain of F such
    that F is defined over whole X. The non-negativity thresholding pos is
    used to define F in the real numbers.

    The proximal operator of the convex conjugate, F_cc, of F is

        F_cc(p) = sum_i (-g ln(pos(1_X - p))_i + ind_P(1_X - p)

    where p is the variable dual to x, and 1_X is a vector in the space X with
    all components set to 1.

    The proximal operator of the convex conjugate of F is

        prox_sigma[F_cc](x) = 1/2 (lam_X + x - sqrt((x - lam_X)^2 + 4 lam
        sigma g)

    with the step size parameter sigma and lam_X is a vector in the space X
    with all components set to lam.

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp` spaces
        The space X which is the domain of the functional F
    g : `DiscreteLpVector`
        An element in ``space``
    lam : positive `float`
        Scaling factor

    Returns
    -------
    prox : `Operator`
        Returns the proximal operator to be initialized

    Notes
    -----
    KL based objectives are common in MLEM optimization problems and are often
    used when data noise governed by a multivariate Poisson probability
    distribution is significant.

    The intermediate image estimates can have negative values even though
    the converged solution will be non-negative. Non-negative intermediate
    image estimates can be enforced by adding an indicator function ind_P
    the primal objective.
    """
    lam = float(lam)

    if g is None:
        g = space.zero()
    else:
        if g not in space:
            raise TypeError('{} is not an element of {}'.format(g, space))

    class _ProximalConvConjKL(Operator):

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
            """Apply the operator to ``x`` and stores the result in ``out``."""
            sig = self.sigma

            # 1 / 2 (lam_X + x - sqrt((x - lam_X) ^ 2 + 4; lam sigma g)

            # TODO: optimize
            # out = x - lam_X
            out.lincomb(1, x, -lam, space.one())

            # (out)^2
            out.ufunc.square(out=out)

            # out = out + 4 lam sigma g
            out.lincomb(1, out, 4 * lam * sig, g)

            # out = sqrt(out)
            out.ufunc.sqrt(out=out)

            # out = x - out
            out.lincomb(1, x, -1, out)

            # out = lam_X + out
            out.lincomb(lam, space.one(), 1, out)

            # out = 1/2 * out
            out /= 2

    return _ProximalConvConjKL


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
