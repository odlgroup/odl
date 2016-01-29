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

"""Proximal operators.

The Proximal operators of f(x) is denoted by prox_tau[f](x) (or sometimes as
prox_{ tau f}( x) and defined as

    prox_tau[f](x) = arg min_y { f(y) + 1 / (2 * tau) * L2(x - y)^2 }

Separable sum property: if f is separable across two variables, i.e.
f(x, y) = g(x) + h(y), then

    prox_f(x, y) = prox_g(x) + prox_f(y)

Indicator function:

    ind_{S}(x) = {0 if x in S, infty if x not in S}

Special indicator function:

    ind_{box(a)}(x) = {0 if ||x||_infty <= a, infty if ||x||_infty > a}
"""
# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

# External
import scipy as sp

# Internal
from odl.operator.operator import Operator
from odl.operator.default_ops import IdentityOperator
from odl.operator.pspace_ops import ProductSpaceOperator
from odl.set.pspace import ProductSpace

__all__ = ('combine_proximals', 'proximal_zero', 'proximal_nonnegativity',
           'proximal_convexconjugate_l1', 'proximal_convexconjugate_l2')

# TODO: updated doc

# TODO: remove diagonal op
def combine_proximals(factory_list):

    def diagonal_operator(operators, dom=None, ran=None):
        """Broadcast argument to set of operators.

        Parameters
        ----------
        operators : array-like
            An array of `Operator`'s
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

        return  diagonal_operator(
            [factory(step_size) for factory in factory_list])

    return make_diag


def proximal_zero(space):
    """Function to create the proximal operator of f(x) = 0.

    Factory function which provides a function to initialize the proximal
    operator of ``f(x) = 0`` where ``x`` is an element in ``space``. The
    proximal operator of this functional is the identity operator.

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp`
        Domain of the functional ``f(x)``

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
            The proximal operator instance of ``f(x) = 0`` which is the
            identity operator
        """

        return IdentityOperator(space)

    return make_prox


# TODO: update doc
def proximal_nonnegativity(space):
    """Function to create the proximal operator of f(x) = 0.

    Factory function which provides a function to initialize the proximal
    operator of ``f(x)`` where ``x`` is an element in ``space``.

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp`
        Domain of the functional ``f(x)``

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
        id : `Operator`
            The proximal operator instance
        """

        class _ProxOpNonNegative(Operator):

            """The proximal operator."""

            def __init__(self):
                """Initialize the proximal operator.

                Parameters
                ----------
                tau : positive `float`
                """
                super().__init__(domain=space, range=space, linear=False)

            def _call(self, x, out):
                """Apply the operator to ``x`` and store the result in
                ``out``"""

                tmp = x.asarray()
                tmp[tmp < 0] = 0
                out[:] = tmp

        return _ProxOpNonNegative()

    return make_prox

# TODO: update doc
def proximal_convexconjugate_l2(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of an L2-data and
    L1-regularisation objective.

    Factory function providing a function to initialize the proximal
    operator of the convex conjugate of the functional ``F`` which is given
    by an L2-data term and an L1 semi-norm regularisation. The primal
    minimization problem thus reads

        F(x) = F(y, z) = 1/2 * ||y - g||_2^2 + lambda * ||(|z|)||_1

    The convex conjugate, F^*, of F is given by

        F^*(x) = 1/2 * ||y||_2^2 + <y,g> + ind_{box(lam)}(|z|)

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp`
        Domain of F(x)
    g : `DiscreteLpVector`
        Element in ``space``
    lam : positive `float`
        Regularization parameter

    Returns
    -------
    make_prox : `callable`
        Function which initializes the proximal operator at a given
        parameter value
    """
    lam = float(lam)

    if not g is None:
        if not g in space:
            raise TypeError('element ({}) does not belong to {}'
                            ''.format(g, space))
    else:  # g is None
        g = space.zero()

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
                super().__init__(domain=space, range=space, linear=True)

            def _call(self, x, out):
                """Apply the operator to ``x`` and stores the result in
                ``out``"""

                # (x - sig*g) / (1 + sig/lam)

                sig = self.sigma
                out.lincomb(1 / (1 + sig / lam), x, -sig / (1 + sig / lam), g)

        return _ProximalConvConjL2(sigma)

    return make_prox


# TODO: update doc
# TODO: add g to algorithm!!
def proximal_convexconjugate_l1(space, lam=1, g=None):
    """Proximal operator factory of the convex conjugate of an L2-data and
    L1-regularisation objective.

    Factory function providing a function to initialize the proximal
    operator of the convex conjugate of the functional ``F`` which is given
    by an L2-data term and an L1 semi-norm regularisation. The primal
    minimization problem thus reads

        F(x) = F(y, z) = 1/2 * ||y - g||_2^2 + lambda * ||(|z|)||_1

    The convex conjugate, F^*, of F is given by

        F^*(x) = 1/2 * ||y||_2^2 + <y,g> + ind_{box(lam)}(|z|)

    Parameters
    ----------
    space : `DiscreteLp` or `ProductSpace` of `DiscreteLp`
        Domain of F(x)
    g : `DiscreteLpVector`
        Element in ``space``
    lam : positive `float`
        Regularization parameter

    Returns
    -------
    make_prox : `callable`
        Function which initializes the proximal operator at a given
        parameter value
    """
    lam = float(lam)

    if not g is None:
        if not g in space:
            raise TypeError('element ({}) does not belong to {}'
                            ''.format(g, space))
    else:  # g is None
        g = space.zero()

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
                """Apply the operator to ``x`` and stores the result in
                ``out``"""
                sig = self.sigma

                # lam * (x - sig * g) / max(lam, |x - sig * g|)
                diff = x - sig * g

                if isinstance(x.space, ProductSpace):
                    # Calculate |x| = pointwise 2-norm of x

                    tmp = diff[0] ** 2
                    sq_tmp = x[0].space.element()
                    for xi in diff[1:]:
                        sq_tmp.multiply(xi, xi)
                        tmp += sq_tmp
                    tmp.ufunc.sqrt(out=tmp)

                    # Pointwise maximum of |x| and lambda
                    tmp.ufunc.maximum(lam, out=tmp)
                    # print('\ntmp', tmp.asarray())
                    # print('g', g[0].asarray(), g[1].asarray())
                    tmp /= lam

                    for oi, xi in zip(out, diff):
                        oi.divide(xi, tmp)

                else:
                    # Calculate |x| = pointwise 2-norm of x
                    tmp = diff.copy()
                    tmp.ufunc.absolute(out=tmp)

                    # Pointwise maximum of |x| and lambda
                    tmp.ufunc.maximum(lam, out=tmp)
                    tmp /= lam

                    out.divide(diff, tmp)

        return _ProximalConvConjL1(sigma)

    return make_prox


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
