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

"""Maximum Likelihood Expectation Maximization algorithm."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np

__all__ = ('mlem', 'loglikelihood')


AVAILABLE_MLEM_NOISE = ('poisson',)


def mlem(op, x, rhs, niter=1, noise='poisson', callback=None):

    """Maximum Likelihood Expectation Maximation algorithm.

    Attempts to solve::

        min_x L(x | rhs)

    where ``L(x, | rhs)`` is the likelihood of ``x`` given data ``rhs``. The
    likelihood depends on the underlying noise model and the forward operator.

    Parameters
    ----------
    op : `Operator`
        Operator in the inverse problem. It has to have adjoint that is called
        via `op.adjoint`

    x : ``element`` of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
        The initial value of `x` should be non negative.

    rhs : ``element`` of the range of ``op``
        Right-hand side of the equation defining the inverse problem

    niter : `int`, optional
        Number of iterations.

    noise : {'poisson'}, optional
        Implementation back-end for the noise.

    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate

    Notes
    -----
    Given a forward model :math:`A`, data :math:`g` and poisson noise the
    algorithm is given by:

    .. math::

       x_{n+1} = \\frac{x_n}{A^* 1} A^* (g / A(x_n))

    See Also
    --------
    loglikelihood : Function for calculating the logarithm of the likelihood
    """
    if np.any(np.less(x, 0)):
        raise ValueError('`x` must be non-negative')

    noise, noise_in = str(noise).lower(), noise
    if noise not in AVAILABLE_MLEM_NOISE:
        raise NotImplemented("noise '{}' not understood"
                             ''.format(noise_in))

    if noise == 'poisson':
        eps = 1e-8

        norm = np.maximum(op.adjoint(op.range.one()), eps)
        tmp_dom = op.domain.element()
        tmp_ran = op.range.element()

        for _ in range(niter):
            op(x, out=tmp_ran)
            tmp_ran.ufunc.maximum(eps, out=tmp_ran)
            tmp_ran.divide(rhs, tmp_ran)

            op.adjoint(tmp_ran, out=tmp_dom)
            tmp_dom /= norm

            x *= tmp_dom

            if callback is not None:
                callback(x)
    else:
        raise RuntimeError('unknown noise model')


def loglikelihood(op, x, data, noise='poisson'):
    """Evaluate a log-likelihood at a given point.

    Parameters
    ----------
    op : `Operator`
        Forward operator of the given problem, the operator in
        the inverse problem.

    x : `element` of the domain of ``op``
        A point where the logarithm of the likelihood density is evaluated

    data : `element` of the range of ``op``
        Right-hand side of the equation defining the inverse problem
    '
    noise : {'poisson'}, optional
        Implementation back-end for the noise.
    """
    noise, noise_in = str(noise).lower(), noise
    if noise not in AVAILABLE_MLEM_NOISE:
        raise NotImplemented("noise '{}' not understood"
                             ''.format(noise_in))

    if noise == 'poisson':
        projection = op(x)
        projection += 1e-8
        log_proj = np.log(projection)
        return np.sum(data * log_proj - projection)
    else:
        raise RuntimeError('unknown noise model')
