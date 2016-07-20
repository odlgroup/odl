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

__all__ = ('mlem_poisson',)


def mlem_poisson(op, x, data, iter=1, noise='poisson', callback=None):

    """Implementation of Maximum Likelihood Expectation Maximation algorithm.

    This method computes the MLEM estimates for Poisson noise.

    Parameters
    ----------
    op : `Operator`
        Operator in the inverse problem. It has to have adjoint that is called
        via `op.adjoint`

    x : `element` of the domain of ``op``
        Vector to which the result is written. Its initial value is
        used as starting point of the iteration, and its values are
        updated in each iteration step.
        The initial value of `x` should be non negative.

    data : `element` of the range of ``op``
        Right-hand side of the equation defining the inverse problem

    niter : `int`, optional
        Maximum number of iterations

    callback : `callable`, optional
        Object executing code per iteration, e.g. plotting each iterate

    noise : `str`, optional
        Implementation back-end for the noise.
    """
    if x <= 0:
        raise ValueError('Initial value of x needs to be positive')

    noise = str(noise).lower()

    if noise.startswith('poisson'):
        norm = op.adjoint(op.range.one())
        eps = 1e-8
        for _ in range(iter):
            projection = op(x)
            sino_per_proj = data / np.maximum(projection, eps)
            update = op.adjoint(sino_per_proj)
            x = (x * update) / np.maximum(norm, eps)

            if callback is not None:
                callback(x)
    else:
        raise NotImplemented('implemented for poisson noise, got {}'
                             ''.format(noise))

    return x


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
    noise : `str`, optional
        Implementation back-end for the noise.
    """
    noise = str(noise).lower()

    if noise.startswith('poisson'):
        projection = op(x)
        projection += 1e-8
        log_proj = np.log(projection)
        return np.sum(data * log_proj - projection)

    else:
        raise NotImplemented('implemented for poisson noise, got {}'
                             ''.format(noise))


def gradient_poisson_loglikelihood(op, x, data, noise='poisson'):
    """Evaluate the derivative of the log-likelihood at x.

    Parameters
    ----------
    op : `Operator`
        Forward operator of the given problem, the operator in
        the inverse problem. It has to have adjoint that is called
        via `op.adjoint`

    x : `element` of the domain of ``op``
        A point where the gradient of the log-likelihood is evaluated

    data : `element` of the range of ``op``
        Right-hand side of the equation defining the inverse problem

    noise : `str`, optional
        Implementation back-end for the noise.
    """
    noise = str(noise).lower()

    if noise.startswith('poisson'):
        projection = op(x)
        sino_per_proj = data / np.maximum(projection, 1e-8)
        grad = op.adjoint(sino_per_proj) - op.adjoint(op.range.one())
        return grad
    else:
        raise NotImplemented('implemented for poisson noise, got {}'
                             ''.format(noise))
