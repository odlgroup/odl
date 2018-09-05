# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Method(s) to find optimal reconstruction parameter(s) w.r.t. given FOM."""

import numpy as np
import scipy.optimize

__all__ = ('optimal_parameters', )


def optimal_parameters(reconstruction, fom, phantoms, data,
                       initial=None, univariate=False):
    r"""Find the optimal parameters for a reconstruction method.

    Notes
    -----
    For a forward operator :math:`A : X \to Y`, a reconstruction operator
    parametrized by :math:`\theta` is some operator
    :math:`R_\theta : Y \to X`
    such that

    .. math::
        R_\theta(A(x)) \approx x.

    The optimal choice of :math:`\theta` is given by

    .. math::
        \theta = \arg\min_\theta fom(R(A(x) + noise), x)

    where :math:`fom : X \times X \to \mathbb{R}` is a figure of merit.

    Parameters
    ----------
    reconstruction : callable
        Function that takes two parameters:

            * data : The data to be reconstructed
            * parameters : Parameters of the reconstruction method

        The function should return the reconstructed image.
    fom : callable
        Function that takes two parameters:

            * reconstructed_image
            * true_image

        and returns a scalar figure of merit.
    phantoms : sequence
        True images.
    data : sequence
        The data to reconstruct from.
    initial : array-like or pair
        Initial guess for the parameters. It is
        - a required array in the multivariate case
        - an optional pair in the univariate case.
    univariate : bool, optional
        Whether to use a univariate solver

    Returns
    -------
    parameters : 'numpy.ndarray'
        The  optimal parameters for the reconstruction problem.
    """

    def func(lam):
        # Function to be minimized by scipy
        return sum(fom(reconstruction(datai, lam), phantomi)
                   for phantomi, datai in zip(phantoms, data))

    # Pick resolution to fit the one used by the space
    tol = np.finfo(phantoms[0].space.dtype).resolution * 10

    if univariate:
        # We use a faster optimizer for the one parameter case
        result = scipy.optimize.minimize_scalar(
            func, bracket=initial, tol=tol, bounds=None,
            options={'disp': False})
        return result.x
    else:
        # Use a gradient free method to find the best parameters
        initial = np.asarray(initial)
        parameters = scipy.optimize.fmin_powell(
            func, initial, xtol=tol, ftol=tol, disp=False)
        return parameters
