# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for parameter optimization."""

import pytest
import numpy as np
import odl
import odl.contrib.fom
import odl.contrib.param_opt
from odl.util.testutils import simple_fixture

space = simple_fixture('space',
                       [odl.rn(3),
                        odl.uniform_discr([0, 0], [1, 1], [9, 11]),
                        odl.uniform_discr(0, 1, 10)])

fom = simple_fixture('fom',
                     [odl.contrib.fom.mean_squared_error,
                      odl.contrib.fom.mean_absolute_error])


def test_optimal_parameters_one_parameter(space, fom):
    """Tests if optimal_parameters works for some simple examples."""
    noise = [odl.phantom.white_noise(space) for _ in range(2)]
    phantoms = noise.copy()
    data = noise.copy()

    def reconstruction(data, lam):
        """Simple reconstruction operator.

        The optimal reconstruction parameter in any fom is lam=0.
        """
        return 1.23 * data / lam

    result = odl.contrib.param_opt.optimal_parameters(
        reconstruction, fom, phantoms, data, [.9, 1.1], univariate=True)
    assert result == pytest.approx(1.23, abs=1e-4)


def test_optimal_parameters_two_parameters(space, fom):
    """Tests if optimal_parameters works for some simple examples."""
    noise = [odl.phantom.white_noise(space) for _ in range(2)]
    # Normalize to reduce test fails due to randomness
    noise = [noise_elem / noise_elem.norm() for noise_elem in noise]
    phantoms = noise.copy()
    data = noise.copy()

    def reconstruction1(data, params):
        """Perturbs the data scaling it with a nonlinear function of params."""
        return data * (1 - params[1]) / np.abs(params[0])

    result1 = odl.contrib.param_opt.optimal_parameters(reconstruction1, fom,
                                                       phantoms, data, [1, 2])
    assert 1 - result1[1] == pytest.approx(result1[0], abs=1e-4)

    def reconstruction2(data, params):
        """Perturbs the data by adding the sum of params to it."""
        return data + sum(params)

    result2 = odl.contrib.param_opt.optimal_parameters(reconstruction2, fom,
                                                       phantoms, data, [1, 2])
    assert sum(result2) == pytest.approx(0, abs=1e-4)


if __name__ == '__main__':
    odl.util.test_file(__file__)
