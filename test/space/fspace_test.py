# Copyright 2014, 2015 The ODL development group
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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest
from math import pi
import numpy as np

# ODL imports
import odl
from odl import FunctionSpace
from odl.util.testutils import almost_equal


# Pytest fixture


# Simply modify exp_params to modify the fixture
exp_params = [2.0, 1.0, float('inf'), 0.5, 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]
exp_fixture = pytest.fixture(scope="module", ids=exp_ids, params=exp_params)


@exp_fixture
def exponent(request):
    return request.param


def test_interval(exponent):
    fspace = FunctionSpace(odl.Interval(0, pi))
    lpdiscr = odl.uniform_discr(fspace, 10, exponent=exponent)

    if exponent == float('inf'):
        sine = fspace.element(np.sin)
        discr_sine = lpdiscr.element(sine)
        assert discr_sine.norm() <= 1
    else:
        sine_p = fspace.element(lambda x: np.sin(x) ** (1 / exponent))
        discr_sine_p = lpdiscr.element(sine_p)
        assert almost_equal(discr_sine_p.norm(), 2 ** (1 / exponent), places=2)


# TODO: sort out the vectorization issue first


@pytest.skip('vectorized evaluation broken')
def test_rectangle(exponent):
    fspace = FunctionSpace(odl.Rectangle((0, 0), (pi, 2 * pi)))
    n, m = 10, 10
    lpdiscr = odl.uniform_discr(fspace, (n, m), exponent=exponent)

    if exponent == float('inf'):
        sine2 = fspace.element(lambda x, y: np.sin(x) * np.sin(y))
        discr_sine = lpdiscr.element(sine2)
        assert discr_sine.norm() <= 1
    else:
        sine_p = fspace.element(
            lambda x, y: (np.sin(x) * np.sin(y)) ** (1 / exponent))
        discr_sine_p = lpdiscr.element(sine_p)
        assert almost_equal(discr_sine_p.norm(), 4 ** (1 / exponent), places=2)


def test_addition():
    fspace = FunctionSpace(odl.Interval(0, pi))
    sine = fspace.element(np.sin)
    cosine = fspace.element(np.cos)

    sum_func = sine + cosine

    for x in [0.0, 0.2, 1.0]:
        assert almost_equal(sum_func(x), np.sin(x) + np.cos(x))


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
