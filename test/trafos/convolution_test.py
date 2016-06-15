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

"""Unit tests for `tensor_ops`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np

import odl
from odl.discr.discr_ops import Convolution
from odl.util.testutils import all_almost_equal_array


# ---- Convolution ---- #


ker_array = np.zeros((8, 8))
ker_array[4:5, 4:5] = 1


array_in = np.zeros((8, 8))
array_in[3, 5] = 1


def kernel(x):
    scaled_x = [xi / (np.sqrt(2) * 0.1) for xi in x]
    sq_sum = sum(xi ** 2 for xi in scaled_x)
    return np.exp(-sq_sum) / (2 * np.pi * 0.1 ** 2)


def kernel_autoconv(x):
    scaled_x = [xi / (2 * 0.1) for xi in x]
    sq_sum = sum(xi ** 2 for xi in scaled_x)
    return np.exp(-sq_sum) / (4 * np.pi * 0.1 ** 2)


def test_convolution_init_properties():
    # Checking if code runs at all

    # Real
    space = odl.uniform_discr([-1, -1], [1, 1], (8, 8))
    ft = odl.trafos.FourierTransform(space)

    Convolution(space, kernel)
    Convolution(space, space.element(kernel))
    Convolution(space, kernel, kernel_mode='real')
    Convolution(space, kernel, kernel_mode='real', impl='default_ft')
    Convolution(space, kernel, kernel_mode='real', impl='scipy_convolve')
    Convolution(space, kernel, kernel_mode='real', impl='scipy_fftconvolve')
    Convolution(space, kernel, kernel_mode='real', axes=(1,))
    Convolution(space, kernel, kernel_mode='real', impl='default_ft',
                axes=(1,))
    Convolution(space, kernel, kernel_mode='real', impl='scipy_convolve',
                axes=(1,))
    Convolution(space, kernel, kernel_mode='real', impl='scipy_fftconvolve',
                axes=(1,))
    Convolution(space, kernel, kernel_mode='ft')
    Convolution(space, kernel, kernel_mode='ft', axes=(1,))
    Convolution(space, kernel, kernel_mode='ft_hc')
    Convolution(space, kernel, kernel_mode='ft_hc', axes=(1,))
    Convolution(space, ft.range.element(kernel), kernel_mode='ft_hc')

    # Bad parameter values
    with pytest.raises(ValueError):
        Convolution(space, kernel, kernel_mode='fourier')

    with pytest.raises(ValueError):
        Convolution(space, kernel, impl='scipy')

    # Invalid combinations
    with pytest.raises(ValueError):
        Convolution(space, kernel, kernel_mode='ft', impl='scipy_convolve')

    with pytest.raises(ValueError):
        Convolution(space, kernel, kernel_mode='ft', impl='scipy_fftconvolve')

    with pytest.raises(ValueError):
        Convolution(space, kernel, kernel_mode='ft_hc', impl='scipy_convolve')

    with pytest.raises(ValueError):
        Convolution(space, kernel, kernel_mode='ft_hc',
                    impl='scipy_fftconvolve')

    # Custom range not implemented
    with pytest.raises(NotImplementedError):
        Convolution(space, kernel, ran=space)


var_params = [('real', 'default_ft'), ('real', 'scipy_convolve'),
              ('real', 'scipy_fftconvolve'), ('ft', 'default_ft'),
              ('ft_hc', 'default_ft')]
var_ids = ['kernel_mode = {}, impl = {}'.format(mode, impl)
           for (mode, impl) in var_params]


@pytest.fixture(scope="module", ids=var_ids, params=var_params)
def variant(request):
    return request.param


def test_convolution_call(variant):
    mode, impl = variant

    space = odl.uniform_discr([-2, -2], [2, 2], (8, 8))
    conv = Convolution(space, kernel=ker_array, kernel_mode=mode, impl=impl)

    true_conv_img = space.element(kernel_autoconv)
    conv_img = conv(kernel)
    assert all_almost_equal_array(autoconv, true_autoconv, places=7)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
