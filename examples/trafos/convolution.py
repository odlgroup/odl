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

"""Example for the usage of the convolution operator."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import odl


# Parameters for the gaussian kernel
gamma = 0.1
norm_const = 2 * np.pi * gamma ** 2


def gaussian(x):
    sum_sq = sum(xi ** 2 for xi in x)
    return np.exp(-sum_sq / (2 * gamma ** 2)) / norm_const


# Test function
def square(x):
    onedim_arrs = [np.where((xi >= -1) & (xi <= 1), 1.0, 0.0) for xi in x]
    out = onedim_arrs[0]
    for arr in onedim_arrs[1:]:
        out = out * arr
    return out


space = odl.uniform_discr([-2, -2], [2, 2], (2048, 2048))

# Showing just for fun
real_ker = space.element(gaussian)
real_ker.show(title='Gaussian kernel')
func = space.element(square)
func.show(title='Test function, a square')

conv = odl.Convolution(space, kernel=gaussian, kernel_mode='real',
                       impl='pyfftw_ft')

out = space.element()
conv.transform.create_temporaries()
conv.transform.init_fftw_plan()

with odl.util.testutils.Timer('first run, mode real'):
    func_conv = conv(func, out=out)

with odl.util.testutils.Timer('second run, mode real'):
    func_conv = conv(func, out=out)

conv.transform.create_temporaries()
with odl.util.testutils.Timer('third run, mode real, with tmp'):
    func_conv = conv(func, out=out)

func_conv.show(title='Convolved function')

# Giving the kernel via FT
ft = odl.trafos.FourierTransform(space, impl='pyfftw')
ker_ft = ft(gaussian)
ker_ft.show(title='FT of Gaussian kernel, half-complex')

conv_with_ker_ft = odl.Convolution(space, kernel=ker_ft, kernel_mode='ft_hc',
                                   impl='pyfftw_ft')

with odl.util.testutils.Timer('first run, mode ft_hc'):
    func_conv = conv_with_ker_ft(func, out=out)

conv_with_ker_ft.transform.create_temporaries()
with odl.util.testutils.Timer('second run, mode ft_hc, using tmp'):
    func_conv = conv_with_ker_ft(func, out=out)

func_conv.show(title='Convolved function, using kernel ft')
