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

"""Total variation denoising using the Douglas-Rachford solver.

Solves the optimization problem

    min_{0 <= x <= 255}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1

where ``grad`` is the spatial gradient and ``g`` the given noisy data.
"""

import numpy as np
import scipy

import odl

# Parameters
n = 256
filter_width = 4.0

# Create a space
space = odl.uniform_discr([0, 0], [n, n], [n, n])

# Smoothing by fourier formula
ft = odl.trafos.FourierTransform(space)
const = filter_width**2 / 4.0**2  # by fourier transform of gaussian function
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * const))
smoothing = ft.inverse * gaussian * ft

# Load cameraman data and noise
data = space.element(np.rot90(scipy.misc.ascent()[::2, ::2], 3))
noise = space.element(np.random.randn(*space.shape) * 2.0)

# Create noisy convolved data
noisy_data = smoothing(data) + noise
data.show('Original data')
noisy_data.show('Noisy convolved data')

# Gradient for TV regularization
gradient = odl.Gradient(space)

# Assemble all operators
lin_ops = [smoothing, gradient]

# Create proximals as needed
prox_cc_g = [odl.solvers.proximal_cconj_l1(space, g=noisy_data),
             odl.solvers.proximal_cconj_l1(gradient.range, lam=0.03)]
prox_f = odl.solvers.proximal_box_constraint(space, 0, 255)

# Solve
x = space.zero()
callback = (odl.solvers.CallbackShow('results',
                                     display_step=20, clim=[0, 255]) &
            odl.solvers.CallbackPrintIteration())
odl.solvers.douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops,
                                tau=1.0, sigma=[1.0, 0.2], lam=1.0,
                                niter=2000, callback=callback)
