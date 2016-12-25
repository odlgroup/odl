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

"""Total variation MRI inversion using the Douglas-Rachford solver.

Solves the optimization problem

    min_{0 <= x <= 1} ||Ax - g||_2^2 + lam || |grad(x)| ||_1

where ``A`` is a simplified MRI imaging operator, ``grad`` is the spatial
gradient and ``g`` the given noisy data.
"""

import numpy as np
import odl

# Parameters
n = 256
subsampling = 0.5  # propotion of data to use
lam = 0.003

# Create a space
space = odl.uniform_discr([0, 0], [n, n], [n, n])

# Create MRI operator. First fourier transform, then subsample
ft = odl.trafos.FourierTransform(space)
sampling_points = np.random.rand(*ft.range.shape) < subsampling
sampling_mask = ft.range.element(sampling_points)
mri_op = sampling_mask * ft

# Create noisy MRI data
phantom = odl.phantom.shepp_logan(space, modified=True)
noisy_data = mri_op(phantom) + odl.phantom.white_noise(mri_op.range) * 0.1
phantom.show('Phantom')
noisy_data.show('Noisy MRI data')

# Gradient for TV regularization
gradient = odl.Gradient(space)

# Assemble all operators
lin_ops = [mri_op, gradient]

# Create functionals as needed
g = [odl.solvers.L2Norm(mri_op.range).translated(noisy_data),
     lam * odl.solvers.L1Norm(gradient.range)]
f = odl.solvers.IndicatorBox(space, 0, 1)

# Solve
x = mri_op.domain.zero()
callback = (odl.solvers.CallbackShow(display_step=20, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=1.0, sigma=[1.0, 0.2],
                                niter=500, callback=callback)

x.show('douglas rachford result')
ft.inverse(noisy_data).show('fourier inversion result')
