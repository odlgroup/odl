"""Example of using shearlet coefficients for signal separation.

In this example we define the shearlet transform S, the wavelet transform W and
use them to separate a signal according to

    min_{w,s}  ||W(w)||_1 + a ||S(s)||_1 + b ||w + s - x||_2^2

Here we expect the shearlet coefficients to contain the edges, and the wavelet
coefficients to contain the square.
"""

import odl
from odl.contrib import shearlab

space = odl.uniform_discr([-1, -1], [1, 1], [128, 128])
img = odl.phantom.ellipsoid_phantom(space, [[1, 0.02, 0.3, 0.5, 0, 0]])
img += odl.phantom.cuboid(space, [-0.3, -0.3], [0.3, 0.3])

# Generate noisy data
noise = odl.phantom.white_noise(space) * 0.001
noisy_data = img + noise

# Create shearlet and wavelet transforms
shear_op = shearlab.ShearlabOperator(space, num_scales=2)
wave_op = odl.trafos.WaveletTransform(space, 'haar', nlevels=2)
wave_op = wave_op / wave_op.norm(estimate=True)

# Functionals
sol_space = space ** 2

l1norm_wave = odl.solvers.L1Norm(wave_op.range)
l1norm_shear = odl.solvers.L1Norm(shear_op.range)
data_matching = 1000 * odl.solvers.L2NormSquared(space)
data_matching = data_matching.translated(noisy_data)

f = odl.solvers.ZeroFunctional(sol_space)
penalizer = odl.solvers.SeparableSum(0.05 * l1norm_wave,
                                     l1norm_shear)

# Forward operators
sum_op = odl.ReductionOperator(
    odl.IdentityOperator(space), 2)
coeff_op = odl.DiagonalOperator(wave_op, shear_op)

# Solve using douglas_rachford_pd
g = [data_matching, penalizer]
L = [sum_op, coeff_op]

tau = 1
opnorms = [odl.power_method_opnorm(op) for op in L]
sigma = [1 / opnorm ** 2 for opnorm in opnorms]

callback = odl.solvers.CallbackShow(step=10)

images = sol_space.zero()
odl.solvers.douglas_rachford_pd(
    images, f, g, L, tau, sigma, niter=1000,
    callback=callback)
