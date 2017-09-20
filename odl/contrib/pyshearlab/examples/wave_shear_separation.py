import odl
from odl.contrib.pyshearlab import PyShearlabOperator

space = odl.uniform_discr([-1, -1], [1, 1], [128, 128])
img = odl.phantom.forbild(space)
phantom = odl.phantom.derenzo_sources(space) + img

geom = odl.tomo.parallel_beam_geometry(space)
ray_trafo = odl.tomo.RayTransform(space, geom)
data = ray_trafo(phantom)

# Generate noisy data
noise = odl.phantom.white_noise(ray_trafo.range) * 0.01
noisy_data = data + noise

# Create shearlet and wavelet transforms
shear_op = PyShearlabOperator(space, scales=2)
wave_op = odl.trafos.WaveletTransform(space, 'haar', nlevels=2)
wave_op = wave_op / 89

# Functionals
sol_space = odl.ProductSpace(space, 2)

l1norm_wave = odl.solvers.L1Norm(wave_op.range)
l1norm_shear = odl.solvers.L1Norm(shear_op.range)
data_matching = odl.solvers.L2NormSquared(ray_trafo.range)
data_matching = data_matching.translated(noisy_data)

f = odl.solvers.ZeroFunctional(sol_space)
penalizer = odl.solvers.SeparableSum(100 * l1norm_wave,
                                     l1norm_shear)

# Forward operators
sum_op = odl.ReductionOperator(
    odl.IdentityOperator(space), 2)
coeff_op = odl.DiagonalOperator(wave_op, shear_op)

# Solve using douglas_rachford_pd
g = [data_matching, penalizer]
L = [ray_trafo * sum_op, coeff_op]

tau = 1
opnorms = [odl.power_method_opnorm(op) for op in L]
sigma = [1 / opnorm ** 2 for opnorm in opnorms]

callback = odl.solvers.CallbackShow(step=10)

images = sol_space.zero()
odl.solvers.douglas_rachford_pd(
    images, f, g, L, tau, sigma, niter=1000,
    callback=callback)
