"""Example of Bregman-TV in tomography.

The example does five iterations in a Bregman-TV scheme, applied to a
tomography problem.
"""

import odl
import numpy as np

# Discrete reconstruction space
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(0, np.pi, 100)
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the ray transform
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')


# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
noise_free_data = ray_trafo(discr_phantom)
noise = odl.phantom.white_noise(ray_trafo.range)
noise = noise * 1/noise.norm() * noise_free_data.norm() * 0.10
data = noise_free_data + noise


# Do a FBP-reconstruction
fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.4)
fbp_reco = fbp_op(data)


# Display phantom, FBP reconstruction, and sinograms
discr_phantom.show(title='Phantom')
fbp_reco.show(title='FBP reconstruction')
data.show(title='Simulated data (Sinogram)')


# Define components of the variational problem
reg_param = 0.3
gradient = odl.Gradient(reco_space)

# Column vector of two operators
op = odl.BroadcastOperator(ray_trafo, gradient)

# Do not use the g functional, set it to zero
g = odl.solvers.ZeroFunctional(reco_space)

# l2-squared data matching
l2_norm = 0.5 * odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# Isotropic TV-regularization
l1_norm = odl.solvers.GroupL1Norm(gradient.range)

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 500  # Number of iterations in the inner loop
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Create initial guess for the solver.
x = reco_space.zero()

# This defines the Bregman iterations
for breg_iter in range(5):
    print('Outer Bregman iteration: {}'.format(breg_iter))

    # Define the bregman functional with base point in the previous iterate
    bregman_func = reg_param * l1_norm.bregman(gradient(x))

    # Combine functionals, order must correspond to the operator K
    f = odl.solvers.SeparableSum(l2_norm, bregman_func)

    # Used to print iteration number.
    callback_inner = odl.solvers.CallbackPrintIteration(step=50)

    # Call the solver. x is updated in-place with the consecutive iterates.
    odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                     callback=callback_inner)

    # Display the result after this iteration
    x.show(title='Outer Bregman iteration {}'.format(breg_iter),
           force_show=True)
    x.show(title='Outer Bregman iteration {}, pixel values [0,1]'
           ''.format(breg_iter), clim=[0, 1], force_show=True)
