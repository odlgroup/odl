"""Tomography using the `bfgs_method` solver.

Solves the optimization problem

    min_x ||A(x) - g||_2^2

Where ``A`` is a parallel beam forward projector, ``x`` the result and
 ``g`` is given noisy data.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 200 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[200, 200])

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 400, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 400)

# Detector: uniformly sampled, n = 400, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 400)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# --- Generate artificial data --- #


# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
data = ray_trafo(discr_phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1

# --- Set up optimization problem and solve --- #

# Create objective functional ||Ax - b||_2^2 as composition of l2 norm squared
# and the residual operator.
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - data)

# Create line search
line_search = 1.0
# line_search = odl.solvers.BacktrackingLineSearch(obj_fun)

# Create initial estimate of the inverse Hessian by a diagonal estimate
opnorm = odl.power_method_opnorm(ray_trafo)
hessinv_estimate = odl.ScalingOperator(reco_space, 1 / opnorm ** 2)

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))

# Pick parameters
maxiter = 20
num_store = 5  # only save some vectors (Limited memory)

# Choose a starting point
x = ray_trafo.domain.zero()

# Run the algorithm
odl.solvers.bfgs_method(
    obj_fun, x, line_search=line_search, maxiter=maxiter, num_store=num_store,
    hessinv_estimate=hessinv_estimate, callback=callback)

odl.solvers.douglas_rachford_pd

# Display images
discr_phantom.show(title='Original Image')
data.show(title='Sinogram')
x.show(title='Reconstructed Image', force_show=True)
