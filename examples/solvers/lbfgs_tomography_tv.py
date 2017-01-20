"""Tomography using the `bfgs_method` solver.

Solves an approximation of the optimization problem

    min_x ||A(x) - g||_2^2 + lam || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.

The problem is approximated by applying the Moreau envelope to ``|| . ||_1``
which gives a differentiable functional. This functional is equal to the so
called Huber functional.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
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

# Create data term ||Ax - b||_2^2 as composition of the squared L2 norm and the
# ray trafo translated by the data.
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range)
data_discrepancy = l2_norm * (ray_trafo - data)

# Create regularizing functional || |grad(x)| ||_1 and smooth the functional
# using the Moreau envelope.
# The parameter sigma controls the strength of the regularization.
gradient = odl.Gradient(reco_space)
l1_norm = odl.solvers.GroupL1Norm(gradient.range)
smoothed_l1 = odl.solvers.MoreauEnvelope(l1_norm, sigma=0.03)
regularizer = smoothed_l1 * gradient

# Create full objective functional
obj_fun = data_discrepancy + 0.03 * regularizer

# Create initial estimate of the inverse Hessian by a diagonal estimate
opnorm = odl.power_method_opnorm(ray_trafo)
hessinv_estimate = odl.ScalingOperator(reco_space, 1 / opnorm ** 2)

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Pick parameters
maxiter = 30
num_store = 5  # only save some vectors (Limited memory)

# Choose a starting point
x = ray_trafo.domain.zero()

# Run the algorithm
odl.solvers.bfgs_method(
    obj_fun, x, maxiter=maxiter, num_store=num_store,
    hessinv_estimate=hessinv_estimate, callback=callback)

# Display images
discr_phantom.show(title='original image')
data.show(title='sinogram')
x.show(title='reconstructed image', force_show=True)
