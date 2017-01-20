"""Tomography using the `conjugate_gradient_normal` solver.

Solves the inverse problem

    A(x) = g

Where ``A`` is a parallel beam forward projector, ``x`` the result and
 ``g`` is given noisy data.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)

# Detector: uniformly sampled, n = 300, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 300)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)


# --- Generate artificial data --- #


# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
data = ray_trafo(discr_phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
x = ray_trafo.domain.zero()

# Run the algorithm
odl.solvers.conjugate_gradient_normal(
    ray_trafo, x, data, niter=20, callback=callback)

# Display images
discr_phantom.show(title='original image')
data.show(title='sinogram')
x.show(title='reconstructed image', force_show=True)
