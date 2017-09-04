"""Tomography using the `kaczmarz` solver.

Solves the inverse problem

    A(x) = g

Where ``A`` is a parallel beam forward projector, ``x`` the result and
 ``g`` is given noisy data.

In order to solve this using `kaczmarz`s method, the operator is split into
several sub-operators (each representing a subset of the angles).
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)

# Detector: uniformly sampled, n = 300, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 300)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operators
n = 10  # number of sub-operators
ray_trafos = [odl.tomo.RayTransform(space, geometry[i::n]) for i in range(n)]

# --- Generate artificial data --- #


# Create phantom
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create sinogram of forward projected phantom with noise
data = [ray_trafo(phantom) for ray_trafo in ray_trafos]
noisy_data = [d + odl.phantom.white_noise(d.space) * np.mean(d) * 0.1
              for d in data]

omega = [odl.power_method_opnorm(ray_trafo) ** (-2)
         for ray_trafo in ray_trafos]

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Choose a starting point
x = ray_trafo.domain.zero()

# Run the algorithm
odl.solvers.kaczmarz(
    ray_trafos, x, noisy_data, niter=20, omega=omega, callback=callback)

# Display images
phantom.show(title='original image')
data.show(title='sinogram')
x.show(title='reconstructed image', force_show=True)
