"""Tomography with TV regularization using the ProxImaL solver.

Solves the optimization problem

    min_{0 <= x <= 1}  ||A(x) - g||_2^2 + 0.2 || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.
"""

import numpy as np

import odl
import proximal

# --- Set up the forward operator (ray transform) --- #


# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Initialize the ray transform (forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Convert ray transform to proximal language operator
proximal_lang_ray_trafo = odl.as_proximal_lang_operator(ray_trafo)

# Create sinogram of forward projected phantom with noise
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
phantom.show('phantom')
data = ray_trafo(phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1
data.show('noisy data')

# Convert to array for ProxImaL
rhs_arr = data.asarray()

# Set up optimization problem
# Note that proximal is not aware of the underlying space and only works with
# matrices. Hence the norm in proximal does not match the norm in the ODL space
# exactly.
x = proximal.Variable(reco_space.shape)
funcs = [proximal.sum_squares(proximal_lang_ray_trafo(x) - rhs_arr),
         0.2 * proximal.norm1(proximal.grad(x)),
         proximal.nonneg(x),
         proximal.nonneg(1 - x)]

# Solve the problem using ProxImaL
prob = proximal.Problem(funcs)
prob.solve(verbose=True)

# Convert back to odl and display result
result_odl = reco_space.element(x.value)
result_odl.show('ProxImaL result', force_show=True)
