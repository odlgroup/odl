"""Example creating a filtered back-projection (FBP) in 2d for complex data.

This example is intended to show that FBP works transparently on complex
spaces without any change beyond a re-definition of ``reco_space`` and
possibly a ``phantom`` with non-zero imaginary part.

See the example ``filtered_backprojection_parallel_2d.py`` for further
details.
"""

import numpy as np
import odl


# --- Set-up geometry of the problem --- #


# Discrete reconstruction space: discretized complex-valued functions on the
# rectangle [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='complex64')

# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 1000)

# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 500)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


# --- Create Filtered Back-Projection (FBP) operator --- #


# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create filtered back-projection operator
fbp = odl.tomo.fbp_op(ray_trafo)


# --- Show some examples --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = (odl.phantom.shepp_logan(reco_space, modified=True) +
           1j * odl.phantom.cuboid(reco_space))

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate filtered back-projection of data
fbp_reconstruction = fbp(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(title='Phantom')
proj_data.show(title='Projection data (sinogram)')
fbp_reconstruction.show(title='Filtered back-projection')
(phantom - fbp_reconstruction).show(title='Error')
