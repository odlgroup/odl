"""
Example using a filtered back-projection (FBP) in fan beam using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complicated
methods.

Here we look at a partial scan, where the angular interval is not 2 * pi.
This caues issues for the regular FBP reconstruction, but can be improved
via a Parker weighting.
"""

import numpy as np
import odl


# --- Set-up geometry of the problem --- #


# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32')

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi + fan angle
angle_partition = odl.uniform_partition(0, np.pi + 0.7, 360)
# Detector: uniformly sampled, n = 558, min = -40, max = 40
detector_partition = odl.uniform_partition(-40, 40, 558)
# Geometry with large fan angle
geometry = odl.tomo.FanFlatGeometry(
    angle_partition, detector_partition, src_radius=80, det_radius=40)


# --- Create Filtered Back-Projection (FBP) operator --- #


# Ray transform (= forward projection). We use the ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create FBP operator using utility function
# We select a Hann filter, and only use the lowest 80% of frequencies to avoid
# high frequency noise.
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

# Apply parker weighting in order to improve reconstruction
parker_weighting = odl.tomo.parker_weighting(ray_trafo)
parker_weighting.show()
parker_weighted_fbp = fbp * parker_weighting

# --- Show some examples --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate filtered back-projection of data
fbp_reconstruction = fbp(proj_data)
pw_fbp_reconstruction = parker_weighted_fbp(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(title='Phantom')
proj_data.show(title='Projection data (sinogram)')
fbp_reconstruction.show(title='Filtered back-projection')
pw_fbp_reconstruction.show(title='Parker weighted filtered back-projection')
