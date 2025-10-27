"""
Example using a filtered back-projection (FBP) in fan beam using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complicated
methods.
"""

import numpy as np
import odl
import time

# --- Set up geometry of the problem --- #


# Reconstruction space: discretized functions on the cube
# [-20, 20]^2 with 300 samples per dimension.

for (impl, device) in [('pytorch', 'cpu'), ('pytorch', 'cuda:0'), ('numpy', 'cpu')]:
    reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[300, 300, 300],
    dtype='float32', impl=impl, device=device)

    # Make a circular cone beam geometry with flat detector
    # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
    # Detector: uniformly sampled, n = (512, 512), min = (-40, -40), max = (40, 40)
    detector_partition = odl.uniform_partition([-40, -40], [40, 40], [512, 512])
    # Geometry with large cone and fan angle and tilted axis.
    geometry = odl.applications.tomo.ConeBeamGeometry(
        angle_partition, detector_partition, src_radius=40, det_radius=40,
        axis=[1, 1, 1])


    # --- Create Filtered Back-projection (FBP) operator --- #


    # Ray transform (= forward projection).
    ray_trafo = odl.applications.tomo.RayTransform(reco_space, geometry)

    # Create FBP operator using utility function
    # We select a Hann filter, and only use the lowest 80% of frequencies to avoid
    # high frequency noise.
    fbp = odl.applications.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)


    # --- Show some examples --- #


    # Create a discrete Shepp-Logan phantom (modified version)
    phantom = odl.core.phantom.shepp_logan(reco_space, modified=True)

    t0 = time.time()

    # Create projection data by calling the ray transform on the phantom
    proj_data = ray_trafo(phantom)

    # Calculate filtered back-projection of data
    fbp_reconstruction = fbp(proj_data)

    t1 = time.time()
    print(f'Reconstruction Time for {impl} backend on {device} device : {t1-t0:.2f}')
