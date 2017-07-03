"""Example using the ray transform with circular cone beam geometry."""

import numpy as np
import odl

# Reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[300, 300, 300],
    dtype='float32')

# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = (512, 512), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-30, -30], [30, 30], [512, 512])
geometry = odl.tomo.ConeFlatGeometry(
    angle_partition, detector_partition, src_radius=1000, det_radius=100,
    axis=[1, 0, 0])

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(coords=[None, None, 0], title='Phantom, middle z slice')
proj_data.show(coords=[0, None, None], title='Projection at theta=0')
proj_data.show(coords=[None, None, 0], title='Sinogram, middle slice')
backproj.show(coords=[None, None, 0], title='Back-projection, middle z slice')
