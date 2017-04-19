"""Example for ray transform with 3d parallel beam and anisotropic voxels.

Anisotropic voxels are supported in ASTRA v1.8 and upwards; earlier versions
will trigger an error.
"""

import numpy as np
import odl

# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples in x and y, and 100 samples in z direction.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[300, 300, 100],
    dtype='float32')

# Make a 3d single-axis parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 180, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 180)
# Detector: uniformly sampled, n = (500, 500), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-30, -30], [30, 30], [500, 500])
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Show the slice y=0 of phantom and backprojection, as well as a projection
# image at theta=0 and a sinogram at v=0 (middle detector row)
phantom.show(coords=[None, 0, None], title='Phantom, middle y slice')
backproj.show(coords=[None, 0, None], title='Back-projection, middle y slice')
proj_data.show(coords=[0, None, None], title='Projection at theta=0')
proj_data.show(coords=[None, None, 0], title='Sinogram, middle slice')
