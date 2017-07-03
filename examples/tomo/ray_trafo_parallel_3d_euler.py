"""Example of using the ray transform in 3D parallel geometry with 2 angles.

The `Parallel3dEulerGeometry` is defined in terms of 2 or 3 Euler angles
and is not aligned to a rotation axis.
"""

import numpy as np
import odl


# Reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[300, 300, 300],
    dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: 20 x 20 Euler angles corresponding to an octant of the 3D unit sphere
angle_grid = odl.RectGrid(np.linspace(0, np.pi / 2, 20),
                          np.linspace(0, np.pi / 2, 20))
angle_partition = odl.uniform_partition_fromgrid(angle_grid)

# Detector: uniformly sampled, n = (500, 500), min = (-40, -40), max = (40, 40)
detector_partition = odl.uniform_partition([-40, -40], [40, 40], [500, 500])
# Geometry with tilted axis.
geometry = odl.tomo.Parallel3dEulerGeometry(angle_partition,
                                            detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate back-projection of the data
backproj = ray_trafo.adjoint(proj_data)

# Show a slice of phantom, projections, and reconstruction
phantom.show(title='Phantom')
proj_data.show(title='Simulated data: sinogram for theta=0 and v=0',
               coords=[None, 0, None, 0])
proj_data.show(title='Simulated data: sinogram for phi=0 and v=0',
               coords=[0, None, None, 0])
proj_data.show(title='Simulated data: "cone plot" for u=0 and v=0',
               coords=[None, None, 0, 0])
proj_data.show(title='Simulated data: projection for phi=pi/4 and theta=pi/4',
               coords=[np.pi / 4, np.pi / 4, None, None])

backproj.show(title='Back-projection, slice z=0')
