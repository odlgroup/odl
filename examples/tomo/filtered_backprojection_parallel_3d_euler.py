"""Example using FBP in parallel 3D Euler geometry with regular 2D grid of
angles using `fbp_op`.
Compared against Landweber iteration which computes least squares solution for
discrete matrix system.
"""

import numpy as np
import odl
from matplotlib import pyplot as plt

# Reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
mid = 150
reco_space = odl.uniform_discr(
    min_pt=[-20] * 3, max_pt=[20] * 3, shape=[2 * mid] * 3,
    dtype='float32')

# Detector: uniformly sampled, n = (300, 300), min = (-40, -40), max = (40, 40)
detector_partition = odl.uniform_partition([-40] * 2, [40] * 2, [200] * 2)

# Make a parallel beam geometry with flat detector
# Angles: 20 x 20 Euler angles corresponding to an octant of the 3D unit sphere
angle_grid = odl.RectGrid(np.linspace(0, np.pi, 20),
                          np.linspace(0, np.pi, 20))
angle_partition = odl.uniform_partition_fromgrid(angle_grid)

# Geometry with tilted axis.
geometry = odl.tomo.Parallel3dEulerGeometry(angle_partition,
                                            detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# # Alternative phantom, anisotropic Gaussian
# x = reco_space.meshgrid
# phantom = reco_space.element(
#     np.exp(-.25 * (x[0] ** 2 + 2 * x[1] ** 2 + .5 * x[2] ** 2) / 2))

# Create FBP operator using utility function
# Default filter approximates least-squares solution
fbp = odl.tomo.fbp_op(ray_trafo)
# We can select a Hann filter, and only use the lowest 80% of frequencies to
# avoid high frequency noise.
# fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate filtered back-projection of data
fbp_rec = fbp(proj_data)

# Calculate true least squares solution with Landweber algorithm
lsqr_rec = fbp_rec.copy()
odl.solvers.landweber(ray_trafo, lsqr_rec, proj_data, 20)

plt.figure('FBP vs Landweber least-squares reconstruction')
plt.subplot(131)
plt.plot(fbp_rec.data[mid, mid, :])
plt.plot(lsqr_rec.data[mid, mid, :])
plt.plot(phantom.data[mid, mid, :])
plt.legend(('FBP', 'Landweber', 'phantom'))
plt.title('Slice x=y=0')
plt.subplot(132)
plt.plot(fbp_rec.data[:, mid, mid])
plt.plot(lsqr_rec.data[:, mid, mid])
plt.plot(phantom.data[:, mid, mid])
plt.title('Slice y=z=0')
plt.subplot(133)
plt.plot(fbp_rec.data[mid, :, mid])
plt.plot(lsqr_rec.data[mid, :, mid])
plt.plot(phantom.data[mid, :, mid])
plt.title('Slice x=z=0')

fbp_rec.show(title='FBP')
lsqr_rec.show(title='Landweber')
phantom.show(title='Phantom', force_show=True)
