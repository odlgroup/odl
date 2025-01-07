"""Example using the ray transform with 2d parallel beam geometry."""

import numpy as np
import odl

import matplotlib.pyplot as plt
plt.gray()

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32', impl='pytorch', torch_device='cpu')

angle_partition = odl.uniform_partition(0, np.pi, 180)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)
plt.matshow(proj_data)
plt.savefig('test_ray_trafo_2d_sinogram', bbox_inches='tight')
plt.close()

rec_data = ray_trafo.adjoint(proj_data)
plt.matshow(rec_data[0,:,:])
plt.savefig('test_ray_trafo_2d_reconstruction', bbox_inches='tight')
plt.close()