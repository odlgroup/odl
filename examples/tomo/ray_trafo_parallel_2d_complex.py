"""Example computing the 2d parallel beam ray transform of a complex phantom.

This example demonstrates that forward and back-projections work for
complex-valued functions.
"""

import numpy as np
import odl

# Reconstruction space: discretized complex-valued functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='complex64')

# Parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). The backend is set explicitly -
# possible choices are 'astra_cpu', 'astra_cuda' and 'skimage'.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='skimage')

# Create a discretized phantom that is a Shepp-Logan phantom in the real
# part and a cuboid in the imaginary part
phantom = (odl.phantom.shepp_logan(reco_space, modified=True) +
           1j * odl.phantom.cuboid(reco_space))

# Create projection data by calling the ray transform on the phantom.
# This is equivalent to evaluating the ray transform on the real and
# imaginary parts separately.
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space). Also here,
# real and imaginary parts are back-projected separately.
backproj = ray_trafo.adjoint(proj_data)

# Show phantom, sinogram, and back-projected sinogram
phantom.show(title='Phantom')
proj_data.show(title='Projection data (sinogram)')
backproj.show(title='Back-projected data')
