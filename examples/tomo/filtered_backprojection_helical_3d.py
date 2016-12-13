# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Example using FBP in helical 3D geometry using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complex
methods.

In helical geometries, the data are in general over-sampled which causes
streak artefacts and a wrong scaling. This can be reduced using a
Tam-Danielson window.
"""

import numpy as np
import odl


# --- Set up geometry of the problem --- #


# Reconstruction space: discretized functions on the cube
# [-20, 20]^2 x [0, 40] with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[300, 300, 300],
    dtype='float32')

# Make a helical cone beam geometry with flat detector
# Angles: uniformly spaced, n = 2000, min = 0, max = 8 * 2 * pi
# This gives 8 full turns of the helix.
angle_partition = odl.uniform_partition(0, 8 * 2 * np.pi, 2000)
# Detector: uniformly sampled with a small height,
# n = (558, 60), min = (-30, -4), max = (30, 4)
detector_partition = odl.uniform_partition([-40, -4], [40, 4], [558, 60])
# Create geometry
geometry = odl.tomo.HelicalConeFlatGeometry(
    angle_partition, detector_partition, src_radius=100, det_radius=100,
    pitch=5.0)


# --- Create Filtered Back-Projection (FBP) operator --- #


# Ray transform (= forward projection). We use the ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Unwindowed fbp
# We select a Hamming filter, and only use the lowest 80% of frequencies to
# avoid high frequency noise.
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hamming', frequency_scaling=0.8)

# Create Tam-Danielson window to improve result
windowed_fbp = fbp * odl.tomo.tam_danielson_window(ray_trafo)


# --- Show some examples --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate FBP reconstructions, once without window, once with window
fbp_reconstruction = fbp(proj_data)
w_fbp_reconstruction = windowed_fbp(proj_data)

# Show a slice of phantom, projections, and reconstruction
phantom.show(title='Phantom',
             coords=[0, None, None], clim=[-0.1, 1.1])
proj_data.show(title='Simulated data (sinogram)')
fbp_reconstruction.show(title='Filtered back-projection',
                        coords=[0, None, None], clim=[-0.1, 1.1])
w_fbp_reconstruction.show(title='Windowed filtered back-projection',
                          coords=[0, None, None], clim=[-0.1, 1.1])
