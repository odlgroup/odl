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

"""
Example using a filtered backprojection (FBP) in helical 3d using `fbp_op`.

Note that the FBP is only approximate in this geometry, but still gives a
decent reconstruction that can be used as an initial guess in more complicated
methods.

In helical geometries, the data is in general over-sampled which causes
streaking artefacts and a wrong scaling. This can be reduced using a
Tam-Danielson window.
"""

import numpy as np
import odl


# --- Set-up geometry of the problem --- #


# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^2 x [0, 40] with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[300, 300, 300],
    dtype='float32')

# Make a helical cone beam geometry with flat detector
# Angles: uniformly spaced, n = 2000, min = 0, max = 8 * 2 * pi
angle_partition = odl.uniform_partition(0, 8 * 2 * np.pi, 2000)
# Detector: uniformly sampled, n = (558, 60), min = (-30, -4), max = (30, 4)
detector_partition = odl.uniform_partition([-40, -4], [40, 4], [558, 60])
# Spiral has a pitch of 5, we run 8 rounds (due to max angle = 8 * 2 * pi)
geometry = odl.tomo.HelicalConeFlatGeometry(
    angle_partition, detector_partition, src_radius=100, det_radius=100,
    pitch=5.0)


# --- Create FilteredBackProjection (FBP) operator --- #


# Ray transform (= forward projection). We use the ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Unwindowed fbp
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', filter_cutoff=0.8)

# Create Tam-Danielson window to improve result
windowed_fbp = fbp * odl.tomo.tam_danielson_window(ray_trafo)


# --- Show some examples --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate filtered backprojection of data
fbp_reconstruction = fbp(proj_data)
w_fbp_reconstruction = windowed_fbp(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(title='Phantom')
proj_data.show(title='Projection data (sinogram)')
fbp_reconstruction.show(title='Filtered backprojection',
                        coords=[0, None, None], clim=[-0.1, 1.1])
w_fbp_reconstruction.show(title='Windowed filtered backprojection',
                          coords=[0, None, None], clim=[-0.1, 1.1])
