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

"""Example using the ray transform with 3d parallel beam geometry."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import odl


# Discrete reconstruction space: discretized functions on the cube
# [-20, 20]^3 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_corner=[-20, -20, -20], max_corner=[20, 20, 20],
    nsamples=[300, 300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = (558, 558), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-30, -30], [30, 30], [558, 558])
# Discrete reconstruction space

# Astra cannot handle axis aligned origin_to_det unless it is aligned
# with the third coordinate axis. See issue #18 at ASTRA's github.
# This is fixed in new versions of astra, with older versions, this could
# give a zero result.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

# ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.util.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(indices=np.s_[:, :, 150], title='Phantom, middle z slice')
proj_data.show(indices=np.s_[0, :, :], title='Projection 0')
proj_data.show(indices=np.s_[90, :, :], title='Projection 90')
backproj.show(indices=np.s_[:, :, 150],
              title='back-projection, middle z slice')
