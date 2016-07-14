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

"""Example reconstruction with NiftyRec."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import odl
from odl.tomo import AttenuatedRayTransform


# Create SPECT geometry and reconstruction domain
dpart = odl.uniform_partition([-64, -64], [64, 64], [64, 64])

apart = odl.uniform_partition(0, 2 * np.pi, 64)
geometry = odl.tomo.geometry.ParallelHoleCollimatorGeometry(
    apart, dpart, det_rad=200)

domain = odl.uniform_discr([-20] * 3, [20] * 3, [64] * 3)

# Create phantoms
vol = odl.util.phantom.derenzo_sources(domain)
vol *= 100
derenzo_attenuation = odl.util.phantom.derenzo_sources(domain)
derenzo_attenuation *= 0.02
psf = np.ones((3, 3, 64))

# Create a SPECT projector
projector = AttenuatedRayTransform(
    domain, geometry, attenuation=derenzo_attenuation, impl='niftyrec_cpu',
    psf=psf)

# Calculate projections
data = projector(vol)

# Calculate operator norm for landweber
op_norm = projector.adjoint(data).norm() / vol.norm()
omega = 0.5 / op_norm

# Reconstruct using ODL
recon = domain.one()
odl.solvers.landweber(projector, recon, data, niter=10, omega=omega)
recon.show()
