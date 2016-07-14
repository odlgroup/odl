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
from odl.tomo.operators.spect_trafo import AttenuatedRayTransform


# Create SPECT geometry and reconstruction domain
det_nx_pix = 64
det_ny_pix = 64
det_nx_mm = 2
det_radius = 200
n_proj = 64
det_param = det_nx_mm * det_nx_pix / 2
dpart = odl.uniform_partition([-det_param, -det_param],
                              [det_param, det_param],
                              [det_nx_pix, det_ny_pix])

apart = odl.uniform_partition(0, 2 * np.pi, n_proj)
geometry = odl.tomo.geometry.ParallelHoleCollimatorGeometry(
    apart, dpart, det_rad=det_radius)

domain = odl.uniform_discr([-20] * 3, [20] * 3, [det_nx_pix] * 3)

# Create phantoms
vol = odl.util.phantom.derenzo_sources(domain)
vol *= 100
attenuation = odl.util.phantom.derenzo_sources(domain)
attenuation *= 0.02
psf = np.ones((3, 3, det_nx_pix))

# Create a SPECT projector
projector = AttenuatedRayTransform(
    domain, geometry, attenuation=attenuation, impl='niftyrec_cpu', psf=psf)

# Calculate projections
data = projector(vol)

# Calculate operator norm for landweber
op_norm_est_squared = projector.adjoint(data).norm() / vol.norm()
omega = 0.5 / op_norm_est_squared

# Reconstruct using ODL
recon = domain.one()
odl.solvers.landweber(projector, recon, data, niter=10, omega=omega)
recon.show()
