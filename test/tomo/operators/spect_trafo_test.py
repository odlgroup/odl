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

"""Tests for the Single-photon emission computerized tomography transform."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
from odl.tomo.operators.spect_trafo import AttenuatedRayTransform
from odl.util.testutils import skip_if_no_niftyrec


@skip_if_no_niftyrec
def test_spect_projector():
    """Test discrete SPECT transform."""

    # Geometry
    det_nx_pix = 16
    det_ny_pix = 16
    det_nx_mm = 4
    det_radius = 200
    n_proj = 90
    det_param = det_nx_mm * det_nx_pix
    dpart = odl.uniform_partition([-det_param, -det_param],
                                  [det_param, det_param],
                                  [det_nx_pix, det_ny_pix])

    apart = odl.uniform_partition(0, 2 * np.pi, n_proj)
    geometry = odl.tomo.geometry.spect.ParallelHoleCollimatorGeometry(
        apart, dpart, det_rad=det_radius)

    # Create a discrete domain and a phantom as an element of the domain
    domain = odl.uniform_discr([-20] * 3, [20] * 3, [det_nx_pix] * 3)

    vol = odl.util.phantom.shepp_logan(domain, True)
    # Create a SPECT projector
    projector = AttenuatedRayTransform(
        domain, geometry, attenuation=None, psf=None, impl='niftyrec')

    # Calculate projection and back-projection
    proj = projector(vol)
    backproj = projector.adjoint(proj)

    assert proj in projector.range
    assert backproj in projector.domain

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
