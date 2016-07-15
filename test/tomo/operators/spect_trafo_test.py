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
from odl.util.testutils import (skip_if_no_niftyrec,
                                almost_equal, all_equal,
                                all_almost_equal)
from odl.tomo.util.testutils import skip_if_no_astra_cuda


@skip_if_no_niftyrec
def test_spect_projector():
    """Test discrete SPECT transform."""

    # Geometry
    det_nx_pix = 32
    det_ny_pix = 32
    det_nx_mm = 4
    det_radius = 200
    n_proj = 16
    det_param = det_nx_mm * det_nx_pix
    dpart = odl.uniform_partition([-det_param, -det_param],
                                  [det_param, det_param],
                                  [det_nx_pix, det_ny_pix])

    apart = odl.uniform_partition(0, 2 * np.pi, n_proj)
    geometry = odl.tomo.geometry.ParallelHoleCollimatorGeometry(
        apart, dpart, det_rad=det_radius)

    # Create a discrete domain and a phantom as an element of the domain
    domain = odl.uniform_discr([-20] * 3, [20] * 3, [det_nx_pix] * 3)

    vol = odl.util.phantom.shepp_logan(domain, True)
    # Create a SPECT projector
    projector = AttenuatedRayTransform(
        domain, geometry, attenuation=None, impl='niftyrec_cpu')

    # Calculate projection and back-projection
    proj = projector(vol)
    backproj = projector.adjoint(proj)

    assert all_equal(proj.shape, (n_proj, det_nx_pix, det_ny_pix))
    assert all_equal(backproj.shape, vol.shape)
    assert backproj in projector.domain

    # Adjoint test:  Verified the identity <Ax, Ax> = <A^* A x, x>
    result_AxAx = proj.inner(proj)
    result_xAtAx = backproj.inner(vol)
    # Accept 1% errors
    places = 2
    assert almost_equal(result_AxAx, result_xAtAx, places=places)


@skip_if_no_astra_cuda
@skip_if_no_niftyrec
def test_attenuation():
    det_nx_pix = 16
    det_ny_pix = 16
    det_nx_mm = 0.125
    det_radius = 20
    n_proj = 1
    det_param = det_nx_mm * det_nx_pix

    domain = odl.uniform_discr([-1] * 3, [1] * 3, [det_nx_pix] * 3,
                               dtype='float32')

    dpart = odl.uniform_partition([-det_param / 2, -det_param / 2],
                                  [det_param / 2, det_param / 2],
                                  [det_nx_pix, det_ny_pix])

    apart = odl.uniform_partition(0, 2 * np.pi, n_proj)

    geometry_nifty = odl.tomo.geometry.ParallelHoleCollimatorGeometry(
        apart, dpart, det_rad=det_radius)
    geometry_astra = odl.tomo.geometry.Parallel3dAxisGeometry(
        apart, dpart)

    projector_nifty = AttenuatedRayTransform(
        domain, geometry_nifty, impl='niftyrec_cpu')
    projector_astra = odl.tomo.RayTransform(
        domain, geometry_astra, impl='astra_cuda')

    vol = np.zeros([det_nx_pix] * 3)
    vol[0, 0, 0] = 100
    vol_element = domain.element(vol)

    proj_nifty_noatt = projector_nifty(vol_element)
    proj_astra = projector_astra(vol_element)

    # Accept 0.1 % error
    places = 3
    # Without attenuation this should correcpond to
    # parallel 3D X-ray tomography
    assert all_almost_equal(proj_astra, proj_nifty_noatt, places=places)

    attenuation = np.zeros_like(vol)
    attenuation[0, 0, 0] = 0.2
    projector_nifty_att = AttenuatedRayTransform(
        domain, geometry_nifty, attenuation=attenuation, impl='niftyrec_cpu')
    proj_nifty_att = projector_nifty_att(vol_element)

    attenuated_ray_sum = 100 * domain.cell_sides[0] * np.exp(
        -0.2 * domain.cell_sides[0])
    attenuated_ray_sum_nifty = np.sum(proj_nifty_att.asarray()[0, -1, :])
    assert almost_equal(attenuated_ray_sum_nifty, attenuated_ray_sum,
                        places=places)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
