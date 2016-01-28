# Copyright 2014, 2015 The ODL development group
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

"""Test reconstruction with ASTRA."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import odl
import odl.tomo as tomo
import pytest
import numpy as np


pytestmark = odl.util.skip_if_no_largescale


# Find the valid projectors
projectors = []
if tomo.ASTRA_AVAILABLE:
    projectors += ['par2d cpu', 'cone2d cpu']
if tomo.ASTRA_CUDA_AVAILABLE:
    projectors += ['par2d cuda', 'cone2d cuda',
                   'par3d cuda', 'cone3d cuda']


@pytest.fixture(scope="module",
                params=projectors)
def projector(request):
    geom, version = request.param.split()
    if geom == 'par2d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20],
                                             [20, 20],
                                             [100, 100], dtype='float32')

        # Geometry
        angle_intvl = odl.Interval(0, 2 * np.pi)
        dparams = odl.Interval(-30, 30)
        agrid = odl.uniform_sampling(angle_intvl, 200)
        dgrid = odl.uniform_sampling(dparams, 200)
        geom = tomo.Parallel2dGeometry(angle_intvl, dparams, agrid, dgrid)
        return tomo.XrayTransform(discr_reco_space, geom,
                                  backend='astra_' + version)

    elif geom == 'par3d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, -20],
                                             [20, 20, 20],
                                             [100, 100, 100], dtype='float32')

        # Geometry
        angle_intvl = odl.Interval(0, 2 * np.pi)
        dparams = odl.Rectangle([-30, -30], [30, 30])
        agrid = odl.uniform_sampling(angle_intvl, 200)
        dgrid = odl.uniform_sampling(dparams, [200, 200])
        geom = tomo.Parallel3dGeometry(angle_intvl, dparams, agrid, dgrid,
                                       axis=[1, 0, 0])

        # X-ray transform
        return tomo.XrayTransform(discr_reco_space, geom,
                                  backend='astra_' + version)

    elif geom == 'cone2d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20],
                                             [20, 20],
                                             [100, 100], dtype='float32')

        # Geometry
        angle_intvl = odl.Interval(0, 2 * np.pi)
        dparams = odl.Interval(-30, 30)
        agrid = odl.uniform_sampling(angle_intvl, 200)
        dgrid = odl.uniform_sampling(dparams, 200)
        geom = tomo.FanFlatGeometry(angle_intvl, dparams,
                                    src_radius=200, det_radius=100,
                                    agrid=agrid, dgrid=dgrid)

        # X-ray transform
        return tomo.XrayTransform(discr_reco_space, geom,
                                  backend='astra_' + version)

    elif geom == 'cone3d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, -20],
                                             [20, 20, 20],
                                             [100, 100, 100], dtype='float32')

        # Geometry
        angle_intvl = odl.Interval(0, 2 * np.pi)
        dparams = odl.Rectangle([-30, -30], [30, 30])
        agrid = odl.uniform_sampling(angle_intvl, 200)
        dgrid = odl.uniform_sampling(dparams, [200, 200])
        geom = tomo.CircularConeFlatGeometry(angle_intvl, dparams,
                                             src_radius=200, det_radius=100,
                                             agrid=agrid, dgrid=dgrid,
                                             axis=[1, 0, 0])

        # X-ray transform
        return tomo.XrayTransform(discr_reco_space, geom,
                                  backend='astra_' + version)

    elif geom == 'helical':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, 0],
                                             [20, 20, 40],
                                             [100, 100, 100], dtype='float32')

        # Geometry
        n_angle = 700
        angle_intvl = odl.Interval(0, 8 * 2 * np.pi)
        dparams = odl.Rectangle([-30, -3], [30, 3])
        agrid = odl.uniform_sampling(angle_intvl, n_angle)
        dgrid = odl.uniform_sampling(dparams, [200, 20])
        geom = tomo.HelicalConeFlatGeometry(angle_intvl, dparams, pitch=5.0,
                                            src_radius=200, det_radius=100,
                                            agrid=agrid, dgrid=dgrid)

        # X-ray transform
        return tomo.XrayTransform(discr_reco_space, geom,
                                  backend='astra_' + version)
    else:
        raise ValueError('param not valid')


def test_reconstruction(projector):
    """Test discrete X-ray transform using ASTRA for reconstruction."""

    # Create shepp-logan phantom
    vol = odl.util.shepp_logan(projector.domain, modified=True)

    # Project data
    projections = projector(vol)

    # Calculate operator norm for landweber
    op_norm_est_squared = projector.adjoint(projections).norm() / vol.norm()
    omega = 1.0 / op_norm_est_squared

    # Reconstruct using ODL
    recon = projector.domain.zero()
    odl.solvers.landweber(projector, recon, projections, niter=50,
                          omega=omega)

    # Make sure the result is somewhat close to the actual result.
    assert recon.dist(vol) < vol.norm() / 3.0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v --largescale'))
