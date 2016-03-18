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

"""Test reconstruction with ASTRA."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
import odl.tomo as tomo


pytestmark = odl.util.skip_if_no_largescale


# Find the valid projectors
projectors = ['helical cuda']
if tomo.ASTRA_AVAILABLE:
    projectors += ['par2d cpu', 'cone2d cpu']
if tomo.ASTRA_CUDA_AVAILABLE:
    projectors += ['par2d cuda', 'cone2d cuda',
                   'par3d cuda', 'cone3d cuda']


@pytest.fixture(scope="module", params=projectors)
def projector(request):
    geom, variant = request.param.split()
    if geom == 'par2d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20], [20, 20],
                                             [100, 100], dtype='float32')

        # Geometry
        apart = odl.uniform_partition(0, 2 * np.pi, 200)
        dpart = odl.uniform_partition(-30, 30, 200)
        geom = tomo.Parallel2dGeometry(apart, dpart)

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl='astra_' + variant)

    elif geom == 'par3d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                             [100, 100, 100], dtype='float32')

        # Geometry
        apart = odl.uniform_partition(0, 2 * np.pi, 200)
        dpart = odl.uniform_partition([-30, -30], [30, 30], [200, 200])
        geom = tomo.Parallel3dAxisGeometry(apart, dpart, axis=[1, 0, 0])

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl='astra_' + variant)

    elif geom == 'cone2d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20], [20, 20],
                                             [100, 100], dtype='float32')

        # Geometry
        apart = odl.uniform_partition(0, 2 * np.pi, 200)
        dpart = odl.uniform_partition(-30, 30, 200)
        geom = tomo.FanFlatGeometry(apart, dpart,
                                    src_radius=200, det_radius=100)

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl='astra_' + variant)

    elif geom == 'cone3d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                             [100, 100, 100], dtype='float32')

        # Geometry
        apart = odl.uniform_partition(0, 2 * np.pi, 200)
        dpart = odl.uniform_partition([-30, -30], [30, 30], [200, 200])
        geom = tomo.CircularConeFlatGeometry(
            apart, dpart, src_radius=200, det_radius=100, axis=[1, 0, 0])

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl='astra_' + variant)

    elif geom == 'helical':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                             [100, 100, 100], dtype='float32')

        # Geometry
        n_angle = 700
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angle)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [200, 20])
        geom = tomo.HelicalConeFlatGeometry(apart, dpart, pitch=5.0,
                                            src_radius=200, det_radius=100)

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl='astra_' + variant)
    else:
        raise ValueError('param not valid')


def test_reconstruction(projector):
    """Test discrete Ray transform using ASTRA for reconstruction."""

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
