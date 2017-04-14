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
from pkg_resources import parse_version

# Internal
import odl
import odl.tomo as tomo
from odl.util.testutils import skip_if_no_largescale, simple_fixture
from odl.tomo.util.testutils import (skip_if_no_astra, skip_if_no_astra_cuda,
                                     skip_if_no_skimage)


dtype_params = ['float32', 'float64', 'complex64']
dtype = simple_fixture('dtype', dtype_params)


# Find the valid projectors
projectors = [skip_if_no_astra('par2d astra_cpu uniform'),
              skip_if_no_astra('par2d astra_cpu nonuniform'),
              skip_if_no_astra('par2d astra_cpu random'),
              skip_if_no_astra('cone2d astra_cpu uniform'),
              skip_if_no_astra('cone2d astra_cpu nonuniform'),
              skip_if_no_astra('cone2d astra_cpu random'),
              skip_if_no_astra_cuda('par2d astra_cuda uniform'),
              skip_if_no_astra_cuda('par2d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('par2d astra_cuda random'),
              skip_if_no_astra_cuda('cone2d astra_cuda uniform'),
              skip_if_no_astra_cuda('cone2d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('cone2d astra_cuda random'),
              skip_if_no_astra_cuda('par3d astra_cuda uniform'),
              skip_if_no_astra_cuda('par3d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('par3d astra_cuda random'),
              skip_if_no_astra_cuda('cone3d astra_cuda uniform'),
              skip_if_no_astra_cuda('cone3d astra_cuda nonuniform'),
              skip_if_no_astra_cuda('cone3d astra_cuda random'),
              skip_if_no_astra_cuda('helical astra_cuda uniform'),
              skip_if_no_skimage('par2d skimage uniform')]

projector_ids = ['geom={}, impl={}, angles={}'
                 ''.format(*p.args[1].split()) for p in projectors]


# bug in pytest (ignores pytestmark) forces us to do this this
largescale = " or not pytest.config.getoption('--largescale')"
projectors = [pytest.mark.skipif(p.args[0] + largescale, p.args[1])
              for p in projectors]


weighting = simple_fixture('weighting', ['const', 'none'])


@pytest.fixture(scope="module", params=projectors, ids=projector_ids)
def projector(request, dtype, weighting):

    n_angles = 200

    geom, impl, angle = request.param.split()

    if angle == 'uniform':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
    elif angle == 'random':
        # Linearly spaced with random noise
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt, max_pt, n_angles)
        points += np.random.rand(n_angles) * (max_pt - min_pt) / (5 * n_angles)
        apart = odl.nonuniform_partition(points)
    elif angle == 'nonuniform':
        # Angles spaced quadratically
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt ** 0.5, max_pt ** 0.5, n_angles) ** 2
        apart = odl.nonuniform_partition(points)
    else:
        raise ValueError('angle not valid')

    if geom == 'par2d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20], [20, 20], [100, 100],
                                             dtype=dtype,
                                             weighting=weighting)

        # Geometry
        dpart = odl.uniform_partition(-30, 30, 200)
        geom = tomo.Parallel2dGeometry(apart, dpart)

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl=impl)

    elif geom == 'par3d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                             [100, 100, 100],
                                             dtype=dtype,
                                             weighting=weighting)

        # Geometry
        dpart = odl.uniform_partition([-30, -30], [30, 30], [200, 200])
        geom = tomo.Parallel3dAxisGeometry(apart, dpart, axis=[1, 0, 0])

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl=impl)

    elif geom == 'cone2d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20], [20, 20],
                                             [100, 100], dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition(-30, 30, 200)
        geom = tomo.FanFlatGeometry(apart, dpart,
                                    src_radius=200, det_radius=100)

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl=impl)

    elif geom == 'cone3d':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                             [100, 100, 100], dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition([-30, -30], [30, 30], [200, 200])
        geom = tomo.CircularConeFlatGeometry(
            apart, dpart, src_radius=200, det_radius=100, axis=[1, 0, 0])

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl=impl)

    elif geom == 'helical':
        # Discrete reconstruction space
        discr_reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                             [100, 100, 100], dtype=dtype)

        # Geometry
        # TODO: angles
        n_angle = 700
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angle)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [200, 20])
        geom = tomo.HelicalConeFlatGeometry(apart, dpart, pitch=5.0,
                                            src_radius=200, det_radius=100)

        # Ray transform
        return tomo.RayTransform(discr_reco_space, geom,
                                 impl=impl)
    else:
        raise ValueError('param not valid')


@skip_if_no_largescale
def test_adjoint(projector):
    """Test RayTransform adjoint matches definition."""
    # Relative tolerance, still rather high due to imperfectly matched
    # adjoint in the cone beam case
    if (parse_version(odl.tomo.ASTRA_VERSION) < parse_version('1.8rc1') and
            isinstance(projector.geometry, odl.tomo.HelicalConeFlatGeometry)):
        rtol = 0.1
    else:
        rtol = 0.05

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Calculate projection
    proj = projector(vol)
    backproj = projector.adjoint(proj)

    # Verified the identity <Ax, Ax> = <A^* A x, x>
    result_AxAx = proj.inner(proj)
    result_xAtAx = backproj.inner(vol)
    assert result_AxAx == pytest.approx(result_xAtAx, rel=rtol)


@skip_if_no_largescale
def test_adjoint_of_adjoint(projector):
    """Test RayTransform adjoint of adjoint."""

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Calculate projection
    proj = projector(vol)
    proj_adj_adj = projector.adjoint.adjoint(vol)

    # Verify A(x) == (A^*)^*(x)
    assert proj == proj_adj_adj

    # Calculate adjoints
    proj_adj = projector.adjoint(proj)
    proj_adj_adj_adj = projector.adjoint.adjoint.adjoint(proj)

    # Verify A^*(y) == ((A^*)^*)^*(x)
    assert proj_adj == proj_adj_adj_adj


@skip_if_no_largescale
def test_reconstruction(projector):
    """Test RayTransform for reconstruction."""

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Project data
    projections = projector(vol)

    # Reconstruct using ODL
    recon = projector.domain.zero()
    odl.solvers.conjugate_gradient_normal(projector, recon, projections,
                                          niter=20)

    # Make sure the result is somewhat close to the actual result.
    assert recon.dist(vol) < vol.norm() / 3.0


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--largescale'])
