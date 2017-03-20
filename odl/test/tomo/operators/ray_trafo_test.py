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

"""Tests for the Ray transform."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
from pkg_resources import parse_version
import pytest

import odl
import odl.tomo as tomo
from odl.tomo.backends import ASTRA_VERSION
from odl.tomo.util.testutils import (skip_if_no_astra, skip_if_no_astra_cuda,
                                     skip_if_no_scikit)
from odl.util.testutils import almost_equal, all_almost_equal


# Find the valid projectors
projectors = [skip_if_no_astra('par2d astra_cpu uniform'),
              skip_if_no_astra('par2d astra_cpu nonuniform'),
              skip_if_no_astra('par2d astra_cpu random'),
              skip_if_no_astra('cone2d astra_cpu uniform'),
              skip_if_no_astra('cone2d astra_cpu nonuniform'),
              skip_if_no_astra('cone2d astra_cpu random'),
              skip_if_no_astra_cuda('par2d astra_cuda uniform'),
              skip_if_no_astra_cuda('par2d astra_cuda half_uniform'),
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
              skip_if_no_scikit('par2d scikit uniform'),
              skip_if_no_scikit('par2d scikit half_uniform')]


projector_ids = ['geom={}, impl={}, angles={}'
                 ''.format(*p.args[1].split()) for p in projectors]


@pytest.fixture(scope="module", params=projectors, ids=projector_ids)
def projector(request):
    n = 100
    m = 100
    n_angles = 100
    dtype = 'float32'

    geom, impl, angle = request.param.split()

    if angle == 'uniform':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
    elif angle == 'half_uniform':
        apart = odl.uniform_partition(0, np.pi, n_angles)
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
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = tomo.Parallel2dGeometry(apart, dpart)

        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'par3d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition([-30] * 2, [30] * 2, [n] * 2)
        geom = tomo.Parallel3dAxisGeometry(apart, dpart)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone2d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = tomo.FanFlatGeometry(apart, dpart, src_radius=200,
                                    det_radius=100)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone3d':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition([-60] * 2, [60] * 2, [m] * 2)

        geom = tomo.CircularConeFlatGeometry(apart, dpart, src_radius=200,
                                             det_radius=100)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'helical':
        # Discrete reconstruction space
        reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                       [n] * 3, dtype=dtype)

        # overwrite angle
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [m] * 2)
        geom = tomo.HelicalConeFlatGeometry(apart, dpart, pitch=5.0,
                                            src_radius=200, det_radius=100)

        # Ray transform
        return tomo.RayTransform(reco_space, geom, impl=impl)
    else:
        raise ValueError('geom not valid')


@pytest.fixture(scope="module",
                params=[True, False],
                ids=[' in-place ', ' out-of-place '])
def in_place(request):
    return request.param


def test_projector(projector, in_place):
    """Test Ray transform forward projection."""

    # TODO: this needs to be improved
    # Accept 10% errors
    places = 1

    # Create Shepp-Logan phantom
    vol = projector.domain.one()

    # Calculate projection
    if in_place:
        proj = projector.range.zero()
        projector(vol, out=proj)
    else:
        proj = projector(vol)

    # We expect maximum value to be along diagonal
    expected_max = projector.domain.partition.extent[0] * np.sqrt(2)
    assert almost_equal(proj.ufuncs.max(), expected_max, places=places)


def test_adjoint(projector):
    """Test Ray transform backward projection."""
    # Relative tolerance, still rather high due to imperfectly matched
    # adjoint in the cone beam case
    if (parse_version(ASTRA_VERSION) < parse_version('1.8rc1') and
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


def test_adjoint_of_adjoint(projector):
    """Test Ray transform adjoint of adjoint."""

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Calculate projection
    proj = projector(vol)
    proj_adj_adj = projector.adjoint.adjoint(vol)

    # Verify A(x) == (A^*)^*(x)
    assert all_almost_equal(proj, proj_adj_adj)

    # Calculate adjoints
    proj_adj = projector.adjoint(proj)
    proj_adj_adj_adj = projector.adjoint.adjoint.adjoint(proj)

    # Verify A^*(y) == ((A^*)^*)^*(x)
    assert all_almost_equal(proj_adj, proj_adj_adj_adj)


def test_angles(projector):
    """Test Ray transform angle conventions."""

    # Smoothed line/hyperplane with offset
    vol = projector.domain.element(
        lambda x: np.exp(-(x[0] + 2 * x[1] - 10) ** 2))

    # Create projection
    result = projector(vol).asarray()

    # Find the angle where the projection has a maximum (along the line)
    axes = 1 if projector.domain.ndim == 2 else (1, 2)
    ind_angle = np.argmax(np.max(result, axis=axes))
    maximum_angle = projector.geometry.angles[ind_angle]

    # Verify correct maximum angle
    expected = np.arctan2(1, -2)
    assert almost_equal(np.fmod(maximum_angle, np.pi), expected, places=1)

    # Find the pixel where the projection has a maximum at that angle
    axes = () if projector.domain.ndim == 2 else 1
    ind_pixel = np.argmax(np.max(result[ind_angle], axis=axes))
    max_pixel = projector.geometry.det_partition[ind_pixel, ...].mid_pt[0]

    # This is hard to measure, so mostly check it is on the correct side, i.e.
    # geometry is not flipped.
    expected = 10 * np.sqrt(1.0/3)
    assert np.abs(max_pixel - expected) < expected // 2


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
