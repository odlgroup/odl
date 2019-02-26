# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test reconstruction with ASTRA."""

from __future__ import division

import numpy as np
import pytest
from packaging.version import parse as parse_version

import odl
from odl.tomo.util.testutils import (
    skip_if_no_astra, skip_if_no_astra_cuda, skip_if_no_skimage)
from odl.util.testutils import simple_fixture, skip_if_no_largescale


# --- pytest fixtures --- #


pytestmark = skip_if_no_largescale


dtype_params = ['float32', 'float64', 'complex64']
dtype = simple_fixture('dtype', dtype_params)


# Find the valid projectors
projectors = []
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_astra)
     for proj_cfg in ['par2d astra_cpu uniform',
                      'par2d astra_cpu nonuniform',
                      'par2d astra_cpu random',
                      'cone2d astra_cpu uniform',
                      'cone2d astra_cpu nonuniform',
                      'cone2d astra_cpu random'])
)
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_astra_cuda)
     for proj_cfg in ['par2d astra_cuda uniform',
                      'par2d astra_cuda nonuniform',
                      'par2d astra_cuda random',
                      'cone2d astra_cuda uniform',
                      'cone2d astra_cuda nonuniform',
                      'cone2d astra_cuda random',
                      'par3d astra_cuda uniform',
                      'par3d astra_cuda nonuniform',
                      'par3d astra_cuda random',
                      'cone3d astra_cuda uniform',
                      'cone3d astra_cuda nonuniform',
                      'cone3d astra_cuda random',
                      'helical astra_cuda uniform'])
)
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_skimage)
     for proj_cfg in ['par2d skimage uniform'])
)

projector_ids = [
    " geom='{}' - impl='{}' - angles='{}' ".format(*p.values[0].split())
    for p in projectors
]


weighting = simple_fixture('weighting', [None, 1.0])


@pytest.fixture(scope="module", params=projectors, ids=projector_ids)
def projector(request, dtype, weighting):

    n_angles = 200

    geom, impl, angles = request.param.split()

    if angles == 'uniform':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
    elif angles == 'random':
        # Linearly spaced with random noise
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt, max_pt, n_angles)
        points += np.random.rand(n_angles) * (max_pt - min_pt) / (5 * n_angles)
        apart = odl.nonuniform_partition(points)
    elif angles == 'nonuniform':
        # Angles spaced quadratically
        min_pt = 2 * (2.0 * np.pi) / n_angles
        max_pt = (2.0 * np.pi) - 2 * (2.0 * np.pi) / n_angles
        points = np.linspace(min_pt ** 0.5, max_pt ** 0.5, n_angles) ** 2
        apart = odl.nonuniform_partition(points)
    else:
        assert False

    if geom == 'par2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20], [20, 20], [100, 100],
                                       dtype=dtype, weighting=weighting)

        # Geometry
        dpart = odl.uniform_partition(-30, 30, 200)
        geom = odl.tomo.Parallel2dGeometry(apart, dpart)

        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'par3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                       [100, 100, 100],
                                       dtype=dtype, weighting=weighting)

        # Geometry
        dpart = odl.uniform_partition([-30, -30], [30, 30], [200, 200])
        geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[1, 0, 0])

        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20], [20, 20], [100, 100],
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition(-30, 30, 200)
        geom = odl.tomo.FanBeamGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)

        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20, -20], [20, 20, 20],
                                       [100, 100, 100], dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition([-30, -30], [30, 30], [200, 200])
        geom = odl.tomo.ConeBeamGeometry(
            apart, dpart, src_radius=200, det_radius=100, axis=[1, 0, 0])

        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'helical':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                       [100, 100, 100], dtype=dtype)

        # Geometry
        # TODO: angles
        n_angles = 700
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [200, 20])
        geom = odl.tomo.ConeBeamGeometry(apart, dpart, pitch=5.0,
                                         src_radius=200, det_radius=100)

        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)
    else:
        raise ValueError('param not valid')


# --- RayTransform tests --- #


def test_adjoint(projector):
    """Test RayTransform adjoint matches definition."""
    # Relative tolerance, still rather high due to imperfectly matched
    # adjoint in the cone beam case
    if (
        parse_version(odl.tomo.ASTRA_VERSION) < parse_version('1.8rc1')
        and isinstance(projector.geometry, odl.tomo.ConeBeamGeometry)
    ):
        rtol = 0.1
    else:
        rtol = 0.05

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Calculate projection
    proj = projector(vol)
    backproj = projector.adjoint(proj)

    # Verify the identity <Ax, Ax> = <A^* A x, x>
    result_AxAx = proj.inner(proj)
    result_xAtAx = backproj.inner(vol)
    assert result_AxAx == pytest.approx(result_xAtAx, rel=rtol)


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


def test_reconstruction(projector):
    """Test RayTransform for reconstruction."""
    if (
        isinstance(projector.geometry, odl.tomo.ConeBeamGeometry)
        and projector.geometry.pitch != 0
    ):
        pytest.skip('reconstruction with CG is hopeless with so few angles')

    # Create Shepp-Logan phantom
    vol = odl.phantom.shepp_logan(projector.domain, modified=True)

    # Project data
    projections = projector(vol)

    # Reconstruct using ODL
    recon = projector.domain.zero()
    odl.solvers.conjugate_gradient_normal(projector, recon, projections,
                                          niter=20)

    # Make sure the result is somewhat close to the actual result
    maxerr = vol.norm() * 0.5
    if np.issubsctype(projector.domain.dtype, np.complexfloating):
        # Error has double the amount of components practically
        maxerr *= np.sqrt(2)
    assert recon.dist(vol) < maxerr


if __name__ == '__main__':
    odl.util.test_file(__file__, ['--largescale'])
