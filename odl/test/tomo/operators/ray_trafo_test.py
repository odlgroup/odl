# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the Ray transform."""

from __future__ import division

import numpy as np
import pytest
from packaging.version import parse as parse_version

import odl
from odl.tomo.backends import ASTRA_VERSION
from odl.tomo.util.testutils import (
    skip_if_no_astra, skip_if_no_astra_cuda, skip_if_no_skimage)
from odl.util.testutils import all_almost_equal, simple_fixture


# --- pytest fixtures --- #


impl = simple_fixture(
    name='impl',
    params=[pytest.param('astra_cpu', marks=skip_if_no_astra),
            pytest.param('astra_cuda', marks=skip_if_no_astra_cuda),
            pytest.param('skimage', marks=skip_if_no_skimage)]
)

geometry_params = ['par2d', 'par3d', 'cone2d', 'cone3d', 'helical']
geometry_ids = [" geometry='{}' ".format(p) for p in geometry_params]


@pytest.fixture(scope='module', ids=geometry_ids, params=geometry_params)
def geometry(request):
    geom = request.param
    m = 100
    n_angles = 100

    if geom == 'par2d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geom == 'par3d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -30], [30, 30], (m, m))
        return odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    elif geom == 'cone2d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition(-30, 30, m)
        return odl.tomo.FanBeamGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)
    elif geom == 'cone3d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-60, -60], [60, 60], (m, m))
        return odl.tomo.ConeBeamGeometry(apart, dpart,
                                         src_radius=200, det_radius=100)
    elif geom == 'helical':
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], (m, m))
        return odl.tomo.ConeBeamGeometry(apart, dpart, pitch=5.0,
                                         src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')


geometry_type = simple_fixture(
    'geometry_type',
    ['par2d', 'par3d', 'cone2d', 'cone3d']
)

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
                      'par2d astra_cuda half_uniform',
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
     for proj_cfg in ['par2d skimage uniform',
                      'par2d skimage half_uniform'])
)


projector_ids = [
    " geom='{}' - impl='{}' - angles='{}' ".format(*p.values[0].split())
    for p in projectors
]


@pytest.fixture(scope='module', params=projectors, ids=projector_ids)
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
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype)
        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = odl.tomo.Parallel2dGeometry(apart, dpart)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'par3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype)

        # Geometry
        dpart = odl.uniform_partition([-30] * 2, [30] * 2, [m] * 2)
        geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype)
        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = odl.tomo.FanBeamGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'cone3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype)
        # Geometry
        dpart = odl.uniform_partition([-60] * 2, [60] * 2, [m] * 2)
        geom = odl.tomo.ConeBeamGeometry(apart, dpart,
                                         src_radius=200, det_radius=100)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)

    elif geom == 'helical':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                       [n] * 3, dtype=dtype)
        # Geometry, overwriting angle partition
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [m] * 2)
        geom = odl.tomo.ConeBeamGeometry(apart, dpart, pitch=5.0,
                                         src_radius=200, det_radius=100)
        # Ray transform
        return odl.tomo.RayTransform(reco_space, geom, impl=impl)
    else:
        raise ValueError('geom not valid')


@pytest.fixture(scope='module',
                params=[True, False],
                ids=[' in-place ', ' out-of-place '])
def in_place(request):
    return request.param


# --- RayTransform tests --- #


def test_projector(projector, in_place):
    """Test Ray transform forward projection."""
    # TODO: this needs to be improved
    # Accept 10% errors
    rtol = 1e-1

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
    assert proj.ufuncs.max() == pytest.approx(expected_max, rel=rtol)


def test_adjoint(projector):
    """Test Ray transform backward projection."""
    # Relative tolerance, still rather high due to imperfectly matched
    # adjoint in the cone beam case
    if (
        parse_version(ASTRA_VERSION) < parse_version('1.8rc1')
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
        lambda x: np.exp(-(2 * x[0] - 10 + x[1]) ** 2))

    # Create projection
    result = projector(vol).asarray()

    # Find the angle where the projection has a maximum (along the line).
    # TODO: center of mass would be more robust
    axes = 1 if projector.domain.ndim == 2 else (1, 2)
    ind_angle = np.argmax(np.max(result, axis=axes))
    # Restrict to [0, 2 * pi) for helical
    maximum_angle = np.fmod(projector.geometry.angles[ind_angle], 2 * np.pi)

    # Verify correct maximum angle. The line is defined by the equation
    # x1 = 10 - 2 * x0, i.e. the slope -2. Thus the angle arctan(1/2) should
    # give the maximum projection values.
    expected = np.arctan2(1, 2)
    assert np.fmod(maximum_angle, np.pi) == pytest.approx(expected, abs=0.1)

    # Find the pixel where the projection has a maximum at that angle
    axes = () if projector.domain.ndim == 2 else 1
    ind_pixel = np.argmax(np.max(result[ind_angle], axis=axes))
    max_pixel = projector.geometry.det_partition[ind_pixel, ...].mid_pt[0]

    # The line is at distance 2 * sqrt(5) from the origin, which translates
    # to the same distance from the detector midpoint, with positive sign
    # if the angle is smaller than pi and negative sign otherwise.
    expected = 2 * np.sqrt(5) if maximum_angle < np.pi else -2 * np.sqrt(5)

    # We need to scale with the magnification factor if applicable
    if isinstance(projector.geometry, odl.tomo.DivergentBeamGeometry):
        src_to_det = (
            projector.geometry.src_radius
            + projector.geometry.det_radius
        )
        magnification = src_to_det / projector.geometry.src_radius
        expected *= magnification

    assert max_pixel == pytest.approx(expected, abs=0.2)


def test_complex(impl):
    """Test transform of complex input for parallel 2d geometry."""
    space_c = odl.uniform_discr([-1, -1], [1, 1], (10, 10), dtype='complex64')
    space_r = space_c.real_space
    geom = odl.tomo.parallel_beam_geometry(space_c)
    ray_trafo_c = odl.tomo.RayTransform(space_c, geom, impl=impl)
    ray_trafo_r = odl.tomo.RayTransform(space_r, geom, impl=impl)
    vol = odl.phantom.shepp_logan(space_c)
    vol.imag = odl.phantom.cuboid(space_r)

    data = ray_trafo_c(vol)
    true_data_re = ray_trafo_r(vol.real)
    true_data_im = ray_trafo_r(vol.imag)

    assert all_almost_equal(data.real, true_data_re)
    assert all_almost_equal(data.imag, true_data_im)


def test_anisotropic_voxels(geometry):
    """Test projection and backprojection with anisotropic voxels."""
    ndim = geometry.ndim
    shape = [10] * (ndim - 1) + [5]
    space = odl.uniform_discr([-1] * ndim, [1] * ndim, shape=shape,
                              dtype='float32')

    # If no implementation is available, skip
    if ndim == 2 and not odl.tomo.ASTRA_AVAILABLE:
        pytest.skip(msg='ASTRA not available, skipping 2d test')
    elif ndim == 3 and not odl.tomo.ASTRA_CUDA_AVAILABLE:
        pytest.skip(msg='ASTRA_CUDA not available, skipping 3d test')

    ray_trafo = odl.tomo.RayTransform(space, geometry)
    vol_one = ray_trafo.domain.one()
    data_one = ray_trafo.range.one()

    if ndim == 2:
        # Should raise
        with pytest.raises(NotImplementedError):
            ray_trafo(vol_one)
        with pytest.raises(NotImplementedError):
            ray_trafo.adjoint(data_one)
    elif ndim == 3:
        # Just check that this doesn't crash and computes something nonzero
        data = ray_trafo(vol_one)
        backproj = ray_trafo.adjoint(data_one)
        assert data.norm() > 0
        assert backproj.norm() > 0
    else:
        assert False


def test_shifted_volume(geometry_type):
    """Check that geometry shifts are handled correctly.

    We forward project a square/cube of all ones and check that the
    correct portion of the detector gets nonzero values. In the default
    setup, at angle 0, the source (if existing) is at (0, -s[, 0]), and
    the detector at (0, +d[, 0]) with the positive x axis as (first)
    detector axis. Thus, when shifting enough in the negative x direction,
    the object should be visible at the left half of the detector only.
    A shift in y should not influence the result (much).

    At +90 degrees, a shift in the negative y direction should have the same
    effect.
    """
    apart = odl.nonuniform_partition([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    if geometry_type == 'par2d' and odl.tomo.ASTRA_AVAILABLE:
        ndim = 2
        dpart = odl.uniform_partition(-30, 30, 30)
        geometry = odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geometry_type == 'par3d' and odl.tomo.ASTRA_CUDA_AVAILABLE:
        ndim = 3
        dpart = odl.uniform_partition([-30, -30], [30, 30], (30, 30))
        geometry = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    if geometry_type == 'cone2d' and odl.tomo.ASTRA_AVAILABLE:
        ndim = 2
        dpart = odl.uniform_partition(-30, 30, 30)
        geometry = odl.tomo.FanBeamGeometry(apart, dpart,
                                            src_radius=200, det_radius=100)
    elif geometry_type == 'cone3d' and odl.tomo.ASTRA_CUDA_AVAILABLE:
        ndim = 3
        dpart = odl.uniform_partition([-30, -30], [30, 30], (30, 30))
        geometry = odl.tomo.ConeBeamGeometry(apart, dpart,
                                             src_radius=200, det_radius=100)
    else:
        pytest.skip('no projector available for geometry type')

    min_pt = np.array([-5.0] * ndim)
    max_pt = np.array([5.0] * ndim)
    shift_len = 6  # enough to move the projection to one side of the detector

    # Shift along axis 0
    shift = np.zeros(ndim)
    shift[0] = -shift_len

    # Generate 4 projections with 90 degrees increment
    space = odl.uniform_discr(min_pt + shift, max_pt + shift, [10] * ndim)
    ray_trafo = odl.tomo.RayTransform(space, geometry)
    proj = ray_trafo(space.one())

    # Check that the object is projected to the correct place. With the
    # chosen setup, at least one ray should go through a substantial
    # part of the volume, yielding a value around 10 (=side length).

    # 0 degrees: All on the left
    assert np.max(proj[0, :15]) > 5
    assert np.max(proj[0, 15:]) == 0

    # 90 degrees: Left and right
    assert np.max(proj[1, :15]) > 5
    assert np.max(proj[1, 15:]) > 5

    # 180 degrees: All on the right
    assert np.max(proj[2, :15]) == 0
    assert np.max(proj[2, 15:]) > 5

    # 270 degrees: Left and right
    assert np.max(proj[3, :15]) > 5
    assert np.max(proj[3, 15:]) > 5

    # Do the same for axis 1
    shift = np.zeros(ndim)
    shift[1] = -shift_len

    space = odl.uniform_discr(min_pt + shift, max_pt + shift, [10] * ndim)
    ray_trafo = odl.tomo.RayTransform(space, geometry)
    proj = ray_trafo(space.one())

    # 0 degrees: Left and right
    assert np.max(proj[0, :15]) > 5
    assert np.max(proj[0, 15:]) > 5

    # 90 degrees: All on the left
    assert np.max(proj[1, :15]) > 5
    assert np.max(proj[1, 15:]) == 0

    # 180 degrees: Left and right
    assert np.max(proj[2, :15]) > 5
    assert np.max(proj[2, 15:]) > 5

    # 270 degrees: All on the right
    assert np.max(proj[3, :15]) == 0
    assert np.max(proj[3, 15:]) > 5


if __name__ == '__main__':
    odl.util.test_file(__file__)
