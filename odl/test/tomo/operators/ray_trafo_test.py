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
from functools import partial

import odl
from odl.tomo.backends import ASTRA_AVAILABLE, ASTRA_VERSION
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
        ASTRA_AVAILABLE
        and parse_version(ASTRA_VERSION) < parse_version('1.8rc1')
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

    # test adjoint for complex data
    backproj_r = ray_trafo_r.adjoint
    backproj_c = ray_trafo_c.adjoint
    true_vol_re = backproj_r(data.real)
    true_vol_im = backproj_r(data.imag)
    backproj_vol = backproj_c(data)

    assert all_almost_equal(backproj_vol.real, true_vol_re)
    assert all_almost_equal(backproj_vol.imag, true_vol_im)


def test_anisotropic_voxels(geometry):
    """Test projection and backprojection with anisotropic voxels."""
    ndim = geometry.ndim
    shape = [10] * (ndim - 1) + [5]
    space = odl.uniform_discr([-1] * ndim, [1] * ndim, shape=shape,
                              dtype='float32')

    # If no implementation is available, skip
    if ndim == 2 and not odl.tomo.ASTRA_AVAILABLE:
        pytest.skip(reason='ASTRA not available, skipping 2d test')
    elif ndim == 3 and not odl.tomo.ASTRA_CUDA_AVAILABLE:
        pytest.skip(reason='ASTRA_CUDA not available, skipping 3d test')

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


def test_detector_shifts_2d():
    """Check that detector shifts are handled correctly.

    We forward project a cubic phantom and check that ray transform
    and back-projection with and without detector shifts are
    numerically close (the error depends on domain discretization).
    """

    if not odl.tomo.ASTRA_AVAILABLE:
        pytest.skip(reason='ASTRA not available, skipping 2d test')

    d = 10
    space = odl.uniform_discr([-1] * 2, [1] * 2, [d] * 2)
    phantom = odl.phantom.cuboid(space, [-1 / 3] * 2, [1 / 3] * 2)

    full_angle = 2 * np.pi
    n_angles = 2 * 10
    src_rad = 2
    det_rad = 2
    apart = odl.uniform_partition(0, full_angle, n_angles)
    dpart = odl.uniform_partition(-4, 4, 8 * d)
    geom = odl.tomo.FanBeamGeometry(apart, dpart, src_rad, det_rad)
    k = 3
    shift = k * dpart.cell_sides[0]
    geom_shift = odl.tomo.FanBeamGeometry(
        apart, dpart, src_rad, det_rad,
        det_shift_func=lambda angle: [0.0, shift]
    )

    assert all_almost_equal(geom.angles, geom_shift.angles)
    angles = geom.angles
    assert all_almost_equal(geom.src_position(angles),
                            geom_shift.src_position(angles))
    assert all_almost_equal(geom.det_axis(angles),
                            geom_shift.det_axis(angles))
    assert all_almost_equal(geom.det_refpoint(angles),
                            geom_shift.det_refpoint(angles)
                            + shift * geom_shift.det_axis(angles))

    # check ray transform
    op = odl.tomo.RayTransform(space, geom)
    op_shift = odl.tomo.RayTransform(space, geom_shift)
    y = op(phantom).asarray()
    y_shift = op_shift(phantom).asarray()
    # projection on the shifted detector is shifted regular projection
    data_error = np.max(np.abs(y[:, :-k] - y_shift[:, k:]))
    assert data_error < space.cell_volume

    # check back-projection
    im = op.adjoint(y).asarray()
    im_shift = op_shift.adjoint(y_shift).asarray()
    error = np.abs(im_shift - im)
    rel_error = np.max(error[im > 0] / im[im > 0])
    assert rel_error < space.cell_volume


def test_source_shifts_2d():
    """Check that source shifts are handled correctly.

    We forward project a Shepp-Logan phantom and check that reconstruction
    with flying focal spot is equal to a sum of reconstructions with two
    geometries which mimic ffs by using initial angular offsets and
    detector shifts
    """

    if not odl.tomo.ASTRA_AVAILABLE:
        pytest.skip(reason='ASTRA required but not available')

    d = 10
    space = odl.uniform_discr([-1] * 2, [1] * 2, [d] * 2)
    phantom = odl.phantom.cuboid(space, [-1 / 3] * 2, [1 / 3] * 2)

    full_angle = 2 * np.pi
    n_angles = 2 * 10
    src_rad = 2
    det_rad = 2
    apart = odl.uniform_partition(0, full_angle, n_angles)
    dpart = odl.uniform_partition(-4, 4, 8 * d)
    # Source positions with flying focal spot should correspond to
    # source positions of 2 geometries with different starting positions
    shift1 = np.array([0.0, -0.3])
    shift2 = np.array([0.0, 0.3])
    init = np.array([1, 0], dtype=np.float32)
    det_init = np.array([0, -1], dtype=np.float32)

    ffs = partial(odl.tomo.flying_focal_spot,
                  apart=apart,
                  shifts=[shift1, shift2])
    geom_ffs = odl.tomo.FanBeamGeometry(apart, dpart,
                                        src_rad, det_rad,
                                        src_to_det_init=init,
                                        det_axis_init=det_init,
                                        src_shift_func=ffs,
                                        det_shift_func=ffs)
    # angles must be shifted to match discretization of apart
    ang1 = -full_angle / (n_angles * 2)
    apart1 = odl.uniform_partition(ang1, full_angle + ang1, n_angles // 2)
    ang2 = full_angle / (n_angles * 2)
    apart2 = odl.uniform_partition(ang2, full_angle + ang2, n_angles // 2)

    init1 = init + np.array([0, shift1[1]]) / (src_rad + shift1[0])
    init2 = init + np.array([0, shift2[1]]) / (src_rad + shift2[0])
    # radius also changes when a shift is applied
    src_rad1 = np.linalg.norm(np.array([src_rad, 0]) + shift1)
    src_rad2 = np.linalg.norm(np.array([src_rad, 0]) + shift2)
    det_rad1 = np.linalg.norm(
        np.array([det_rad, shift1[1] / src_rad * det_rad]))
    det_rad2 = np.linalg.norm(
        np.array([det_rad, shift2[1] / src_rad * det_rad]))
    geom1 = odl.tomo.FanBeamGeometry(apart1, dpart,
                                     src_rad1, det_rad1,
                                     src_to_det_init=init1,
                                     det_axis_init=det_init)
    geom2 = odl.tomo.FanBeamGeometry(apart2, dpart,
                                     src_rad2, det_rad2,
                                     src_to_det_init=init2,
                                     det_axis_init=det_init)

    # check ray transform
    op_ffs = odl.tomo.RayTransform(space, geom_ffs)
    op1 = odl.tomo.RayTransform(space, geom1)
    op2 = odl.tomo.RayTransform(space, geom2)
    y_ffs = op_ffs(phantom)
    y1 = op1(phantom).asarray()
    y2 = op2(phantom).asarray()
    assert all_almost_equal(y_ffs[::2], y1)
    assert all_almost_equal(y_ffs[1::2], y2)

    # check back-projection
    im = op_ffs.adjoint(y_ffs).asarray()
    im1 = op1.adjoint(y1).asarray()
    im2 = op2.adjoint(y2).asarray()
    im_combined = (im1 + im2) / 2
    rel_error = np.abs((im - im_combined)[im > 0] / im[im > 0])
    assert np.max(rel_error) < 1e-6


def test_detector_shifts_3d():
    """Check that detector shifts are handled correctly.

    We forward project a cubic phantom and check that ray transform
    and back-projection with and without detector shifts are
    numerically close (the error depends on domain discretization).
    """
    if not odl.tomo.ASTRA_CUDA_AVAILABLE:
        pytest.skip(reason='ASTRA CUDA required but not available')

    d = 100
    space = odl.uniform_discr([-1] * 3, [1] * 3, [d] * 3)
    phantom = odl.phantom.cuboid(space, [-1 / 3] * 3, [1 / 3] * 3)

    full_angle = 2 * np.pi
    n_angles = 2 * 100
    src_rad = 2
    det_rad = 2
    apart = odl.uniform_partition(0, full_angle, n_angles)
    dpart = odl.uniform_partition([-4] * 2, [4] * 2, [8 * d] * 2)
    geom = odl.tomo.ConeBeamGeometry(apart, dpart, src_rad, det_rad)
    k = 3
    l = 2
    shift = np.array([0, k, l]) * dpart.cell_sides[0]
    geom_shift = odl.tomo.ConeBeamGeometry(apart, dpart, src_rad, det_rad,
                                           det_shift_func=lambda angle: shift)

    angles = geom.angles

    assert all_almost_equal(angles, geom_shift.angles)
    assert all_almost_equal(geom.src_position(angles),
                            geom_shift.src_position(angles))
    assert all_almost_equal(geom.det_axes(angles),
                            geom_shift.det_axes(angles))
    assert all_almost_equal(geom.det_refpoint(angles),
                            geom_shift.det_refpoint(angles)
                            + geom_shift.det_axes(angles)[:, 0] * shift[1]
                            - geom_shift.det_axes(angles)[:, 1] * shift[2])

    # check forward pass
    op = odl.tomo.RayTransform(space, geom)
    op_shift = odl.tomo.RayTransform(space, geom_shift)
    y = op(phantom).asarray()
    y_shift = op_shift(phantom).asarray()
    data_error = np.max(np.abs(y[:, :-k, l:] - y_shift[:, k:, :-l]))
    assert data_error < 1e-3

    # check back-projection
    im = op.adjoint(y).asarray()
    im_shift = op_shift.adjoint(y_shift).asarray()
    error = np.max(np.abs(im_shift - im))
    assert error < 1e-3


def test_source_shifts_3d():
    """Check that source shifts are handled correctly.

    We forward project a Shepp-Logan phantom and check that reconstruction
    with flying focal spot is equal to a sum of reconstructions with two
    geometries which mimic ffs by using initial angular offsets and
    detector shifts
    """
    if not odl.tomo.ASTRA_CUDA_AVAILABLE:
        pytest.skip(reason='ASTRA_CUDA not available, skipping 3d test')

    d = 10
    space = odl.uniform_discr([-1] * 3, [1] * 3, [d] * 3)
    phantom = odl.phantom.cuboid(space, [-1 / 3] * 3, [1 / 3] * 3)

    full_angle = 2 * np.pi
    n_angles = 2 * 10
    apart = odl.uniform_partition(0, full_angle, n_angles)
    dpart = odl.uniform_partition([-4] * 2, [4] * 2, [8 * d] * 2)
    src_rad = 2
    det_rad = 2
    pitch = 0.2
    # Source positions with flying focal spot should correspond to
    # source positions of 2 geometries with different starting positions
    shift1 = np.array([0.0, -0.2, 0.1])
    shift2 = np.array([0.0, 0.2, -0.1])
    init = np.array([1, 0, 0], dtype=np.float32)
    det_init = np.array([[0, -1, 0], [0, 0, 1]], dtype=np.float32)
    ffs = partial(odl.tomo.flying_focal_spot,
                  apart=apart,
                  shifts=[shift1, shift2])
    geom_ffs = odl.tomo.ConeBeamGeometry(apart, dpart,
                                         src_rad, det_rad,
                                         src_to_det_init=init,
                                         det_axes_init=det_init,
                                         src_shift_func=ffs,
                                         det_shift_func=ffs,
                                         pitch=pitch)
    # angles must be shifted to match discretization of apart
    ang1 = -full_angle / (n_angles * 2)
    apart1 = odl.uniform_partition(ang1, full_angle + ang1, n_angles // 2)
    ang2 = full_angle / (n_angles * 2)
    apart2 = odl.uniform_partition(ang2, full_angle + ang2, n_angles // 2)

    init1 = init + np.array([0, shift1[1], 0]) / (src_rad + shift1[0])
    init2 = init + np.array([0, shift2[1], 0]) / (src_rad + shift2[0])
    # radius also changes when a shift is applied
    src_rad1 = np.linalg.norm(np.array([src_rad + shift1[0], shift1[1], 0]))
    src_rad2 = np.linalg.norm(np.array([src_rad + shift2[0], shift2[1], 0]))
    det_rad1 = np.linalg.norm(
        np.array([det_rad, det_rad / src_rad * shift1[1], 0]))
    det_rad2 = np.linalg.norm(
        np.array([det_rad, det_rad / src_rad * shift2[1], 0]))
    geom1 = odl.tomo.ConeBeamGeometry(apart1, dpart, src_rad1, det_rad1,
                                      src_to_det_init=init1,
                                      det_axes_init=det_init,
                                      offset_along_axis=shift1[2],
                                      pitch=pitch)
    geom2 = odl.tomo.ConeBeamGeometry(apart2, dpart, src_rad2, det_rad2,
                                      src_to_det_init=init2,
                                      det_axes_init=det_init,
                                      offset_along_axis=shift2[2],
                                      pitch=pitch)

    assert all_almost_equal(geom_ffs.src_position(geom_ffs.angles)[::2],
                            geom1.src_position(geom1.angles))
    assert all_almost_equal(geom_ffs.src_position(geom_ffs.angles)[1::2],
                            geom2.src_position(geom2.angles))

    assert all_almost_equal(geom_ffs.det_refpoint(geom_ffs.angles)[::2],
                            geom1.det_refpoint(geom1.angles))
    assert all_almost_equal(geom_ffs.det_refpoint(geom_ffs.angles)[1::2],
                            geom2.det_refpoint(geom2.angles))

    assert all_almost_equal(geom_ffs.det_axes(geom_ffs.angles)[::2],
                            geom1.det_axes(geom1.angles))
    assert all_almost_equal(geom_ffs.det_axes(geom_ffs.angles)[1::2],
                            geom2.det_axes(geom2.angles))

    op_ffs = odl.tomo.RayTransform(space, geom_ffs)
    op1 = odl.tomo.RayTransform(space, geom1)
    op2 = odl.tomo.RayTransform(space, geom2)
    y_ffs = op_ffs(phantom)
    y1 = op1(phantom)
    y2 = op2(phantom)
    assert all_almost_equal(np.mean(y_ffs[::2], axis=(1, 2)),
                            np.mean(y1, axis=(1, 2)))
    assert all_almost_equal(np.mean(y_ffs[1::2], axis=(1, 2)),
                            np.mean(y2, axis=(1, 2)))
    im = op_ffs.adjoint(y_ffs).asarray()
    im_combined = (op1.adjoint(y1).asarray() + op2.adjoint(y2).asarray())
    # the scaling is a bit off for older versions of astra
    im_combined = im_combined / np.sum(im_combined) * np.sum(im)
    rel_error = np.abs((im - im_combined)[im > 0] / im[im > 0])
    assert np.max(rel_error) < 1e-6


if __name__ == '__main__':
    odl.util.test_file(__file__)
