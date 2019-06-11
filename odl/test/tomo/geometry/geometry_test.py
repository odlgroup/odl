# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test ODL geometry objects for tomography."""

from __future__ import division
from itertools import permutations, product
import pytest
import numpy as np

import odl
from odl.util.testutils import all_almost_equal, all_equal, simple_fixture


# --- pytest fixtures --- #


det_pos_init_2d = simple_fixture(
    name='det_pos_init',
    params=list(set(permutations([1, 0]))
                | set(permutations([-1, 0]))
                | set(permutations([1, -1]))
                )
)
det_pos_init_3d_params = list(set(permutations([1, 0, 0]))
                              | set(permutations([-1, 0, 0]))
                              | set(permutations([1, -1, 0]))
                              | set(product([-1, 1], repeat=3)))
det_pos_init_3d = simple_fixture('det_pos_init', det_pos_init_3d_params)
axis = simple_fixture('axis', det_pos_init_3d_params)
shift = simple_fixture('shift', [0, 1])
detector_type = simple_fixture('detector_type',
                               ['flat', 'cylindrical', 'spherical'])


# --- tests --- #


def test_parallel_2d_props(shift):
    """Test basic properties of 2D parallel geometries."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    translation = np.array([shift, shift], dtype=float)
    geom = odl.tomo.Parallel2dGeometry(apart, dpart, translation=translation)

    assert geom.ndim == 2
    assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # Check defaults
    assert all_almost_equal(geom.det_pos_init, translation + [0, 1])
    assert all_almost_equal(geom.det_refpoint(0), geom.det_pos_init)
    assert all_almost_equal(geom.det_point_position(0, 0), geom.det_pos_init)
    assert all_almost_equal(geom.det_axis_init, [1, 0])
    assert all_almost_equal(geom.det_axis(0), [1, 0])
    assert all_almost_equal(geom.translation, translation)

    # Check that we first rotate, then shift along the rotated axis, which
    # is equivalent to shifting first and then rotating.
    # Here we expect to rotate the reference point to [-1, 0] and then shift
    # by 1 (=detector param) along the detector axis [0, 1] at that angle.
    # Global translation should be added afterwards.
    assert all_almost_equal(geom.det_point_position(np.pi / 2, 1),
                            translation + [-1, 1])
    assert all_almost_equal(geom.det_axis(np.pi / 2), [0, 1])

    # Detector to source vector, should be independent of the detector
    # parameter and of translation.
    # At pi/2 it should point into the (+x) direction.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, 0), [1, 0])
    assert all_almost_equal(geom.det_to_src(np.pi / 2, 1), [1, 0])

    # Rotation matrix, should correspond to counter-clockwise rotation
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2), [[0, -1],
                                                              [1, 0]])

    # Make sure that the boundary cases are treated as valid
    geom.det_point_position(0, 0)
    geom.det_point_position(full_angle, 1)

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Check that str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_parallel_2d_orientation(det_pos_init_2d):
    """Check if the orientation is positive for any ``det_pos_init``."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    det_pos_init = det_pos_init_2d
    geom = odl.tomo.Parallel2dGeometry(apart, dpart, det_pos_init=det_pos_init)

    assert all_almost_equal(geom.det_pos_init, det_pos_init)
    assert all_almost_equal(geom.det_refpoint(0), det_pos_init)
    assert all_almost_equal(geom.det_point_position(0, 0), det_pos_init)

    # The detector-to-source normal should always be perpendicular to the
    # detector axis and form a positively oriented system
    for angle in [0, np.pi / 2, 3 * np.pi / 4, np.pi]:
        dot = np.dot(geom.det_to_src(angle, 0), geom.det_axis(angle))
        assert dot == pytest.approx(0)

        normal_det_to_src = (geom.det_to_src(angle, 0)
                             / np.linalg.norm(geom.det_to_src(angle, 0)))

        orient = np.linalg.det(np.vstack([normal_det_to_src,
                                          geom.det_axis(angle)]))
        assert orient == pytest.approx(1)


def test_parallel_2d_slanted_detector():
    """Check if non-standard detector axis is handled correctly."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)

    # Detector forms a 45 degree angle with the x axis at initial position,
    # with positive direction upwards
    init_axis = [1, 1]
    geom = odl.tomo.Parallel2dGeometry(apart, dpart, det_pos_init=[0, 1],
                                       det_axis_init=init_axis)

    assert all_almost_equal(geom.det_pos_init, [0, 1])

    norm_axis = np.array(init_axis, dtype=float)
    norm_axis /= np.linalg.norm(norm_axis)
    assert all_almost_equal(geom.det_axis_init, norm_axis)

    sqrt_1_2 = np.sqrt(1.0 / 2.0)
    # At angle 0: detector position at param 1 should be
    # [0, 1] + [sqrt(1/2), sqrt(1/2)]
    assert all_almost_equal(geom.det_point_position(0, 1),
                            [sqrt_1_2, 1 + sqrt_1_2])

    # At angle pi/2: detector position at param 1 should be
    # [-1, 0] + [-sqrt(1/2), +sqrt(1/2)]
    assert all_almost_equal(geom.det_point_position(np.pi / 2, 1),
                            [-1 - sqrt_1_2, sqrt_1_2])


def test_parallel_2d_frommatrix():
    """Test the ``frommatrix`` constructor in 2d parallel geometry."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    angle = 3 * np.pi / 4
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

    # Start at [0, 1] with extra rotation by 135 degrees, making 225 degrees
    # in total for the initial position (at the bisector in the 3rd quardant)
    geom = odl.tomo.Parallel2dGeometry.frommatrix(apart, dpart, rot_matrix)

    init_pos = np.array([-1, -1], dtype=float)
    init_pos /= np.linalg.norm(init_pos)
    assert all_almost_equal(geom.det_pos_init, init_pos)

    norm_axis = np.array([-1, 1], dtype=float)
    norm_axis /= np.linalg.norm(norm_axis)
    assert all_almost_equal(geom.det_axis_init, norm_axis)

    # With translation (1, 1)
    matrix = np.hstack([rot_matrix, [[1], [1]]])
    geom = odl.tomo.Parallel2dGeometry.frommatrix(apart, dpart, matrix)

    assert all_almost_equal(geom.translation, [1, 1])

    init_pos_from_center = np.array([-1, -1], dtype=float)
    init_pos_from_center /= np.linalg.norm(init_pos_from_center)
    assert all_almost_equal(geom.det_pos_init,
                            geom.translation + init_pos_from_center)

    # Singular matrix, should raise
    sing_mat = [[1, 1],
                [1, 1]]
    with pytest.raises(np.linalg.LinAlgError):
        geom = odl.tomo.Parallel2dGeometry.frommatrix(apart, dpart, sing_mat)


def test_parallel_3d_props(shift):
    """Test basic properties of 3D parallel geometries."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    translation = np.array([shift, shift, shift], dtype=float)
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart,
                                           translation=translation)

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # Check defaults
    assert all_almost_equal(geom.axis, [0, 0, 1])
    assert all_almost_equal(geom.det_pos_init, translation + [0, 1, 0])
    assert all_almost_equal(geom.det_refpoint(0), geom.det_pos_init)
    assert all_almost_equal(geom.det_point_position(0, [0, 0]),
                            geom.det_pos_init)
    assert all_almost_equal(geom.det_axes_init, ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.det_axes(0), ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.translation, translation)

    # Check that we first rotate, then shift along the initial detector
    # axes rotated according to the angle, which is equivalent to shifting
    # first and then rotating.
    # Here we expect to rotate the reference point to [-1, 0, 0] and then
    # shift by (1, 1) (=detector param) along the detector axes
    # ([0, 1, 0], [0, 0, 1]) at that angle.
    # Global translation should come last.
    assert all_almost_equal(geom.det_point_position(np.pi / 2, [1, 1]),
                            translation + [-1, 1, 1])

    # Detector to source vector, should be independent of the detector
    # parameter. At pi/2 it should point into the (+x) direction.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, [0, 0]), [1, 0, 0])
    assert all_almost_equal(geom.det_to_src(np.pi / 2, [1, 1]), [1, 0, 0])

    # Rotation matrix, should correspond to counter-clockwise rotation
    # arond the z axis
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2),
                            [[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

    # Make sure that the boundary cases are treated as valid
    geom.det_point_position(0, [0, 0])
    geom.det_point_position(full_angle, [1, 1])

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Zero not allowed as axis
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[0, 0, 0])

    # Detector axex should not be parallel or otherwise result in a
    # linear dependent triplet
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dAxisGeometry(
            apart, dpart, det_axes_init=([0, 1, 0], [0, 1, 0]))
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dAxisGeometry(
            apart, dpart, det_axes_init=([0, 0, 0], [0, 1, 0]))

    # Check that str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_parallel_3d_orientation(axis):
    """Check if the orientation is positive for any ``axis``."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=axis)

    norm_axis = np.array(axis, dtype=float) / np.linalg.norm(axis)
    assert all_almost_equal(geom.axis, norm_axis)
    # The symmetry axis should be the second detector axis by default
    assert all_almost_equal(geom.axis, geom.det_axes_init[1])

    # The detector-to-source normal should always be perpendicular to both
    # detector axes, and the triplet (normal, det_ax0, det_ax1) should form
    # a positively oriented system
    for angle in [0, np.pi / 2, 3 * np.pi / 4, np.pi]:
        dot0 = np.dot(geom.det_to_src(angle, [0, 0]), geom.det_axes(angle)[0])
        dot1 = np.dot(geom.det_to_src(angle, [0, 0]), geom.det_axes(angle)[1])
        assert dot0 == pytest.approx(0)
        assert dot1 == pytest.approx(0)

        normal_det_to_src = (geom.det_to_src(angle, [0, 0])
                             / np.linalg.norm(geom.det_to_src(angle, [0, 0])))

        axis_0, axis_1 = geom.det_axes(angle)
        orient = np.linalg.det(np.vstack([normal_det_to_src, axis_0, axis_1]))
        assert orient == pytest.approx(1)

    # check str and repr work without crashing and return a non-empty string
    assert str(geom)
    assert repr(geom)


def test_parallel_3d_slanted_detector():
    """Check if non-standard detector axes are handled correctly."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))

    # Detector axis 0 lies in the bisector of the positive quadrant of the
    # x-y plane. Axis 1 is perpendicular to axis 0 and forms a 45 degree
    # angle with the x-y plane.
    init_axis_0 = [1, 1, 0]
    init_axis_1 = [-1, 1, 1]
    geom = odl.tomo.Parallel3dAxisGeometry(
        apart, dpart, det_axes_init=[init_axis_0, init_axis_1])

    assert all_almost_equal(geom.det_pos_init, [0, 1, 0])

    norm_axis_0 = np.array(init_axis_0, dtype=float)
    norm_axis_0 /= np.linalg.norm(norm_axis_0)
    norm_axis_1 = np.array(init_axis_1, dtype=float)
    norm_axis_1 /= np.linalg.norm(norm_axis_1)
    assert all_almost_equal(geom.det_axes_init, [norm_axis_0, norm_axis_1])

    # At angle 0: detector position at param (1, 1) should be
    # [0, 1, 0] +
    #   [sqrt(1/2), sqrt(1/2), 0] + [-sqrt(1/3), sqrt(1/3), sqrt(1/3)]
    sqrt_1_2 = np.sqrt(1.0 / 2.0)
    sqrt_1_3 = np.sqrt(1.0 / 3.0)
    true_det_pt = [sqrt_1_2 - sqrt_1_3, 1 + sqrt_1_2 + sqrt_1_3, sqrt_1_3]
    assert all_almost_equal(geom.det_point_position(0, [1, 1]), true_det_pt)

    # At angle pi/2: detector position at param (1, 1) should be
    # [-1, 0, 0] +
    #   [-sqrt(1/2), sqrt(1/2), 0] + [-sqrt(1/3), -sqrt(1/3), sqrt(1/3)]
    true_det_pt = [-1 - sqrt_1_2 - sqrt_1_3, sqrt_1_2 - sqrt_1_3, sqrt_1_3]
    assert all_almost_equal(geom.det_point_position(np.pi / 2, [1, 1]),
                            true_det_pt)

    # Check that str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_parallel_3d_frommatrix():
    """Test the ``frommatrix`` constructor in 3d parallel geometry."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    # Need a somewhat simpler rotation in 3d to not go crazy imagining the
    # rotation.
    # This one rotates by +pi/2 in phi and -pi/2 in theta, resulting in the
    # axis remapping x->y, y->-z, z->-x.
    rot_matrix = np.array([[0, 0, -1],
                           [1, 0, 0],
                           [0, -1, 0]], dtype=float)
    geom = odl.tomo.Parallel3dAxisGeometry.frommatrix(apart, dpart, rot_matrix)

    # Axis was [0, 0, 1], gets mapped to [-1, 0, 0]
    assert all_almost_equal(geom.axis, [-1, 0, 0])

    # Detector position starts at [0, 1, 0] and gets mapped to [0, 0, -1]
    assert all_almost_equal(geom.det_pos_init, [0, 0, -1])

    # Detector axes start at [1, 0, 0] and [0, 0, 1], and get mapped to
    # [0, 1, 0] and [-1, 0, 0]
    assert all_almost_equal(geom.det_axes_init, ([0, 1, 0], [-1, 0, 0]))

    # With translation (1, 1, 1)
    matrix = np.hstack([rot_matrix, [[1], [1], [1]]])
    geom = odl.tomo.Parallel3dAxisGeometry.frommatrix(apart, dpart, matrix)

    assert all_almost_equal(geom.translation, (1, 1, 1))
    assert all_almost_equal(geom.det_pos_init, geom.translation + [0, 0, -1])


def test_parallel_beam_geometry_helper():
    """Test that parallel_beam_geometry satisfies the sampling conditions.

    See the `parallel_beam_geometry` documentation for the exact conditions.
    """
    # --- 2d case ---
    space = odl.uniform_discr([-1, -1], [1, 1], [20, 20])
    geometry = odl.tomo.parallel_beam_geometry(space)

    rho = np.sqrt(2)
    omega = np.pi * 10.0

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert geometry.motion_partition.extent == pytest.approx(np.pi)
    assert geometry.motion_partition.cell_sides <= np.pi / (rho * omega)

    # Validate detector
    assert geometry.det_partition.cell_sides <= np.pi / omega
    assert geometry.det_partition.extent == pytest.approx(2 * rho)

    # --- 3d case ---

    space = odl.uniform_discr([-1, -1, 0], [1, 1, 2], [20, 20, 40])
    geometry = odl.tomo.parallel_beam_geometry(space)

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert geometry.motion_partition.extent == pytest.approx(np.pi)
    assert geometry.motion_partition.cell_sides <= np.pi / (rho * omega)

    # Validate detector
    assert geometry.det_partition.cell_sides[0] <= np.pi / omega
    assert geometry.det_partition.cell_sides[1] == pytest.approx(0.05)
    assert geometry.det_partition.extent[0] == pytest.approx(2 * rho)

    # Validate that new detector axis is correctly aligned with the rotation
    # axis and that there is one detector row per slice
    assert geometry.det_partition.min_pt[1] == pytest.approx(0.0)
    assert geometry.det_partition.max_pt[1] == pytest.approx(2.0)
    assert geometry.det_partition.shape[1] == space.shape[2]
    assert all_equal(geometry.det_axes_init[1], geometry.axis)

    # --- offset geometry ---
    space = odl.uniform_discr([0, 0], [2, 2], [20, 20])
    geometry = odl.tomo.parallel_beam_geometry(space)

    rho = np.sqrt(2) * 2
    omega = np.pi * 10.0

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert geometry.motion_partition.extent == pytest.approx(np.pi)
    assert geometry.motion_partition.cell_sides <= np.pi / (rho * omega)

    # Validate detector
    assert geometry.det_partition.cell_sides <= np.pi / omega
    assert geometry.det_partition.extent == pytest.approx(2 * rho)


def test_fanbeam_props(detector_type, shift):
    """Test basic properties of 2d fan beam geometries."""
    full_angle = 2 * np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(-np.pi / 2, np.pi / 2, 10)
    src_rad = 10
    det_rad = 5
    curve_rad = src_rad + det_rad + 1 if detector_type != "flat" else None
    translation = np.array([shift, shift], dtype=float)
    geom = odl.tomo.FanBeamGeometry(apart, dpart, src_rad, det_rad,
                                    det_curvature_radius=curve_rad,
                                    translation=translation)

    assert geom.ndim == 2
    if detector_type != 'flat':
        assert isinstance(geom.detector, odl.tomo.CircularDetector)
    else:
        assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # Check defaults
    assert all_almost_equal(geom.src_to_det_init, [0, 1])
    assert all_almost_equal(geom.src_position(0), translation + [0, -src_rad])
    assert all_almost_equal(geom.det_refpoint(0), translation + [0, det_rad])
    assert all_almost_equal(geom.det_point_position(0, 0),
                            geom.det_refpoint(0))
    assert all_almost_equal(geom.det_axis_init, [1, 0])
    assert all_almost_equal(geom.det_axis(0), [1, 0])
    assert all_almost_equal(geom.translation, translation)

    # Check that we first rotate, then shift along the rotated axis, which
    # is equivalent to shifting first and then rotating.
    # Here we expect to rotate the reference point to [-det_rad, 0] and then
    # shift by 1 (=detector param) along the detector axis [0, 1] at that
    # angle. For curved detector, we have to take curvature of into account.
    # Global translation should come afterwards.
    if detector_type != 'flat':
        det_param = np.pi / 6
        dx = curve_rad * (1 - np.cos(det_param))
        dy = curve_rad * np.sin(det_param)
        true_pos = translation + [-det_rad + dx, dy]
    else:
        det_param = 1
        true_pos = translation + [-det_rad, det_param]

    assert all_almost_equal(
        geom.det_point_position(np.pi / 2, det_param), true_pos
    )
    assert all_almost_equal(geom.det_axis(np.pi / 2), [0, 1])

    # Detector to source vector. At param=0 it should be perpendicular to
    # the detector towards the source, here at pi/2 it should point into
    # the (+x) direction.
    # At any other parameter, when adding the non-normalized vector to the
    # detector point position, one should get the source position.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, 0), [1, 0])
    src_pos = (geom.det_point_position(np.pi / 2, 1)
               + geom.det_to_src(np.pi / 2, 1, normalized=False))
    assert all_almost_equal(src_pos, geom.src_position(np.pi / 2))

    # Rotation matrix, should correspond to counter-clockwise rotation
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2), [[0, -1],
                                                              [1, 0]])

    # Make sure that the boundary cases are treated as valid
    geom.det_point_position(0, -np.pi / 2)
    geom.det_point_position(full_angle, np.pi / 2)

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Both radii zero
    with pytest.raises(ValueError):
        odl.tomo.FanBeamGeometry(apart, dpart, src_radius=0, det_radius=0)

    # Check that str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_fanbeam_frommatrix():
    """Test the ``frommatrix`` constructor in 2d fan beam geometry."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    src_rad = 10
    det_rad = 5
    angle = 3 * np.pi / 4
    rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

    # Start at [0, 1] with extra rotation by 135 degrees, making 225 degrees
    # in total for the initial position (at the bisector in the 3rd quardant)
    geom = odl.tomo.FanBeamGeometry.frommatrix(apart, dpart, src_rad, det_rad,
                                               rot_matrix)

    init_src_to_det = np.array([-1, -1], dtype=float)
    init_src_to_det /= np.linalg.norm(init_src_to_det)
    assert all_almost_equal(geom.src_to_det_init, init_src_to_det)

    norm_axis = np.array([-1, 1], dtype=float)
    norm_axis /= np.linalg.norm(norm_axis)
    assert all_almost_equal(geom.det_axis_init, norm_axis)

    # With translation (1, 1)
    matrix = np.hstack([rot_matrix, [[1], [1]]])
    geom = odl.tomo.FanBeamGeometry.frommatrix(apart, dpart, src_rad, det_rad,
                                               matrix)

    assert all_almost_equal(geom.translation, [1, 1])

    init_pos_from_center = np.array([-1, -1], dtype=float)
    init_pos_from_center /= np.linalg.norm(init_pos_from_center)
    init_pos_from_center *= det_rad
    assert all_almost_equal(geom.det_refpoint(0),
                            geom.translation + init_pos_from_center)

    # Singular matrix, should raise
    sing_mat = [[1, 1],
                [1, 1]]
    with pytest.raises(np.linalg.LinAlgError):
        geom = odl.tomo.FanBeamGeometry.frommatrix(apart, dpart, src_rad,
                                                   det_rad, sing_mat)


def test_helical_cone_beam_props(detector_type, shift):
    """Test basic properties of 3D helical cone beam geometries."""
    full_angle = 2 * np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    src_rad = 10
    det_rad = 5
    pitch = 2.0
    translation = np.array([shift, shift, shift], dtype=float)
    if detector_type == 'spherical':
        curve_rad = [src_rad + det_rad + 1] * 2
    elif detector_type == 'cylindrical':
        curve_rad = [src_rad + det_rad + 1, None]
    else:
        curve_rad = None
    geom = odl.tomo.ConeBeamGeometry(apart, dpart, src_rad, det_rad,
                                     det_curvature_radius=curve_rad,
                                     pitch=pitch, translation=translation)

    assert geom.ndim == 3
    if detector_type == 'spherical':
        assert isinstance(geom.detector, odl.tomo.SphericalDetector)
    elif detector_type == 'cylindrical':
        assert isinstance(geom.detector, odl.tomo.CylindricalDetector)
    else:
        assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # Check defaults
    assert all_almost_equal(geom.axis, [0, 0, 1])
    assert all_almost_equal(geom.src_to_det_init, [0, 1, 0])
    assert all_almost_equal(geom.src_position(0),
                            translation + [0, -src_rad, 0])
    assert all_almost_equal(geom.det_refpoint(0),
                            translation + [0, det_rad, 0])
    assert all_almost_equal(geom.det_point_position(0, [0, 0]),
                            geom.det_refpoint(0))
    assert all_almost_equal(geom.det_axes_init, ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.det_axes(0), ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.translation, translation)

    # Here we expect to rotate the reference point to [-det_rad, 0, 0] and
    # then shift by (pi/4, pi/4) (=detector param) along the detector axes
    # at that angle. In addition, everything is shifted along the rotation
    # axis [0, 0, 1] by 1/4 of the pitch
    # (since the pitch is the vertical distance after a full turn 2*pi).

    # Check that we first rotate, then shift along the initial detector
    # axes rotated according to the angle, which is equivalent to shifting
    # first and then rotating.
    # Here we expect to rotate the reference point to [-det_rad, 0, 0] and,
    # in the case of flat detector, shift by (1, 1) (=detector param)
    # along the detector axes ([0, 1, 0], [0, 0, 1]) at that angle.
    # In the case of curved detectors, curvature is taken into account.
    # In addition, everything is shifted along the rotation axis [0, 0, 1]
    # by 1/4 of the pitch (since the pitch is the vertical distance after
    # a full turn 2*pi). Global translation should come last.
    if detector_type == 'spherical':
        det_param = [np.pi / 4, np.pi / 6]
        dx = curve_rad[0] * (1 - np.cos(det_param[0]) * np.cos(det_param[1]))
        dy = curve_rad[0] * np.sin(det_param[0]) * np.cos(det_param[1])
        dz = curve_rad[0] * np.sin(det_param[1])
        det_pos = [-det_rad + dx, dy, dz]
    elif detector_type == 'cylindrical':
        det_param = [np.pi / 4, 1]
        dx = curve_rad[0] * (1 - np.cos(det_param[0]))
        dy = curve_rad[0] * np.sin(det_param[0])
        det_pos = [-det_rad + dx, dy, 1]
    else:
        det_param = [1, 1]
        det_pos = [-det_rad, 1, 1]
    z_shift = np.array([0, 0, pitch / 4])
    assert all_almost_equal(geom.det_point_position(np.pi / 2, det_param),
                            translation + det_pos + z_shift)

    # Make sure that source and detector move at the same height and stay
    # opposite of each other
    src_to_det_ref = (geom.det_refpoint(np.pi / 2)
                      - geom.src_position(np.pi / 2))
    assert np.dot(geom.axis, src_to_det_ref) == pytest.approx(0)
    assert np.linalg.norm(src_to_det_ref) == pytest.approx(src_rad + det_rad)

    # Detector to source vector. At param=0 it should be perpendicular to
    # the detector towards the source, here at pi/2 it should point into
    # the (+x) direction.
    # At any other parameter, when adding the non-normalized vector to the
    # detector point position, one should get the source position.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, [0, 0]), [1, 0, 0])
    src_pos = (geom.det_point_position(np.pi / 2, det_param)
               + geom.det_to_src(np.pi / 2, det_param, normalized=False))
    assert all_almost_equal(src_pos, geom.src_position(np.pi / 2))

    # Rotation matrix, should correspond to counter-clockwise rotation
    # arond the z axis
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2),
                            [[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

    # offset_along_axis
    geom = odl.tomo.ConeBeamGeometry(apart, dpart, src_rad, det_rad,
                                     pitch=pitch, offset_along_axis=0.5)
    assert all_almost_equal(geom.det_refpoint(0), [0, det_rad, 0.5])

    # Make sure that the boundary cases are treated as valid
    geom.det_point_position(0, [0, 0])
    geom.det_point_position(full_angle, [1, 1])

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Zero not allowed as axis
    with pytest.raises(ValueError):
        odl.tomo.ConeBeamGeometry(apart, dpart, src_rad, det_rad,
                                  pitch=pitch, axis=[0, 0, 0])

    # Detector axex should not be parallel or otherwise result in a
    # linear dependent triplet
    with pytest.raises(ValueError):
        odl.tomo.ConeBeamGeometry(
            apart, dpart, src_rad, det_rad, pitch=pitch,
            det_axes_init=([0, 1, 0], [0, 1, 0]))
    with pytest.raises(ValueError):
        odl.tomo.ConeBeamGeometry(
            apart, dpart, src_rad, det_rad, pitch=pitch,
            det_axes_init=([0, 0, 0], [0, 1, 0]))

    # Both radii zero
    with pytest.raises(ValueError):
        odl.tomo.ConeBeamGeometry(apart, dpart, src_radius=0, det_radius=0,
                                  pitch=pitch)

    # Check that str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_cone_beam_slanted_detector():
    """Check if non-standard detector axes are handled correctly."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([-np.pi / 2, 0], [np.pi / 2, 1], (10, 10))

    # Detector axis 0 lies in the bisector of the positive quadrant of the
    # x-z plane. Axis 1 is perpendicular to axis 0 and forms a 45 degree
    # angle with the x-y plane.
    init_axis_0 = [1, 0, 1]
    init_axis_1 = [-1, 0, 1]
    geom = odl.tomo.ConeBeamGeometry(apart, dpart,
                                     src_radius=1, det_radius=1,
                                     det_curvature_radius=(1, None),
                                     det_axes_init=[init_axis_0, init_axis_1])

    assert all_almost_equal(geom.det_refpoint(0), [0, 1, 0])

    norm_axis_0 = np.array(init_axis_0, dtype=float)
    norm_axis_0 /= np.linalg.norm(norm_axis_0)
    norm_axis_1 = np.array(init_axis_1, dtype=float)
    norm_axis_1 /= np.linalg.norm(norm_axis_1)
    assert all_almost_equal(geom.det_axes_init, [norm_axis_0, norm_axis_1])

    # At angle 0 detector position at param (ang, h) should be
    # [0, 1, 0] +
    #   h * axis_1 + sin(ang) * axis_0 + (1 - cos(ang)) * normal_axis
    ang = np.pi / 3
    h = 0.3
    normal_axis = np.array([0, -1, 0])
    true_det_pt = (np.array([0, 1, 0])
                   + h * norm_axis_1
                   + np.sin(ang) * norm_axis_0
                   + (1 - np.cos(ang)) * normal_axis)
    assert all_almost_equal(geom.det_point_position(0, [ang, h]),
                            true_det_pt)

    # axes are not perpendicular
    with pytest.raises(ValueError):
        odl.tomo.ConeBeamGeometry(apart, dpart,
                                  src_radius=5, det_radius=10,
                                  det_curvature_radius=(1, None),
                                  det_axes_init=[init_axis_0, [-2, 0, 1]])

    # Check that str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_cone_beam_geometry_helper():
    """Test that cone_beam_geometry satisfies the sampling conditions.

    See the `cone_beam_geometry` documentation for the exact conditions.
    """
    # --- 2d case ---
    space = odl.uniform_discr([-1, -1], [1, 1], [20, 20])
    src_radius = 3
    det_radius = 9
    magnification = (src_radius + det_radius) / src_radius
    geometry = odl.tomo.cone_beam_geometry(space, src_radius, det_radius)

    rho = np.sqrt(2)
    omega = np.pi * 10.0
    r = src_radius + det_radius

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert geometry.motion_partition.extent == pytest.approx(2 * np.pi)
    assert (geometry.motion_partition.cell_sides
            <= (r + rho) / r * np.pi / (rho * omega))

    # Validate detector
    det_width = 2 * magnification * rho
    R = np.hypot(r, det_width / 2)
    assert geometry.det_partition.cell_sides <= np.pi * R / (r * omega)
    assert geometry.det_partition.extent == pytest.approx(det_width)

    # Short scan option
    fan_angle = 2 * np.arctan(det_width / (2 * r))
    geometry = odl.tomo.cone_beam_geometry(space, src_radius, det_radius,
                                           short_scan=True)
    assert geometry.motion_params.extent == pytest.approx(np.pi + fan_angle)

    # --- 3d case ---

    space = odl.uniform_discr([-1, -1, 0], [1, 1, 2], [20, 20, 40])
    geometry = odl.tomo.cone_beam_geometry(space, src_radius, det_radius)

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert geometry.motion_partition.extent == pytest.approx(2 * np.pi)
    assert (geometry.motion_partition.cell_sides
            <= (r + rho) / r * np.pi / (rho * omega))

    # Validate detector
    assert geometry.det_partition.cell_sides[0] <= np.pi * R / (r * omega)
    half_cone_angle = np.arctan(2 / (3 - rho))
    det_height = 2 * np.sin(half_cone_angle) * (3 + 9)
    mag = (3 + 9) / (3 + rho)
    delta_h = space.cell_sides[2] * mag
    assert geometry.det_partition.cell_sides[1] == pytest.approx(delta_h)
    assert np.all(geometry.det_partition.extent >= [det_width, det_height])

    # --- offset geometry (2d) ---

    space = odl.uniform_discr([0, 0], [2, 2], [20, 20])
    geometry = odl.tomo.cone_beam_geometry(space, src_radius, det_radius)

    rho = np.sqrt(2) * 2
    omega = np.pi * 10.0

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert geometry.motion_partition.extent == pytest.approx(2 * np.pi)
    assert (geometry.motion_partition.cell_sides
            <= (r + rho) / r * np.pi / (rho * omega))

    # Validate detector
    det_width = 2 * magnification * rho
    R = np.hypot(r, det_width / 2)
    assert geometry.det_partition.cell_sides <= np.pi * R / (r * omega)
    assert geometry.det_partition.extent == pytest.approx(det_width)


def test_helical_geometry_helper():
    """Test that helical_geometry satisfies the sampling conditions.

    See the `helical_geometry` documentation for the exact conditions.
    """
    # Parameters
    src_radius = 3
    det_radius = 9
    num_turns = 4

    magnification = (src_radius + det_radius) / src_radius
    rho = np.sqrt(2)
    omega = np.pi * 10.0
    r = src_radius + det_radius

    # Create object
    space = odl.uniform_discr([-1, -1, -2], [1, 1, 2], [20, 20, 40])
    geometry = odl.tomo.helical_geometry(space, src_radius, det_radius,
                                         num_turns=num_turns)

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert (geometry.motion_partition.extent
            == pytest.approx(num_turns * 2 * np.pi))
    assert (geometry.motion_partition.cell_sides
            <= (r + rho) / r * np.pi / (rho * omega))

    # Validate detector
    det_width = 2 * magnification * rho
    R = np.hypot(r, det_width / 2)
    assert geometry.det_partition.cell_sides[0] <= np.pi * R / (r * omega)
    mag = (3 + 9) / (3 + rho)
    delta_h = space.cell_sides[2] * mag
    assert geometry.det_partition.cell_sides[1] <= delta_h


if __name__ == '__main__':
    odl.util.test_file(__file__)
