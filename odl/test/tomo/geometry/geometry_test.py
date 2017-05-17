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


det_pos_init_2d_params = list(set(permutations((1, 0))) |
                              set(permutations((-1, 0))) |
                              set(permutations((1, -1))))
det_pos_init_2d = simple_fixture('det_pos_init', det_pos_init_2d_params)

det_pos_init_3d_params = list(set(permutations((1, 0, 0))) |
                              set(permutations((-1, 0, 0))) |
                              set(permutations((1, -1, 0))) |
                              set(product((-1, 1), repeat=3)))
det_pos_init_3d = simple_fixture('det_pos_init', det_pos_init_3d_params)
axis = simple_fixture('axis', det_pos_init_3d_params)


# --- helpers --- #


def rotate(vector, axis, angle):
    """Rotate ``vector`` about ``axis`` by ``angle`` (right-handed)."""
    axis = np.asarray(axis) / np.linalg.norm(axis)
    return (np.cos(angle) * vector +
            np.sin(angle) * np.cross(axis, vector) +
            (1 - np.cos(angle)) * np.dot(axis, vector) * axis)


# --- tests --- #


def test_parallel_2d_props():
    """Test basic properties of 2D parallel geometries."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    geom = odl.tomo.Parallel2dGeometry(apart, dpart)

    assert geom.ndim == 2
    assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # Check defaults
    assert all_almost_equal(geom.det_pos_init, [0, 1])
    assert all_almost_equal(geom.det_refpoint(0), [0, 1])
    assert all_almost_equal(geom.det_point_position(0, 0), [0, 1])
    assert all_almost_equal(geom.det_axis_init, [1, 0])
    assert all_almost_equal(geom.det_axis(0), [1, 0])

    # Check that we first rotate, then shift along the rotated axis, which
    # is equivalent to shifting first and then rotating.
    # Here we expect to rotate the reference point to [-1, 0] and then shift
    # by 1 (=detector param) along the detector axis [0, 1] at that angle.
    assert all_almost_equal(geom.det_point_position(np.pi / 2, 1), [-1, 1])
    assert all_almost_equal(geom.det_axis(np.pi / 2), [0, 1])

    # Detector to source vector, should be independent of the detector
    # parameter. At pi/2 it should point into the (+x) direction.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, 0), [1, 0])
    assert all_almost_equal(geom.det_to_src(np.pi / 2, 1), [1, 0])

    # Rotation matrix, should correspond to counter-clockwise rotation
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2), [[0, -1],
                                                              [1, 0]])

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Singular rotation matrix
    extra_rot = [[1, 1],
                 [1, 1]]
    with pytest.raises(ValueError):
        odl.tomo.Parallel2dGeometry(apart, dpart, extra_rot=extra_rot)

    # check str and repr work without crashing and return a non-empty string
    assert str(geom) > ''
    assert repr(geom) > ''


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

        normal_det_to_src = (geom.det_to_src(angle, 0) /
                             np.linalg.norm(geom.det_to_src(angle, 0)))

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


def test_parallel_2d_extra_rot():
    """Test the ``extra_rotation`` parameter in 2d parallel geometry."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    angle = 3 * np.pi / 4
    extra_rot = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

    # Start at [0, 1] with extra rotation by 135 degrees, making 225 degrees
    # in total for the initial position (at the bisector in the 3rd quardant)
    geom = odl.tomo.Parallel2dGeometry(apart, dpart, extra_rot=extra_rot)

    init_pos = np.array([-1, -1], dtype=float)
    init_pos /= np.linalg.norm(init_pos)
    assert all_almost_equal(geom.det_pos_init, init_pos)

    norm_axis = np.array([-1, 1], dtype=float)
    norm_axis /= np.linalg.norm(norm_axis)
    assert all_almost_equal(geom.det_axis_init, norm_axis)


def test_parallel_3d_props():
    """Test basic properties of 3D parallel geometries."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # Check defaults
    assert all_almost_equal(geom.axis, [0, 0, 1])
    assert all_almost_equal(geom.det_pos_init, [0, 1, 0])
    assert all_almost_equal(geom.det_refpoint(0), [0, 1, 0])
    assert all_almost_equal(geom.det_point_position(0, [0, 0]), [0, 1, 0])
    assert all_almost_equal(geom.det_axes_init, ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.det_axes(0), ([1, 0, 0], [0, 0, 1]))

    # Check that we first rotate, then shift along the initial detector
    # axes rotated according to the angle, which is equivalent to shifting
    # first and then rotating.
    # Here we expect to rotate the reference point to [-1, 0, 0] and then
    # shift by (1, 1) (=detector param) along the detector axes
    # ([0, 1, 0], [0, 0, 1]) at that angle.
    assert all_almost_equal(geom.det_point_position(np.pi / 2, [1, 1]),
                            [-1, 1, 1])

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

    # check str and repr work without crashing and return a non-empty string
    assert str(geom) > ''
    assert repr(geom) > ''


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

        normal_det_to_src = (geom.det_to_src(angle, [0, 0]) /
                             np.linalg.norm(geom.det_to_src(angle, [0, 0])))

        axis_0, axis_1 = geom.det_axes(angle)
        orient = np.linalg.det(np.vstack([normal_det_to_src, axis_0, axis_1]))
        assert orient == pytest.approx(1)


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


def test_parallel_3d_extra_rot():
    """Test the ``extra_rotation`` parameter in 3d parallel geometry."""
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    # Need a somewhat simpler rotation in 3d to not go crazy imagining the
    # rotation.
    # This one rotates by +pi/2 in phi and -pi/2 in theta, resulting in the
    # axis remapping x->y, y->-z, z->-x.
    extra_rot = np.array([[0, 0, -1],
                          [1, 0, 0],
                          [0, -1, 0]], dtype=float)

    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, extra_rot=extra_rot)

    # Axis was [0, 0, 1], gets mapped to [-1, 0, 0]
    assert all_almost_equal(geom.axis, [-1, 0, 0])

    # Detector position starts at [0, 1, 0] and gets mapped to [0, 0, -1]
    assert all_almost_equal(geom.det_pos_init, [0, 0, -1])

    # Detector axes start at [1, 0, 0] and [0, 0, 1], and get mapped to
    # [0, 1, 0] and [-1, 0, 0]
    assert all_almost_equal(geom.det_axes_init, ([0, 1, 0], [-1, 0, 0]))


def test_parallel_beam_geometry_helper():
    """Test that parallel_beam_geometry satisfies the sampling conditions."""
    # --- 2d case ---
    space = odl.uniform_discr([-1, -1], [1, 1], [20, 20])
    geometry = odl.tomo.parallel_beam_geometry(space)

    rho = np.sqrt(2)
    omega = np.pi * 10.0

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert pytest.approx(geometry.motion_partition.extent, np.pi)
    assert geometry.motion_partition.cell_sides <= np.pi / (rho * omega)

    # Validate detector
    assert geometry.det_partition.cell_sides <= np.pi / omega
    assert pytest.approx(geometry.det_partition.extent, 2 * rho)

    # --- 3d case ---

    space = odl.uniform_discr([-1, -1, 0], [1, 1, 2], [20, 20, 40])
    geometry = odl.tomo.parallel_beam_geometry(space)

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert pytest.approx(geometry.motion_partition.extent, np.pi)
    assert geometry.motion_partition.cell_sides <= np.pi / (rho * omega)

    # Validate detector
    assert geometry.det_partition.cell_sides[0] <= np.pi / omega
    assert pytest.approx(geometry.det_partition.cell_sides[0], 0.05)
    assert pytest.approx(geometry.det_partition.extent[0], 2 * rho)

    # Validate that new detector axis is correctly aligned with the rotation
    # axis and that there is one detector row per slice
    assert pytest.approx(geometry.det_partition.min_pt[1], 0.0)
    assert pytest.approx(geometry.det_partition.max_pt[1], 2.0)
    assert geometry.det_partition.shape[1] == space.shape[2]
    assert all_equal(geometry.det_axes_init[1], geometry.axis)

    # --- offset geometry ---
    space = odl.uniform_discr([0, 0], [2, 2], [20, 20])
    geometry = odl.tomo.parallel_beam_geometry(space)

    rho = np.sqrt(2) * 2
    omega = np.pi * 10.0

    # Validate angles
    assert geometry.motion_partition.is_uniform
    assert pytest.approx(geometry.motion_partition.extent, np.pi)
    assert geometry.motion_partition.cell_sides <= np.pi / (rho * omega)

    # Validate detector
    assert geometry.det_partition.cell_sides <= np.pi / omega
    assert pytest.approx(geometry.det_partition.extent, 2 * rho)


def test_fanflat_props():
    """Test basic properties of 2d fanflat geometries."""
    full_angle = 2 * np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    src_rad = 10
    det_rad = 5
    geom = odl.tomo.FanFlatGeometry(apart, dpart, src_rad, det_rad)

    assert geom.ndim == 2
    assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # Check defaults
    assert all_almost_equal(geom.src_to_det_init, [0, 1])
    assert all_almost_equal(geom.src_position(0), [0, -src_rad])
    assert all_almost_equal(geom.det_refpoint(0), [0, det_rad])
    assert all_almost_equal(geom.det_point_position(0, 0), [0, det_rad])
    assert all_almost_equal(geom.det_axis_init, [1, 0])
    assert all_almost_equal(geom.det_axis(0), [1, 0])

    # Check that we first rotate, then shift along the rotated axis, which
    # is equivalent to shifting first and then rotating.
    # Here we expect to rotate the reference point to [-det_rad, 0] and then
    # shift by 1 (=detector param) along the detector axis [0, 1] at that
    # angle.
    assert all_almost_equal(geom.det_point_position(np.pi / 2, 1),
                            [-det_rad, 1])
    assert all_almost_equal(geom.det_axis(np.pi / 2), [0, 1])

    # Detector to source vector. At param=0 it should be perpendicular to
    # the detector towards the source, here at pi/2 it should point into
    # the (+x) direction.
    # At any other parameter, when adding the non-normalized vector to the
    # detector point position, one should get the source position.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, 0), [1, 0])
    src_pos = (geom.det_point_position(np.pi / 2, 1) +
               geom.det_to_src(np.pi / 2, 1, normalized=False))
    assert all_almost_equal(src_pos, geom.src_position(np.pi / 2))

    # Rotation matrix, should correspond to counter-clockwise rotation
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2), [[0, -1],
                                                              [1, 0]])

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Both radii zero
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(apart, dpart, src_radius=0, det_radius=0)

    # Singular rotation matrix
    extra_rot = [[1, 1],
                 [1, 1]]
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(apart, dpart, src_rad, det_rad,
                                 extra_rot=extra_rot)

    # check str and repr work without crashing and return a non-empty string
    assert str(geom) > ''
    assert repr(geom) > ''


def test_circular_cone_flat_props():
    """Test basic properties of 3D circular cone beam geometries."""
    full_angle = 2 * np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    src_rad = 10
    det_rad = 5
    geom = odl.tomo.CircularConeFlatGeometry(apart, dpart, src_rad, det_rad)

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # Check defaults
    assert all_almost_equal(geom.axis, [0, 0, 1])
    assert all_almost_equal(geom.src_to_det_init, [0, 1, 0])
    assert all_almost_equal(geom.src_position(0), [0, -src_rad, 0])
    assert all_almost_equal(geom.det_refpoint(0), [0, det_rad, 0])
    assert all_almost_equal(geom.det_point_position(0, [0, 0]),
                            [0, det_rad, 0])
    assert all_almost_equal(geom.det_axes_init, ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.det_axes(0), ([1, 0, 0], [0, 0, 1]))

    # Check that we first rotate, then shift along the initial detector
    # axes rotated according to the angle, which is equivalent to shifting
    # first and then rotating.
    # Here we expect to rotate the reference point to [-det_rad, 0, 0] and
    # then shift by (1, 1) (=detector param) along the detector axes
    # ([0, 1, 0], [0, 0, 1]) at that angle.
    assert all_almost_equal(geom.det_point_position(np.pi / 2, [1, 1]),
                            [-det_rad, 1, 1])

    # Detector to source vector. At param=0 it should be perpendicular to
    # the detector towards the source, here at pi/2 it should point into
    # the (+x) direction.
    # At any other parameter, when adding the non-normalized vector to the
    # detector point position, one should get the source position.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, [0, 0]), [1, 0, 0])
    src_pos = (geom.det_point_position(np.pi / 2, [1, 1]) +
               geom.det_to_src(np.pi / 2, [1, 1], normalized=False))
    assert all_almost_equal(src_pos, geom.src_position(np.pi / 2))

    # Rotation matrix, should correspond to counter-clockwise rotation
    # arond the z axis
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2),
                            [[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Zero not allowed as axis
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(apart, dpart, src_rad, det_rad,
                                          axis=[0, 0, 0])

    # Detector axex should not be parallel or otherwise result in a
    # linear dependent triplet
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(
            apart, dpart, src_rad, det_rad,
            det_axes_init=([0, 1, 0], [0, 1, 0]))
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(
            apart, dpart, src_rad, det_rad,
            det_axes_init=([0, 0, 0], [0, 1, 0]))

    # Both radii zero
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(apart, dpart, src_radius=0,
                                          det_radius=0)

    # check str and repr work without crashing and return a non-empty string
    assert str(geom) > ''
    assert repr(geom) > ''


def test_helical_cone_flat_props():
    """Test basic properties of 3D helical cone beam geometries."""
    full_angle = 2 * np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], (10, 10))
    src_rad = 10
    det_rad = 5
    pitch = 2.0
    geom = odl.tomo.HelicalConeFlatGeometry(apart, dpart, src_rad, det_rad,
                                            pitch)

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # Check defaults
    assert all_almost_equal(geom.axis, [0, 0, 1])
    assert all_almost_equal(geom.src_to_det_init, [0, 1, 0])
    assert all_almost_equal(geom.src_position(0), [0, -src_rad, 0])
    assert all_almost_equal(geom.det_refpoint(0), [0, det_rad, 0])
    assert all_almost_equal(geom.det_point_position(0, [0, 0]),
                            [0, det_rad, 0])
    assert all_almost_equal(geom.det_axes_init, ([1, 0, 0], [0, 0, 1]))
    assert all_almost_equal(geom.det_axes(0), ([1, 0, 0], [0, 0, 1]))

    # Check that we first rotate, then shift along the initial detector
    # axes rotated according to the angle, which is equivalent to shifting
    # first and then rotating.
    # Here we expect to rotate the reference point to [-det_rad, 0, 0] and
    # then shift by (1, 1) (=detector param) along the detector axes
    # ([0, 1, 0], [0, 0, 1]) at that angle. In addition, everything is
    # shifted along the rotation axis [0, 0, 1] by 1/4 of the pitch
    # (since the pitch is the vertical distance after a full turn 2*pi).
    assert all_almost_equal(geom.det_point_position(np.pi / 2, [1, 1]),
                            [-det_rad, 1, 1 + pitch / 4])

    # Make sure that source and detector move at the same height and stay
    # opposite of each other
    src_to_det_ref = (geom.det_refpoint(np.pi / 2) -
                      geom.src_position(np.pi / 2))
    assert np.dot(geom.axis, src_to_det_ref) == pytest.approx(0)
    assert np.linalg.norm(src_to_det_ref) == pytest.approx(src_rad + det_rad)

    # Detector to source vector. At param=0 it should be perpendicular to
    # the detector towards the source, here at pi/2 it should point into
    # the (+x) direction.
    # At any other parameter, when adding the non-normalized vector to the
    # detector point position, one should get the source position.
    assert all_almost_equal(geom.det_to_src(np.pi / 2, [0, 0]), [1, 0, 0])
    src_pos = (geom.det_point_position(np.pi / 2, [1, 1]) +
               geom.det_to_src(np.pi / 2, [1, 1], normalized=False))
    assert all_almost_equal(src_pos, geom.src_position(np.pi / 2))

    # Rotation matrix, should correspond to counter-clockwise rotation
    # arond the z axis
    assert all_almost_equal(geom.rotation_matrix(np.pi / 2),
                            [[0, -1, 0],
                             [1, 0, 0],
                             [0, 0, 1]])

    # pitch_offset
    geom = odl.tomo.HelicalConeFlatGeometry(apart, dpart, src_rad, det_rad,
                                            pitch, pitch_offset=0.5)
    assert all_almost_equal(geom.det_refpoint(0), [0, det_rad, 0.5])

    # Invalid parameter
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # Zero not allowed as axis
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(apart, dpart, src_rad, det_rad,
                                         pitch, axis=[0, 0, 0])

    # Detector axex should not be parallel or otherwise result in a
    # linear dependent triplet
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(
            apart, dpart, src_rad, det_rad, pitch,
            det_axes_init=([0, 1, 0], [0, 1, 0]))
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(
            apart, dpart, src_rad, det_rad, pitch,
            det_axes_init=([0, 0, 0], [0, 1, 0]))

    # Both radii zero
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(apart, dpart, src_radius=0,
                                         det_radius=0, pitch=pitch)

    # check str and repr work without crashing and return a non-empty string
    assert str(geom) > ''
    assert repr(geom) > ''


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
