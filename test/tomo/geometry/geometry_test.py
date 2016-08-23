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

"""Test ODL geometry objects for tomography."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
from odl.util.testutils import almost_equal, all_almost_equal, all_equal


# TODO: test for rotations about arbitrary axis

def test_parallel_2d_geometry():
    """General parallel 2D geometries."""

    # Parameters
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    geom = odl.tomo.Parallel2dGeometry(apart, dpart)

    assert geom.ndim == 2
    assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # detector rotation
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_almost_equal(rot_mat.dot([1, 0]), [0, 1])
    assert all_almost_equal(rot_mat.dot([0, 1]), [-1, 0])

    # check str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_parallel_3d_single_axis_geometry():
    """General parallel 3D geometries."""

    # Parameters
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], [10, 10])

    # Bad init
    with pytest.raises(TypeError):
        odl.tomo.Parallel3dAxisGeometry([0, 1], dpart)
    with pytest.raises(TypeError):
        odl.tomo.Parallel3dAxisGeometry(apart, [0, 1])

    # Initialize
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[0, 0, 1])

    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # rotation of cartesian basis vectors about each other
    coords = np.eye(3)

    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[1, 0, 0])
    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_almost_equal(rot_mat.dot(coords), [[1, 0, 0],
                                                  [0, 0, -1],
                                                  [0, 1, 0]])

    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[0, 1, 0])
    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_almost_equal(rot_mat.dot(coords), [[0, 0, 1],
                                                  [0, 1, 0],
                                                  [-1, 0, 0]])

    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[0, 0, 1])
    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_almost_equal(rot_mat.dot(coords), [[0, -1, 0],
                                                  [1, 0, 0],
                                                  [0, 0, 1]])

    # rotation axis
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[1, 0, 0])
    assert all_equal(geom.axis, np.array([1, 0, 0]))
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[0, 1, 0])
    assert all_equal(geom.axis, np.array([0, 1, 0]))
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[0, 0, 1])
    assert all_equal(geom.axis, np.array([0, 0, 1]))
    geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=[1, 2, 3])
    assert all_equal(geom.axis,
                     np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3]))

    with pytest.raises(ValueError):
        odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=(1,))
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=(1, 2))
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dAxisGeometry(apart, dpart, axis=(1, 2, 3, 4))

    # check str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_fanflat():
    """2D fanbeam geometry with 1D line detector."""

    # Parameters
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition(0, 1, 10)
    src_rad = 10
    det_rad = 5

    with pytest.raises(TypeError):
        odl.tomo.FanFlatGeometry([0, 1], dpart, src_rad, det_rad)
    with pytest.raises(TypeError):
        odl.tomo.FanFlatGeometry(apart, [0, 1], src_rad, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(apart, dpart, -1, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(apart, dpart, src_rad, -1)

    # Initialize
    geom = odl.tomo.FanFlatGeometry(apart, dpart, src_rad, det_rad)

    assert all_almost_equal(geom.angles, apart.points().ravel())

    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert geom.det_refpoint(0)[0] == det_rad
    assert geom.det_refpoint(np.pi)[0] == -det_rad

    assert geom.ndim == 2
    assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # detector rotation
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_almost_equal(rot_mat.dot([1, 0]), [0, 1])
    assert all_almost_equal(rot_mat.dot([0, 1]), [-1, 0])

    # check str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_circular_cone_flat():
    """Conebeam geometry with circular acquisition and a flat 2D detector."""

    # Parameters
    full_angle = np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], [10, 10])
    src_rad = 10
    det_rad = 5

    with pytest.raises(TypeError):
        odl.tomo.CircularConeFlatGeometry([0, 1], dpart, src_rad, det_rad)
    with pytest.raises(TypeError):
        odl.tomo.CircularConeFlatGeometry(apart, [0, 1], src_rad, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(apart, dpart, -1, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(apart, dpart, src_rad, -1)

    # Initialize
    geom = odl.tomo.CircularConeFlatGeometry(apart, dpart, src_rad, det_rad)

    assert all_almost_equal(geom.angles, apart.points().ravel())

    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert geom.ndim == 3
    assert np.linalg.norm(geom.det_refpoint(0)) == det_rad
    assert np.linalg.norm(geom.src_position(np.pi)) == src_rad
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # check str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


def test_helical_cone_flat():
    """Conebeam geometry with helical acquisition and a flat 2D detector."""

    # Parameters
    full_angle = 2 * np.pi
    apart = odl.uniform_partition(0, full_angle, 10)
    dpart = odl.uniform_partition([0, 0], [1, 1], [10, 10])
    src_rad = 10.0
    det_rad = 5.0
    pitch = 1.5

    with pytest.raises(TypeError):
        odl.tomo.HelicalConeFlatGeometry([0, 1], dpart,
                                         src_rad, det_rad, pitch)
    with pytest.raises(TypeError):
        odl.tomo.HelicalConeFlatGeometry(apart, [0, 1],
                                         src_rad, det_rad, pitch)
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(apart, dpart, -1, det_rad, pitch)
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(apart, dpart, src_rad, -1, pitch)

    # Initialize
    geom = odl.tomo.HelicalConeFlatGeometry(apart, dpart,
                                            src_rad, det_rad, pitch)

    assert all_almost_equal(geom.angles, apart.points().ravel())

    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert np.linalg.norm(geom.det_refpoint(0)) == det_rad
    assert almost_equal(np.linalg.norm(geom.det_refpoint(np.pi / 4)[0:2]),
                        det_rad)

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    det_refpoint = geom.det_refpoint(2 * np.pi)
    assert almost_equal(np.linalg.norm(det_refpoint[0:2]), det_rad)

    angles = geom.angles
    num_angles = geom.angles.size

    src_rad = geom.src_radius
    det_rad = geom.det_radius
    pitch = geom.pitch

    for ang_ind in range(num_angles):
        angle = angles[ang_ind]
        z = pitch * angle / (2 * np.pi)

        # source
        pnt = (-np.cos(angle) * src_rad, -np.sin(angle) * src_rad, z)
        assert all_almost_equal(geom.src_position(angle), pnt)

        # center of detector
        det = geom.det_refpoint(angle)
        src = geom.src_position(angle)
        val0 = np.linalg.norm(src[0:2]) / src_rad
        val1 = np.linalg.norm(det[0:2]) / det_rad
        assert almost_equal(val0, val1)
        assert almost_equal(src[2], det[2])

    # check str and repr work without crashing and return something
    assert str(geom)
    assert repr(geom)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
