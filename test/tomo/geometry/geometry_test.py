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


# TODO: test for rotations about arbitray axis once implemented

def test_parallel_2d_geometry():
    """General parallel 2D geometries."""

    # initialize
    full_angle = np.pi
    angle_intvl = odl.Interval(0, full_angle)
    agrid = odl.uniform_sampling(angle_intvl, 10)
    dparams = odl.Interval(0, 1)
    dgrid = odl.uniform_sampling(dparams, 10)
    odl.tomo.Parallel2dGeometry(angle_intvl, dparams)

    with pytest.raises(TypeError):
        odl.tomo.Parallel2dGeometry([0, 1], dparams)
    with pytest.raises(TypeError):
        odl.tomo.Parallel2dGeometry(angle_intvl, [0, 1])
    with pytest.raises(ValueError):
        odl.tomo.Parallel2dGeometry(angle_intvl, dparams,
                                    agrid=odl.TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        odl.tomo.Parallel2dGeometry(angle_intvl, dparams,
                                    dgrid=odl.TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        odl.tomo.Parallel2dGeometry(angle_intvl, dparams,
                                    dgrid=odl.TensorGrid([0, 0.1], [0.2, 0.3]))

    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, agrid, dgrid)

    assert geom.ndim == 2
    assert isinstance(geom.detector, odl.tomo.Flat1dDetector)

    # detector rotation
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    rot_mat = geom.rotation_matrix(np.pi / 2)
    e1 = np.matrix([[1, 0]]).transpose()
    e2 = np.matrix([0, 1]).transpose()

    e1r = rot_mat * e1
    e2r = rot_mat * e2

    assert almost_equal(np.sum(e1r - e2), 0, places=16)
    assert almost_equal(np.sum(e2r + e1), 0, places=16)


def test_parallel_3d_geometry():
    """General parallel 3D geometries."""

    # initialize
    full_angle = np.pi
    angle_intvl = odl.Interval(0, full_angle)
    dparams = odl.IntervalProd([0, 0], [1, 1])
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 0, 0))

    with pytest.raises(TypeError):
        odl.tomo.Parallel3dGeometry([0, 1], dparams)
    with pytest.raises(TypeError):
        odl.tomo.Parallel3dGeometry(angle_intvl, [0, 1])

    with pytest.raises(ValueError):
        odl.tomo.Parallel3dGeometry(angle_intvl, dparams,
                                    agrid=odl.TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dGeometry(angle_intvl, dparams,
                                    dgrid=odl.TensorGrid([0, 0.5], [0.5, 1.5]))

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    # detector rotation
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 0, 0))
    with pytest.raises(ValueError):
        geom.rotation_matrix(2 * full_angle)

    # rotation of cartesian basis vectors about each other
    a = 32.1
    e1 = np.matrix([a, 0, 0]).transpose()
    e2 = np.matrix([0, a, 0]).transpose()
    e3 = np.matrix([0, 0, a]).transpose()

    places = 14

    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 0, 0))
    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_equal(rot_mat * e1, e1)
    assert almost_equal(np.sum(rot_mat * e2 - e3), 0, places=places)
    assert almost_equal(np.sum(rot_mat * e3 - (-e2)), 0, places=places)

    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(0, 1, 0))
    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_equal(rot_mat * e2, e2)
    assert almost_equal(np.sum(rot_mat * e3 - e1), 0, places=places)
    assert almost_equal(np.sum(rot_mat * e1 - (-e3)), 0, places=places)

    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(0, 0, 1))
    rot_mat = geom.rotation_matrix(np.pi / 2)
    assert all_equal(rot_mat * e3, e3)
    assert almost_equal(np.sum(rot_mat * e1 - e2), 0, places=places)
    assert almost_equal(np.sum(rot_mat * e2 - (-e1)), 0, places=places)
    assert all_almost_equal(np.array(rot_mat * e2), np.array(-e1),
                            places=places)

    # rotation axis
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 0, 0))
    assert all_equal(geom.axis, np.array([1, 0, 0]))
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(0, 1, 0))
    assert all_equal(geom.axis, np.array([0, 1, 0]))
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(0, 0, 1))
    assert all_equal(geom.axis, np.array([0, 0, 1]))
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 2, 3))
    assert all_equal(geom.axis,
                     np.array([1, 2, 3]) / np.linalg.norm(np.array([1, 2, 3])))

    with pytest.raises(ValueError):
        odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1,))
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 2))
    with pytest.raises(ValueError):
        odl.tomo.Parallel3dGeometry(angle_intvl, dparams, axis=(1, 2, 3, 4))


def test_fanflat():
    """2D fanbeam geometry with 1D line detector."""

    # initialize
    full_angle = np.pi
    angle_intvl = odl.Interval(0, full_angle)
    dparams = odl.Interval(0, 1)
    src_rad = 10
    det_rad = 5
    geom = odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad)

    with pytest.raises(TypeError):
        odl.tomo.FanFlatGeometry([0, 1], dparams, src_rad, det_rad)
    with pytest.raises(TypeError):
        odl.tomo.FanFlatGeometry(angle_intvl, [0, 1], src_rad, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                 agrid=odl.TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                 dgrid=odl.TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                 dgrid=odl.TensorGrid([0, 0.1], [0.2, 0.3]))
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(angle_intvl, dparams, -1, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, -1)
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
    e1 = np.matrix([[1, 0]]).transpose()
    e2 = np.matrix([0, 1]).transpose()

    e1r = rot_mat * e1
    e2r = rot_mat * e2

    assert almost_equal(np.sum(e1r - e2), 0, places=16)
    assert almost_equal(np.sum(e2r + e1), 0, places=16)


def test_circular_cone_flat():
    """Conebeam geometry with circular acquisition and a flat 2D detector."""

    # initialize
    full_angle = np.pi
    angle_intvl = odl.Interval(0, full_angle)
    dparams = odl.IntervalProd([0, 0], [1, 1])
    src_rad = 10
    det_rad = 5

    geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad,
                                             det_rad)

    with pytest.raises(TypeError):
        odl.tomo.CircularConeFlatGeometry([0, 1], dparams, src_rad, det_rad)
    with pytest.raises(TypeError):
        odl.tomo.CircularConeFlatGeometry(angle_intvl, [0, 1], src_rad,
                                          det_rad)
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad,
                                          det_rad, agrid=odl.TensorGrid(
                                              [0, 2 * full_angle]))
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad,
                                          det_rad,
                                          dgrid=odl.TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad,
                                          det_rad,
                                          dgrid=odl.TensorGrid([0, 0.1],
                                                               [0.2, 0.3],
                                                               [0.3, 0.4,
                                                                0.5]))
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, -1, det_rad)
    with pytest.raises(ValueError):
        odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad, -1)
    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert geom.ndim == 3
    assert np.linalg.norm(geom.det_refpoint(0)) == det_rad
    assert np.linalg.norm(geom.src_position(np.pi)) == src_rad
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)


def test_helical_cone_flat():
    """Conebeam geometry with helical acquisition and a flat 2D detector."""

    # initialize
    full_angle = 2 * np.pi
    angle_intvl = odl.Interval(0, full_angle)
    motion_grid = odl.uniform_sampling(angle_intvl, 5, as_midp=False)
    dparams = odl.IntervalProd([-40, -3], [40, 3])
    det_grid = odl.uniform_sampling(dparams, (10, 5))
    src_rad = 10.0
    det_rad = 5.0
    pitch = 1.5

    geom = odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                            det_rad, pitch)

    with pytest.raises(TypeError):
        odl.tomo.HelicalConeFlatGeometry([0, 1], dparams, src_rad, det_rad,
                                         pitch)
    with pytest.raises(TypeError):
        odl.tomo.HelicalConeFlatGeometry(angle_intvl, [0, 1], src_rad,
                                         det_rad, pitch)
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                         det_rad, pitch,
                                         agrid=odl.TensorGrid(
                                             [0, 2 * full_angle]))
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                         det_rad, pitch,
                                         dgrid=odl.TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                         det_rad, pitch,
                                         dgrid=odl.TensorGrid(
                                             [0, 0.1], [0.2, 0.3],
                                             [0.3, 0.4, 0.5]))
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, -1, det_rad,
                                         pitch)
    with pytest.raises(ValueError):
        odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, -1,
                                         pitch)
    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert np.linalg.norm(geom.det_refpoint(0)) == det_rad
    assert almost_equal(np.linalg.norm(geom.det_refpoint(np.pi / 4)[0:2]),
                        det_rad)

    assert geom.ndim == 3
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)

    assert all_equal(dparams.size, dparams.max() - dparams.min())

    det_refpoint = geom.det_refpoint(2 * np.pi)
    assert almost_equal(np.linalg.norm(det_refpoint[0:2]), det_rad)

    geom = odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                            det_rad, pitch, motion_grid,
                                            det_grid)

    angles = geom.motion_grid
    num_angles = geom.motion_grid.ntotal

    src_rad = geom.src_radius
    det_rad = geom.det_radius
    pitch = geom.pitch

    for ang_ind in range(num_angles):
        angle = angles[ang_ind][0]
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


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -vs')
