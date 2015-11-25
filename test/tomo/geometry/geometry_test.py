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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
from odl.set.domain import Interval, IntervalProd
from odl.discr.grid import TensorGrid, uniform_sampling
from odl.tomo.geometry.detector import LineDetector, Flat2dDetector
from odl.tomo.geometry.parallel import Parallel2dGeometry, Parallel3dGeometry
from odl.tomo.geometry.fanbeam import FanFlatGeometry
from odl.tomo.geometry.conebeam import (CircularConeFlatGeometry,
    HelicalConeFlatGeometry)
from odl.util.testutils import almost_equal, all_almost_equal


# TODO: test for axis orientation and scaling factors

def test_parallel_2d_geometry():
    # initialize
    full_angle = np.pi
    angle_intvl = Interval(0, full_angle)
    agrid = uniform_sampling(angle_intvl, 10)
    dparams = Interval(0, 1)
    dgrid = uniform_sampling(dparams, 10)
    Parallel2dGeometry(angle_intvl, dparams)

    with pytest.raises(TypeError):
        Parallel2dGeometry([0, 1], dparams)
    with pytest.raises(TypeError):
        Parallel2dGeometry(angle_intvl, [0, 1])
    with pytest.raises(ValueError):
        Parallel2dGeometry(angle_intvl, dparams,
                           agrid=TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        Parallel2dGeometry(angle_intvl, dparams, dgrid=TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        Parallel2dGeometry(angle_intvl, dparams,
                           dgrid=TensorGrid([0, 0.1], [0.2, 0.3]))

    geom = Parallel2dGeometry(angle_intvl, dparams, agrid, dgrid)

    assert geom.ndim == 2
    assert isinstance(geom.detector, LineDetector)

    # detector rotation
    with pytest.raises(ValueError):
        geom.det_rotation(2 * full_angle)

    rot_mat = geom.det_rotation(np.pi / 2)
    e1 = np.matrix([[1, 0]]).transpose()
    e2 = np.matrix([0, 1]).transpose()

    e1r = rot_mat * e1
    e2r = rot_mat * e2

    assert almost_equal(np.sum(e1r - e2), 0, places=16)
    assert almost_equal(np.sum(e2r + e1), 0, places=16)

    sp = geom.src_position(0)
    # print('\n source position:', sp)

    dts = geom.det_to_src(0, 1)
    # print('\n detector to source:', dts)


def test_parallel_3d_geometry():
    # initialize
    full_angle = np.pi
    angle_intvl = Interval(0, full_angle)
    dparams = IntervalProd([0, 0], [1, 1])
    geom = Parallel3dGeometry(angle_intvl, dparams)

    with pytest.raises(TypeError):
        Parallel3dGeometry([0, 1], dparams)
    with pytest.raises(TypeError):
        Parallel3dGeometry(angle_intvl, [0, 1])

    with pytest.raises(ValueError):
        Parallel3dGeometry(angle_intvl, dparams,
                           agrid=TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        Parallel3dGeometry(angle_intvl, dparams,
                           dgrid=TensorGrid([0, 0.5], [0.5, 1.5]))

    assert geom.ndim == 3
    assert isinstance(geom.detector, Flat2dDetector)

    # detector rotation
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=0)
    with pytest.raises(ValueError):
        geom.det_rotation(2 * full_angle)

    # rotation of cartesion basis vectors about each other
    a = 32.1
    e1 = np.matrix([a, 0, 0]).transpose()
    e2 = np.matrix([0, a, 0]).transpose()
    e3 = np.matrix([0, 0, a]).transpose()

    places = 14

    geom = Parallel3dGeometry(angle_intvl, dparams, axis=0)
    rot_mat = geom.det_rotation(np.pi / 2)
    assert all(rot_mat * e1 == e1)
    assert almost_equal(np.sum(rot_mat * e2 - e3), 0, places=places)
    assert almost_equal(np.sum(rot_mat * e3 - (-e2)), 0, places=places)

    geom = Parallel3dGeometry(angle_intvl, dparams, axis=1)
    rot_mat = geom.det_rotation(np.pi / 2)
    assert all(rot_mat * e2 == e2)
    assert almost_equal(np.sum(rot_mat * e3 - e1), 0, places=places)
    assert almost_equal(np.sum(rot_mat * e1 - (-e3)), 0, places=places)

    geom = Parallel3dGeometry(angle_intvl, dparams, axis=2)
    rot_mat = geom.det_rotation(np.pi / 2)
    assert all(rot_mat * e3 == e3)
    assert almost_equal(np.sum(rot_mat * e1 - e2), 0, places=places)
    assert almost_equal(np.sum(rot_mat * e2 - (-e1)), 0, places=places)
    assert all_almost_equal(np.array(rot_mat * e2), np.array(-e1),
                            places=places)

    # np.set_printoptions(precision=4, suppress=True)
    # print('\n\n', rot_mat * e1, '\n\n', rot_mat * e2, '\n\n', rot_mat * e3,
    #       '\n\n', rot_mat)

    # rotation axis
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=None)
    assert all(geom.axis == np.array([0, 0, 1]))
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=0)
    assert all(geom.axis == np.array([1, 0, 0]))
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=1)
    assert all(geom.axis == np.array([0, 1, 0]))
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=2)
    assert all(geom.axis == np.array([0, 0, 1]))
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=(1, 2, 3))
    assert all(geom.axis == np.array([1, 2, 3]))
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=np.array((1, 2, 3)))
    assert all(geom.axis == np.array([1, 2, 3]))

    geom = Parallel3dGeometry(angle_intvl, dparams, axis=(1,))
    with pytest.raises(ValueError):
        geom.axis
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=(1, 2))
    with pytest.raises(ValueError):
        geom.axis
    geom = Parallel3dGeometry(angle_intvl, dparams, axis=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        geom.axis


def test_fanflat():
    # initialize
    full_angle = np.pi
    angle_intvl = Interval(0, full_angle)
    dparams = Interval(0, 1)
    src_rad = 10
    det_rad = 5
    geom = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad)

    with pytest.raises(TypeError):
        FanFlatGeometry([0, 1], dparams, src_rad, det_rad)
    with pytest.raises(TypeError):
        FanFlatGeometry(angle_intvl, [0, 1], src_rad, det_rad)
    with pytest.raises(ValueError):
        FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                        agrid=TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                        dgrid=TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                        dgrid=TensorGrid([0, 0.1], [0.2, 0.3]))
    with pytest.raises(ValueError):
        FanFlatGeometry(angle_intvl, dparams, -1, det_rad)
    with pytest.raises(ValueError):
        FanFlatGeometry(angle_intvl, dparams, src_rad, -1)
    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert geom.det_refpoint(0)[0] == det_rad
    assert geom.det_refpoint(np.pi)[0] == -det_rad

    assert geom.ndim == 2
    assert isinstance(geom.detector, LineDetector)

    # detector rotation
    with pytest.raises(ValueError):
        geom.det_rotation(2 * full_angle)

    rot_mat = geom.det_rotation(np.pi / 2)
    e1 = np.matrix([[1, 0]]).transpose()
    e2 = np.matrix([0, 1]).transpose()

    e1r = rot_mat * e1
    e2r = rot_mat * e2

    assert almost_equal(np.sum(e1r - e2), 0, places=16)
    assert almost_equal(np.sum(e2r + e1), 0, places=16)


def test_circular_cone_flat():
    # initialize
    full_angle = np.pi
    angle_intvl = Interval(0, full_angle)
    dparams = IntervalProd([0, 0], [1, 1])
    src_rad = 10
    det_rad = 5

    geom = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad)

    with pytest.raises(TypeError):
        CircularConeFlatGeometry([0, 1], dparams, src_rad, det_rad)
    with pytest.raises(TypeError):
        CircularConeFlatGeometry(angle_intvl, [0, 1], src_rad, det_rad)
    with pytest.raises(ValueError):
        CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                 agrid=TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                 dgrid=TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                 dgrid=TensorGrid([0, 0.1], [0.2, 0.3],
                                                  [0.3, 0.4, 0.5]))
    with pytest.raises(ValueError):
        CircularConeFlatGeometry(angle_intvl, dparams, -1, det_rad)
    with pytest.raises(ValueError):
        CircularConeFlatGeometry(angle_intvl, dparams, src_rad, -1)
    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert geom.det_refpoint(0)[0] == det_rad
    assert geom.det_refpoint(np.pi)[0] == -det_rad

    assert geom.ndim == 3
    assert isinstance(geom.detector, Flat2dDetector)


def test_helical_cone_flat():
    # initialize
    full_angle = 2 * np.pi
    angle_intvl = Interval(0, full_angle)
    angle_grid = uniform_sampling(angle_intvl, 5, as_midp=False)
    dparams = IntervalProd([-40, -3], [40, 3])
    det_grid = uniform_sampling(dparams, (10, 5))
    src_rad = 10
    det_rad = 5
    spiral_pitch_factor = 1

    geom = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                   spiral_pitch_factor)

    with pytest.raises(TypeError):
        HelicalConeFlatGeometry([0, 1], dparams, src_rad, det_rad,
                                spiral_pitch_factor)
    with pytest.raises(TypeError):
        HelicalConeFlatGeometry(angle_intvl, [0, 1], src_rad, det_rad,
                                spiral_pitch_factor)
    with pytest.raises(ValueError):
        HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                spiral_pitch_factor,
                                agrid=TensorGrid([0, 2 * full_angle]))
    with pytest.raises(ValueError):
        HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                spiral_pitch_factor, dgrid=TensorGrid([0, 2]))
    with pytest.raises(ValueError):
        HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                spiral_pitch_factor, dgrid=TensorGrid(
                                    [0, 0.1], [0.2, 0.3], [0.3, 0.4, 0.5]))
    with pytest.raises(ValueError):
        HelicalConeFlatGeometry(angle_intvl, dparams, -1, det_rad,
                                spiral_pitch_factor)
    with pytest.raises(ValueError):
        HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, -1,
                                spiral_pitch_factor)
    with pytest.raises(ValueError):
        geom.det_refpoint(2 * full_angle)

    assert geom.det_refpoint(0)[0] == det_rad
    assert geom.det_refpoint(np.pi)[0] == -det_rad

    assert geom.ndim == 3
    assert isinstance(geom.detector, Flat2dDetector)

    det_height = dparams.size[1]
    assert all(dparams.size == (dparams.max() - dparams.min()))
    assert geom.table_feed_per_rotation == spiral_pitch_factor * det_height

    det_refpoint = geom.det_refpoint(angle_intvl.max())
    assert det_refpoint[0] == det_rad
    assert almost_equal(det_refpoint[1], 0, places=14)
    assert almost_equal(det_refpoint[2], det_height, places=14)

    det_refpoint = geom.det_refpoint(2 * np.pi)
    assert det_refpoint[0] == det_rad
    assert almost_equal(det_refpoint[1], 0, places=14)
    assert almost_equal(det_refpoint[2], det_height * spiral_pitch_factor,
                        places=14)

    # Each row of vectors corresponds to a single projection, and consists of:
    #  ( srcX, srcY, srcZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
    #  src : the ray source
    #  d   : the center of the detector
    #  u   : the vector from detector pixel (0,0) to (0,1)
    #  v   : the vector from detector pixel (0,0) to (1,0)

    angle_offset = np.pi / 2
    geom = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                   spiral_pitch_factor, angle_grid,
                                   det_grid, angle_offset)
    print('\n angle offset', geom.angle_offset)

    angles = geom.angle_grid
    num_angles = geom.angle_grid.ntotal
    vec = np.zeros((num_angles, 12))
    vectors = np.zeros((num_angles, 12))

    src_radius = geom.src_radius
    det_radius = geom.det_radius
    det_pix_width = geom.det_grid.stride[0]
    det_pix_height = geom.det_grid.stride[1]
    tfpr = geom.table_feed_per_rotation

    for nn in range(num_angles):
        angle = angles[nn][0]
        z = tfpr * angle / (2 * np.pi)

        # source
        vec[nn, 0:3] = geom.src_position(angle)

        # center of detector
        vec[nn, 3:6] = geom.det_refpoint(angle)

        # vector from detector pixel (0,0) to (0,1)
        vec[nn, 6] = np.cos(angle) * det_pix_width
        vec[nn, 7] = np.sin(angle) * det_pix_width

        # vector from detector pixel (0,0) to (1,0)
        vec[nn, 11] = det_pix_height

        # source
        vectors[nn, 0] = np.sin(angle) * src_radius
        vectors[nn, 1] = -np.cos(angle) * src_radius
        vectors[nn, 2] = z

        # center of detector
        vectors[nn, 3] = -np.sin(angle) * det_radius
        vectors[nn, 4] = np.cos(angle) * det_radius
        vectors[nn, 5] = z

        # vector from detector pixel (0,0) to (0,1)
        vectors[nn, 6] = np.cos(angle) * det_pix_width
        vectors[nn, 7] = np.sin(angle) * det_pix_width
        vectors[nn, 8] = 0

        # vector from detector pixel (0,0) to (1,0)
        vectors[nn, 9] = 0
        vectors[nn, 10] = 0
        vectors[nn, 11] = det_pix_height

        forstr = '{:5.1f}, ' * 3
        print('\nastra :', nn, ' angle: {:03.2f}'.format(angle / np.pi),
              ' src: ' + forstr.format(*vectors[nn, 0:3]),
              ' det: ' + forstr.format(*vectors[nn, 3:6]),
              ' vpd: ' + forstr.format(*vectors[nn, 6:9]),
              ' hpd: ' + forstr.format(*vectors[nn, 9:12]))

        print('odl   :', nn, ' angle: {:03.2f}'.format(angle / np.pi),
              ' src: ' + forstr.format(*vec[nn, 0:3]),
              ' det: ' + forstr.format(*vec[nn, 3:6]),
              ' vpd: ' + forstr.format(*vec[nn, 6:9]),
              ' hpd: ' + forstr.format(*vec[nn, 9:12]))


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -vs')
