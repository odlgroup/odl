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

from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
# import pytest

# Internal
from odl.set.domain import Interval, Rectangle, Cuboid
from odl.space.fspace import FunctionSpace
from odl.discr.lp_discr import uniform_discr
from odl.discr.grid import uniform_sampling, RegularGrid
# from odl.util.testutils import all_equal, is_subdict
from odl.tomo.backends import ASTRA_AVAILABLE
from odl.tomo.geometry.parallel import Parallel2dGeometry, Parallel3dGeometry
from odl.tomo.geometry.fanbeam import FanFlatGeometry
from odl.tomo.geometry.conebeam import (CircularConeFlatGeometry,
                                        HelicalConeFlatGeometry)
from odl.tomo.backends.astra_cuda import (astra_gpu_forward_projector_call,
                                          astra_gpu_backward_projector_call)
from odl.tomo.util.testutils import skip_if_no_astra


# TODO: test other interpolations once implemented

def save_slice(data, folder, name):
    data.show('imshow', saveto='{}{}.png'.format(
        folder, name.replace(' ', '_')),
              title='{} [:,:]'.format(name))


@skip_if_no_astra
def test_astra_gpu_projector_call_2d():
    """Test 2D forward and backward projection functions on the GPU."""

    for_dir = '/home/jmoosmann/Documents/astra_odl/forward/'
    back_dir = '/home/jmoosmann/Documents/astra_odl/backward/'

    def save_slice(data, folder, name):
        data.show('imshow', saveto='{}{}.png'.format(
            folder, name.replace(' ', '_')),
                  title='{} [:,:]'.format(name))

    # 2D phantom
    vol_space = FunctionSpace(Rectangle([-0.8, -0.7], [0.8, 0.7]))
    nvoxels = (80, 70)
    discr_vol_space = uniform_discr(vol_space, nvoxels, dtype='float32')
    phan = np.zeros(nvoxels)
    phan[15:25, 15:25] = 1
    discr_data = discr_vol_space.element(phan)
    save_slice(discr_data, back_dir, 'phantom 2d gpu')

    # Parameters
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 180, as_midp=False)
    dparams = Interval(-2, 2)
    det_grid = uniform_sampling(dparams, 100)

    # Parallel 2D Geometry
    geom_p2d = Parallel2dGeometry(angle_intvl, dparams, angle_grid, det_grid)

    # Fanflat Geometry
    src_rad = 1000
    det_rad = 100
    geom_ff = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              angle_grid, det_grid, angle_offset)

    # Projection space
    ind = 1
    proj_rect = angle_intvl.insert(dparams, ind)
    proj_space = FunctionSpace(proj_rect)
    npixels = angle_grid.insert(det_grid, ind).shape
    discr_proj_space = uniform_discr(proj_space, npixels, dtype='float32')

    # PARALLEL 2D: forward
    proj_data_p2d = astra_gpu_forward_projector_call(discr_data, geom_p2d,
                                                     discr_proj_space)
    save_slice(proj_data_p2d, for_dir, 'parallel 2d gpu')

    # FANFLAT: forward
    discr_data = discr_vol_space.element(phan)
    proj_data_ff = astra_gpu_forward_projector_call(discr_data, geom_ff,
                                                    discr_proj_space)
    save_slice(proj_data_ff, for_dir, 'fanflat gpu')

    # PARALLEL 2D: backward
    reco_data_p2d = astra_gpu_backward_projector_call(proj_data_p2d, geom_p2d,
                                                      discr_vol_space)
    save_slice(reco_data_p2d, back_dir, 'parallel 2d gpu')

    # FANFLAT: backward
    reco_data_ff = astra_gpu_backward_projector_call(proj_data_ff, geom_ff,
                                                     discr_vol_space)
    save_slice(reco_data_ff, back_dir, 'fanflat_gpu')

    print('\n 2D sino:', proj_data_p2d.shape)


def test_astra_gpu_projector_2d_3d_correspondence():
    """The 3D projector for a single-slice volume should match the 2D
    projector.
    """
    folder = '/home/jmoosmann/Documents/astra_odl/2d3d/'

    # 2D phantom
    vol_space2 = FunctionSpace(Rectangle([-1, -1], [1, 1]))
    nvoxels2 = (100, 100)
    discr_vol_space2 = uniform_discr(vol_space2, nvoxels2, dtype='float32')
    phan = np.zeros(nvoxels2)
    phan[20:30, 20:30] = 1
    discr_data2 = discr_vol_space2.element(phan)
    save_slice(discr_data2, folder, 'phantom 2d gpu')

    # 3D phantom
    vol_space3 = FunctionSpace(Cuboid([-1, -1, -0.01], [1, 1, 0.01]))
    nvoxels3 = (100, 100, 1)
    discr_vol_space3 = uniform_discr(vol_space3, nvoxels3, dtype='float32')
    phan = np.zeros(nvoxels3)
    phan[20:30, 20:30, 0] = 1
    discr_data3 = discr_vol_space3.element(phan)
    save_slice(discr_data3, folder, 'phantom 3D gpu')

    # Angles
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 180, as_midp=False)

    # 2D detector
    dparams2 = Interval(-1, 1)
    det_grid2 = uniform_sampling(dparams2, 100)
    print('\n geom 2d: det stride:', det_grid2.stride)

    # 3d detector
    dparams3 = Rectangle([-1, -0.01], [1, 0.01])
    det_grid3 = uniform_sampling(dparams3, (100, 1))

    # 2d geometry
    geom2 = Parallel2dGeometry(angle_intvl, dparams2, angle_grid, det_grid2)
    print('\n geom 2d: det stride:', geom2.det_grid.stride)

    # 3d geometry
    geom3 = Parallel3dGeometry(angle_intvl, dparams3, angle_grid, det_grid3)
    print('\n geom 3d: det stride:', geom3.det_grid.stride)

    # 2D projection space
    ind = 1
    proj_rect2 = angle_intvl.insert(dparams2, ind)
    proj_space2 = FunctionSpace(proj_rect2)
    npixels2 = angle_grid.insert(det_grid2, ind).shape
    discr_proj_space2 = uniform_discr(proj_space2, npixels2, dtype='float32')

    # 3D projection space
    ind = 0
    proj_rect3 = dparams3.insert(angle_intvl, ind)
    proj_space3 = FunctionSpace(proj_rect3)
    proj_grid3 = det_grid3.insert(angle_grid, ind)
    npixels3 = proj_grid3.shape
    discr_proj_space3 = uniform_discr(proj_space3, npixels3, dtype='float32')

    # 2D forward
    proj_data2 = astra_gpu_forward_projector_call(discr_data2, geom2,
                                                     discr_proj_space2)
    save_slice(proj_data2, folder, 'parallel forward 2d gpu')

    # 3D forward
    proj_data3 = astra_gpu_forward_projector_call(discr_data3, geom3,
                                                     discr_proj_space3)
    # save_slice(proj_data3, folder, 'parallel forward 3d gpu')


@skip_if_no_astra
def test_astra_gpu_projector_call_3d():
    """Test 3D forward and backward projection functions on the GPU."""

    for_dir = '/home/jmoosmann/Documents/astra_odl/forward/'
    back_dir = '/home/jmoosmann/Documents/astra_odl/backward/'

    def save_ortho_slices(data, folder, name, sli):
        x, y, z = np.asarray(sli, int)
        data.show('imshow', saveto='{}{}_z{:03d}.png'.format(
            folder, name.replace(' ', '_'), z),
                  title='{} [:,:,{}]'.format(name, z),
                  indices=[slice(None), slice(None), z])
        data.show('imshow', saveto='{}{}_y{:03d}.png'.format(
            folder, name.replace(' ', '_'), y),
                  title='{} [:,{},:]'.format(name, y),
                  indices=[slice(None), y, slice(None)])
        data.show('imshow', saveto=folder + '{0}_x{1:03d}.png'.format(
            name.replace(' ', '_'), x),
                  title='{} [{},:,:]'.format(name, x),
                  indices=[x, slice(None), slice(None)])

    # Volumetric phantom
    vol_space = FunctionSpace(Cuboid([-0.8, -0.7, -0.6], [0.8, 0.7, 0.6]))
    nvoxels = (80, 70, 60)
    discr_vol_space = uniform_discr(vol_space, nvoxels, dtype='float32')
    phan = np.zeros(nvoxels)
    sli0 = np.round(0.1 * np.array(nvoxels))
    sli1 = np.round(0.4 * np.array(nvoxels))
    phan[sli0[0]:sli1[0], sli0[1]:sli1[1], sli0[2]:sli1[2]] = 1
    discr_data = discr_vol_space.element(phan)
    sli = np.round(0.25 * np.array(nvoxels))
    save_ortho_slices(discr_data, back_dir, 'phantom 3d gpu', sli)

    # Angles
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 180, as_midp=False)

    # Detector
    dparams = Rectangle([-1, -0.9], [1, 0.9])
    det_grid = uniform_sampling(dparams, (100, 90))

    # Geometry: Parallel 3D
    geom_p3d = Parallel3dGeometry(angle_intvl, dparams, angle_grid, det_grid)

    # Geomtery: Circular cone beam
    src_rad = 1000
    det_rad = 100
    geom_ccf = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                        angle_grid, det_grid, angle_offset)

    # Geometry: Helical cone beam
    spiral_pitch_factor = 0.01
    geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                       det_rad, spiral_pitch_factor,
                                       angle_grid, det_grid, angle_offset)

    # Projection space (3D) (DiscreteLp)
    ind = 0
    proj_rect = dparams.insert(angle_intvl, ind)
    proj_space = FunctionSpace(proj_rect)
    proj_grid = det_grid.insert(angle_grid, ind)
    npixels = proj_grid.shape
    sli = (0, np.round(0.25*npixels[1]), np.round(0.25*npixels[2]))
    discr_proj_space = uniform_discr(proj_space, npixels, dtype='float32')

    # Forward: PARALLEL 3D
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_p3d,
                                                 discr_proj_space)
    save_ortho_slices(proj_data, for_dir, 'parallel 3d gpu', sli)
    print('sino 3d pp :', proj_data.shape)
    # proj = proj_data.asarray()
    # print('\n proj_data:', type(proj), proj.shape, proj.min(), proj.max())
    # import matplotlib.pyplot as plt
    # im = plt.imshow(proj[:,:,1], interpolation='none', cmap=plt.cm.Greys)
    # plt.imsave('/home/jmoosmann/Documents/astra_odl/p2d_xy.png', im)

    # Forward: CIRCULAR CONE FLAT
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_ccf,
                                                 discr_proj_space)
    save_ortho_slices(proj_data, for_dir, 'conebeam circular gpu', sli)
    print('sino 3d ccb:', proj_data.shape)

    # HELICAL CONE BEAM
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_hcf,
                                                 discr_proj_space)
    save_ortho_slices(proj_data, for_dir, 'conebeam helical gpu', sli)

    print('sino 3d hcb:', proj_data.shape)
