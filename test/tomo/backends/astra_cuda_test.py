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

"""Test CUDA backend for ASTRA."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Internal
from odl.tomo.backends import ASTRA_AVAILABLE
from odl.set.domain import Interval, Rectangle
from odl.space.fspace import FunctionSpace
from odl.discr.lp_discr import uniform_discr, uniform_discr_fromspace
from odl.discr.grid import uniform_sampling
from odl.tomo.backends.astra_cuda import (astra_gpu_forward_projector_call,
                                          astra_gpu_backward_projector_call)
from odl.tomo.geometry.parallel import Parallel2dGeometry, Parallel3dGeometry
from odl.tomo.geometry.fanbeam import FanFlatGeometry
from odl.tomo.geometry.conebeam import (CircularConeFlatGeometry,
                                        HelicalConeFlatGeometry)
from odl.tomo.util.testutils import skip_if_no_astra
# from odl.util.testutils import (all_equal, all_almost_equal, almost_equal,
#                                 is_subdict)


# TODO: test other interpolations once implemented

@skip_if_no_astra
def test_astra_gpu_projector_call_2d():
    """Test 2D forward and backward projection functions on the GPU."""

    for_dir = '/home/jmoosmann/Documents/astra_odl/forward/'
    back_dir = '/home/jmoosmann/Documents/astra_odl/backward/'

    def save_slice(data, folder, name):
        """Save image.

        Parameters
        ----------
        data : `DiscreteLp`
        folder : `str`
        name : `str`
        """
        data.show('imshow', saveto='{}{}.png'.format(
            folder, name.replace(' ', '_')),
                  title='{} [:,:]'.format(name))
        plt.close('all')

    # 2D phantom
    vol_space = FunctionSpace(Rectangle([-1.0, -1.1], [1.0, 1.1]))
    nvoxels = (100, 110)
    discr_vol_space = uniform_discr_fromspace(vol_space, nvoxels,
                                              dtype='float32')
    phan = np.zeros(nvoxels)
    phan[20:30, 20:30] = 1
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
    discr_proj_space = uniform_discr_fromspace(proj_space, npixels,
                                               dtype='float32')

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


@skip_if_no_astra
def test_astra_gpu_projector_call_3d():
    """Test 3D forward and backward projection functions on the GPU."""

    for_dir = '/home/jmoosmann/Documents/astra_odl/forward/'
    back_dir = '/home/jmoosmann/Documents/astra_odl/backward/'

    def save_ortho_slices(data, folder, name, sli):
        """Save three orthogonal slices.

        Parameters
        ----------
        data : `DiscreteLp`
        folder : `str`
        name : `str`
        sli : three-element array-like
        """
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
        plt.close('all')

    # Volume space
    vol_shape = (80, 70, 60)
    discr_vol_space = uniform_discr([-0.8, -0.7, -0.6], [0.8, 0.7, 0.6],
                                    vol_shape, dtype='float32')

    # Phantom
    phan = np.zeros(vol_shape)
    sli0 = np.round(0.1 * np.array(vol_shape))
    sli1 = np.round(0.4 * np.array(vol_shape))
    sliz0 = np.round(0.1 * np.array(vol_shape))
    sliz1 = np.round(0.9 * np.array(vol_shape))
    phan[sliz0[0]:sliz1[0], sli0[1]:sli1[1], sli0[2]:sli1[2]] = 1

    discr_data = discr_vol_space.element(phan)
    vol_sli = np.round(0.25 * np.array(vol_shape))
    save_ortho_slices(discr_data, back_dir, 'phantom 3d gpu', vol_sli)

    # Angles
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 110, as_midp=False)

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
    spiral_pitch_factor = 0.5
    geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                       det_rad, spiral_pitch_factor,
                                       angle_grid, det_grid, angle_offset)

    # Projection space (3D) (DiscreteLp)
    ind = 0
    proj_rect = dparams.insert(angle_intvl, ind)
    proj_space = FunctionSpace(proj_rect)
    proj_grid = det_grid.insert(angle_grid, ind)
    npixels = proj_grid.shape
    proj_sli = (0, np.round(0.25*npixels[1]), np.round(0.25*npixels[2]))
    discr_proj_space = uniform_discr_fromspace(proj_space, npixels,
                                               dtype='float32')

    # Forward: Parallel 3D
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_p3d,
                                                 discr_proj_space)
    save_ortho_slices(proj_data, for_dir, 'parallel 3d gpu', proj_sli)

    # Backward: Parallel 3D
    rec_data = astra_gpu_backward_projector_call(proj_data, geom_p3d,
                                                 discr_vol_space)
    save_ortho_slices(rec_data, back_dir, 'parallel 3d gpu', vol_sli)

    # Forward: Circular Cone Flat
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_ccf,
                                                 discr_proj_space)
    save_ortho_slices(proj_data, for_dir, 'conebeam circular gpu', proj_sli)

    # Backward: Circular Cone Flat
    rec_data = astra_gpu_backward_projector_call(proj_data, geom_ccf,
                                                 discr_vol_space)
    save_ortho_slices(rec_data, back_dir, 'conebeam circular gpu', vol_sli)

    # Forward: Helical Cone Flat
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_hcf,
                                                 discr_proj_space)
    save_ortho_slices(proj_data, for_dir, 'conebeam helical gpu', proj_sli)

    # Backward: Helical Cone Flat
    rec_data = astra_gpu_backward_projector_call(proj_data, geom_hcf,
                                                 discr_vol_space)
    save_ortho_slices(rec_data, back_dir, 'conebeam helical gpu', vol_sli)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
