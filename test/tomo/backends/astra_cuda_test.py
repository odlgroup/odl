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
from odl.discr.grid import uniform_sampling
# from odl.util.testutils import all_equal, is_subdict
from odl.tomo import ASTRA_AVAILABLE
from odl.tomo import (Parallel2dGeometry, FanFlatGeometry,
    Parallel3dGeometry, CircularConeFlatGeometry, HelicalConeFlatGeometry)
from odl.tomo.backends.astra_cuda import (astra_gpu_forward_projector_call,
                                          astra_gpu_backward_projector_call)
from odl.tomo.util.testutils import skip_if_no_astra

# if ASTRA_AVAILABLE:
#     import astra
# else:
#     astra = None


# TODO: test other interpolations once implemented
@skip_if_no_astra
def test_astra_gpu_projector_call_2d():

    for_dir = '/home/jmoosmann/Documents/astra_odl/forward/'
    back_dir = '/home/jmoosmann/Documents/astra_odl/backward/'

    # DiscreteLp element
    vol_space = FunctionSpace(Rectangle([-1, -1.1], [1, 1.1]))
    nvoxels = (100, 110)
    discr_vol_space = uniform_discr(vol_space, nvoxels, dtype='float32')
    p = np.zeros(nvoxels)
    p[20:30, 20:30] = 1
    discr_data = discr_vol_space.element(p)
    discr_data.show('imshow', saveto=back_dir + 'phantom.png',
                    title='PHANTOM')

    # motion and detector parameters, and geometry
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 180, as_midp=False)
    dparams = Interval(-2, 2)
    det_grid = uniform_sampling(dparams, 200)
    geom_p2d = Parallel2dGeometry(angle_intvl, dparams, angle_grid, det_grid)

    src_rad = 1000
    det_rad = 100
    geom_ff = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              angle_grid, det_grid, angle_offset)

    # DiscreteLp
    ind = 1
    proj_rect = angle_intvl.insert(dparams, ind)
    proj_space = FunctionSpace(proj_rect)
    npixels = angle_grid.insert(det_grid, ind).shape
    discr_proj_space = uniform_discr(proj_space, npixels, dtype='float32')

    # PARALLEL 2D: forward
    p2_proj_data = astra_gpu_forward_projector_call(discr_data, geom_p2d,
                                                 discr_proj_space)
    p2_proj_data.show('imshow', saveto=for_dir + 'parallel2d_gpu.png',
                   title='PARALLEL 2D FORWARD GPU')

    # FANFLAT: forward
    discr_data = discr_vol_space.element(p)
    ff_proj_data = astra_gpu_forward_projector_call(discr_data, geom_ff,
                                                 discr_proj_space)
    ff_proj_data.show('imshow', saveto=for_dir + 'fanflat_gpu.png',
                   title='FANFLAT 2D FORWARD GPU')

    # PARALLEL 2D: backward
    p2_reco_data = astra_gpu_backward_projector_call(p2_proj_data, geom_p2d,
                                                  discr_vol_space)
    p2_reco_data.show('imshow', saveto=back_dir + 'parallel2d_gpu.png',
                   title='PARALLEL 2D BACKWARD GPU')

    # FANFLAT: backward
    # ff_reco_data = astra_gpu_backward_projector_call(ff_proj_data, geom_ff,
    #                                               discr_vol_space)
    # ff_reco_data.show('imshow', saveto=back_dir + 'fanflat_gpu.png',
    #                title='FANFLAT 2D BACKWARD GPU')


@skip_if_no_astra
def test_astra_gpu_projector_call_3d():

    # DiscreteLp element
    vol_space = FunctionSpace(Cuboid([-1, -1.1, -0.8], [1, 1.1, 0.8]))
    nvoxels = (50, 55, 40)
    discr_vol_space = uniform_discr(vol_space, nvoxels, dtype='float32')
    discr_data = discr_vol_space.element(1)

    # motion and detector parameters, and geometry
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 36, as_midp=False)
    dparams = Rectangle([-2, -1.5], [2, 1.5])
    det_grid = uniform_sampling(dparams, (40, 30))
    geom_p3d = Parallel3dGeometry(angle_intvl, dparams, angle_grid, det_grid)
    src_rad = 1000
    det_rad = 100
    geom_ccf = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              angle_grid, det_grid, angle_offset)
    spiral_pitch_factor = 2 / (2 * np.pi * 20)
    geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                       det_rad, spiral_pitch_factor,
                                       angle_grid, det_grid, angle_offset)

    # DiscreteLp
    ind = 1
    proj_rect = dparams.insert(angle_intvl, ind)
    proj_space = FunctionSpace(proj_rect)
    # TODO: question: intervals have default index, grids not
    proj_grid = det_grid.insert(angle_grid, ind)
    npixels = proj_grid.shape
    discr_proj_space = uniform_discr(proj_space, npixels, dtype='float32')

    # np.set_printoptions(precision=4, suppress=True)
    print('\n\n angle interval:', angle_intvl, '\n det params:', dparams,
          '\n proj rectangle:', proj_rect)
    print(' vol data:', discr_data.shape,
          np.min(discr_data), np.max(discr_data), np.mean(discr_data),
          discr_vol_space.interp)
    print(' proj:', npixels)

    save_dir = '/home/jmoosmann/Documents/astra_odl/forward/'

    # PARALLEL 3D
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_p3d,
                                                 discr_proj_space)
    print('\n proj p3d:', proj_data.shape, np.min(proj_data), np.max(
        proj_data), np.mean(proj_data), discr_proj_space.interp)

    # proj_data = astra_gpu_projector_call(discr_data, geom_p3d,
    #                                      discr_proj_space, 'backward')

    # proj_data.show('imshow', saveto=save_dir+'parallel3d_cuda.png')

    # CIRCULAR CONE FLAT
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_ccf,
                                                 discr_proj_space)
    print('\n proj ccf:', proj_data.shape, np.min(proj_data), np.max(
        proj_data), np.mean(proj_data), discr_proj_space.interp)
    # proj_data.show('imshow', saveto=save_dir+'fanflat_cuda.png')

    # HELICAL CONE BEAM
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_hcf,
                                                 discr_proj_space)
    print('\n proj hcf:', proj_data.shape, np.min(proj_data), np.max(
    proj_data), np.mean(proj_data), discr_proj_space.interp)
