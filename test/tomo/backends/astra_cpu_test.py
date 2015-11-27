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
from odl.tomo import (Parallel2dGeometry, FanFlatGeometry)
from odl.tomo.backends.astra_cpu import (astra_cpu_forward_projector_call,
                                        astra_cpu_backward_projector_call)
from odl.tomo.util.testutils import skip_if_no_astra

# if ASTRA_AVAILABLE:
#     import astra
# else:
#     astra = None


# TODO: test other interpolations once implemented
@skip_if_no_astra
def test_astra_cpu_projector_call_2d():

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
    # TODO: question: intervals have default index, grids not
    npixels = angle_grid.insert(det_grid, ind).shape
    discr_proj_space = uniform_discr(proj_space, npixels, dtype='float32')

    print('\n\n angle interval:', angle_intvl, '\n det params:', dparams,
          '\n proj rectangle:', proj_rect)
    print(' vol data:', discr_data.shape,
          np.min(discr_data), np.max(discr_data), np.mean(discr_data),
          discr_vol_space.interp)
    print(' proj space:', discr_proj_space.grid.shape)

    # PARALLEL 2D: forward
    proj_data = astra_cpu_forward_projector_call(discr_data, geom_p2d,
                                         discr_proj_space)
    proj_data.show('imshow', saveto=for_dir + 'parallel2d_cpu.png',
                   title='PARALLEL 2D FORWARD CPU')

    # FANFLAT: forward
    discr_data = discr_vol_space.element(p)
    proj_data = astra_cpu_forward_projector_call(discr_data, geom_ff,
                                     discr_proj_space)
    proj_data.show('imshow', saveto=for_dir + 'fanflat_cpu.png',
                   title='FANFLAT 2D FORWARD CPU')

    # PARALLEL 2D: backward
    reco_data = astra_cpu_backward_projector_call(proj_data, geom_p2d,
                                                  discr_vol_space)
    reco_data.show('imshow', saveto=back_dir + 'parallel2d_cpu.png',
                   title='PARALLEL 2D BACKWARD CPU')

    # FANFLAT: backward
    reco_data = astra_cpu_backward_projector_call(proj_data, geom_ff,
                                                  discr_vol_space)
    reco_data.show('imshow', saveto=back_dir + 'fanflat_cpu.png',
               title='FANFLAT 2D BACKWARD CPU')
