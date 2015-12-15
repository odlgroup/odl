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

"""Test ASTRA backend using CPU."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest
import matplotlib
matplotlib.use('agg')

# Internal
from odl.tomo.backends import ASTRA_AVAILABLE
from odl.set.domain import Interval
from odl.space.fspace import FunctionSpace
from odl.discr.lp_discr import uniform_discr, uniform_discr_fromspace
from odl.discr.grid import uniform_sampling
from odl.tomo import (Parallel2dGeometry, FanFlatGeometry)
from odl.tomo.backends.astra_cpu import (astra_cpu_forward_projector_call,
                                         astra_cpu_backward_projector_call)
from odl.tomo.util.testutils import skip_if_no_astra
# from odl.util.testutils import all_equal, is_subdict


# TODO: test other interpolations once implemented
# TODO: move larger tests to examples

@skip_if_no_astra
def test_astra_cpu_projector_call_2d():
    """ASTRA CPU forward and back- projection."""

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
        data.show('imshow',
                  saveto='{}{}.png'.format(folder, name.replace(' ', '_')),
                  title='{} [:,:]'.format(name))

    # DiscreteLp element
    nvoxels = (100, 110)
    discr_vol_space = uniform_discr([-1, -1.1], [1, 1.1], nvoxels,
                                    dtype='float32')
    phantom = np.zeros(nvoxels)
    phantom[20:30, 20:30] = 1
    discr_data = discr_vol_space.element(phantom)
    save_slice(discr_data, back_dir, 'phantom 2d cpu')

    # motion and detector parameters, and geometry
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 180, as_midp=False)
    dparams = Interval(-2, 2)
    det_grid = uniform_sampling(dparams, 100)
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
    discr_proj_space = uniform_discr_fromspace(proj_space, npixels,
                                               dtype='float32')

    # Parallel 2D: forward
    p2_proj_data = astra_cpu_forward_projector_call(discr_data, geom_p2d,
                                                    discr_proj_space)
    save_slice(p2_proj_data, for_dir, 'parallel 2d cpu')

    # Fanflat: forward
    discr_data = discr_vol_space.element(phantom)
    ff_proj_data = astra_cpu_forward_projector_call(discr_data, geom_ff,
                                                    discr_proj_space)
    save_slice(ff_proj_data, for_dir, 'fanflat cpu')

    # Parallel 2D: backward
    reco_data = astra_cpu_backward_projector_call(p2_proj_data, geom_p2d,
                                                  discr_vol_space)
    save_slice(reco_data, back_dir, 'parallel 2d cpu')

    # Fanflat: backward
    reco_data = astra_cpu_backward_projector_call(ff_proj_data, geom_ff,
                                                  discr_vol_space)
    save_slice(reco_data, back_dir, 'fanflat cpu')


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
