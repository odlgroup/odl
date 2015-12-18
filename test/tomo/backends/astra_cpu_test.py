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

# Internal
from odl.set.domain import Interval
from odl.space.fspace import FunctionSpace
from odl.discr.lp_discr import uniform_discr, uniform_discr_fromspace
from odl.discr.grid import uniform_sampling
from odl.tomo.geometry.parallel import Parallel2dGeometry
from odl.tomo.geometry.fanbeam import FanFlatGeometry
from odl.tomo.backends.astra_setup import ASTRA_AVAILABLE
if ASTRA_AVAILABLE:
    from odl.tomo.backends.astra_cpu import (astra_cpu_forward_projector_call,
                                             astra_cpu_backward_projector_call)
from odl.tomo.util.testutils import skip_if_no_astra


# TODO: test other interpolations once implemented

@skip_if_no_astra
def test_astra_cpu_projector_call_2d():
    """ASTRA CPU forward and back projection."""

    # `DiscreteLp` space for volume data
    nvoxels = (4, 5)
    discr_vol_space = uniform_discr([-4, -5], [4, 5], nvoxels,
                                    dtype='float32')

    # Phantom data
    phantom = np.zeros(nvoxels)
    phantom[1, 1] = 1

    # Create an element in the `DiscreteLp` space
    discr_data = discr_vol_space.element(phantom)

    # Angles
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 8, as_midp=False)

    # Detector
    dparams = Interval(-6, 6)
    det_grid = uniform_sampling(dparams, 6)

    # Distances for fanflat geometries
    src_rad = 100
    det_rad = 10

    # 2D geometry instances for parallel and fan beam with flat line detector
    geom_p2d = Parallel2dGeometry(angle_intvl, dparams, angle_grid,
                                  det_grid, angle_offset)
    geom_ff = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              angle_grid, det_grid, angle_offset)

    # Projection space
    proj_space = FunctionSpace(geom_p2d.params)

    # `DiscreteLp` projection space
    npixels = (angle_grid.ntotal, det_grid.ntotal)
    discr_proj_space = uniform_discr_fromspace(proj_space, npixels,
                                               dtype='float32')

    # Forward and back projections

    # Parallel 2D: forward
    proj_data_p2 = astra_cpu_forward_projector_call(discr_data, geom_p2d,
                                                    discr_proj_space)
    assert proj_data_p2.shape == npixels
    assert proj_data_p2.norm() > 0

    # Parallel 2D: backward
    reco_data_p2 = astra_cpu_backward_projector_call(proj_data_p2, geom_p2d,
                                                     discr_vol_space)
    assert reco_data_p2.shape == nvoxels
    assert reco_data_p2.norm() > 0

    # Fanflat: forward
    discr_data = discr_vol_space.element(phantom)
    proj_data_ff = astra_cpu_forward_projector_call(discr_data, geom_ff,
                                                    discr_proj_space)
    assert proj_data_ff.shape == npixels
    assert proj_data_ff > 0

    # Fanflat: backward
    reco_data_ff = astra_cpu_backward_projector_call(proj_data_ff, geom_ff,
                                                     discr_vol_space)
    assert reco_data_ff.shape == nvoxels
    assert reco_data_ff.norm() > 0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
