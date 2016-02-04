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
import odl
from odl.tomo.util.testutils import skip_if_no_astra


# TODO: clean up and improve tests

@skip_if_no_astra
def test_astra_cpu_projector_parallel2d():
    """ASTRA CPU forward and back projection for 2d parallel geometry."""

    # `DiscreteLp` space for volume data
    vol_shape = (5, 5)
    discr_vol_space = odl.uniform_discr([-5, -5], [5, 5], vol_shape,
                                        dtype='float32')

    # Create an element in the `DiscreteLp` space
    discr_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 8)

    # Detector
    dparams = odl.Interval(-6, 6)
    det_grid = odl.uniform_sampling(dparams, 6)

    # 2D geometry instances for parallel and fan beam with flat line detector
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, angle_grid,
                                       det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # forward
    proj_data = odl.tomo.astra_cpu_forward_projector(discr_data, geom,
                                                     discr_proj_space)
    assert proj_data.shape == proj_shape
    assert proj_data.norm() > 0

    # backward
    reco_data = odl.tomo.astra_cpu_back_projector(proj_data, geom,
                                                  discr_vol_space)
    assert reco_data.shape == vol_shape
    assert reco_data.norm() > 0


@skip_if_no_astra
def test_astra_cpu_projector_fanflat():
    """ASTRA CPU forward and back projection for fanflat geometry."""

    # `DiscreteLp` space for volume data
    vol_shape = (5, 5)
    discr_vol_space = odl.uniform_discr([-5, -5], [5, 5], vol_shape,
                                        dtype='float32')

    # Create an element in the `DiscreteLp` space
    discr_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 8)

    # Detector
    dparams = odl.Interval(-6, 6)
    det_grid = odl.uniform_sampling(dparams, 6)

    # Distances for fanflat geometries
    src_rad = 100
    det_rad = 10

    # 2D geometry instances for parallel and fan beam with flat line detector
    geom = odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                    angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # forward
    proj_data = odl.tomo.astra_cpu_forward_projector(discr_data, geom,
                                                     discr_proj_space)
    assert proj_data.shape == proj_shape
    assert proj_data.norm() > 0

    # backward
    reco_data = odl.tomo.astra_cpu_back_projector(proj_data, geom,
                                                  discr_vol_space)
    assert reco_data.shape == vol_shape
    assert reco_data.norm() > 0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
