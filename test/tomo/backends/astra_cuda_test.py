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

"""Test ASTRA backend using CUDA."""

from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
from odl.tomo.backends import ASTRA_CUDA_AVAILABLE
from odl.set.domain import Interval, Rectangle
from odl.space.fspace import FunctionSpace
from odl.discr.lp_discr import uniform_discr, uniform_discr_fromspace
from odl.discr.grid import uniform_sampling
from odl.tomo.geometry.parallel import Parallel2dGeometry, Parallel3dGeometry
from odl.tomo.geometry.fanbeam import FanFlatGeometry
from odl.tomo.geometry.conebeam import (CircularConeFlatGeometry,
                                        HelicalConeFlatGeometry)
if ASTRA_CUDA_AVAILABLE:
    from odl.tomo.backends.astra_cuda import (
        astra_gpu_forward_projector_call, astra_gpu_backward_projector_call)
from odl.tomo.util.testutils import skip_if_no_astra_cuda


# TODO: better tests

@skip_if_no_astra_cuda
def test_astra_gpu_projector_call_2d():
    """2D forward and backward projectors on the GPU."""

    # Create `DiscreteLp` space for volume data
    nvoxels = (4, 5)
    discr_vol_space = uniform_discr([-4, -5], [4, 5], nvoxels,
                                    dtype='float32')

    # Phantom data
    phantom = np.zeros(nvoxels)
    phantom[1, 1] = 1

    # Create an element in the volume space
    discr_vol_data = discr_vol_space.element(phantom)

    # Angles
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 8, as_midp=False)

    # Detector
    dparams = Interval(-6, 6)
    det_grid = uniform_sampling(dparams, 6)

    # Distances for fanflat geometry
    src_rad = 100
    det_rad = 10

    # Create geometry instances
    geom_p2d = Parallel2dGeometry(angle_intvl, dparams, angle_grid, det_grid)
    geom_ff = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              angle_grid, det_grid, angle_offset)

    # Projection space
    proj_rect = angle_intvl.insert(dparams, 1)
    proj_space = FunctionSpace(proj_rect)

    # `DiscreteLp` projection space
    npixels = (angle_grid.ntotal, det_grid.ntotal)
    discr_proj_space = uniform_discr_fromspace(proj_space, npixels,
                                               dtype='float32')

    # Forward and back projections

    # PARALLEL 2D: forward
    proj_data_p2d = astra_gpu_forward_projector_call(discr_vol_data, geom_p2d,
                                                     discr_proj_space)
    assert proj_data_p2d.shape == npixels
    assert proj_data_p2d.norm() > 0

    # PARALLEL 2D: backward
    reco_data_p2d = astra_gpu_backward_projector_call(proj_data_p2d, geom_p2d,
                                                      discr_vol_space)
    assert reco_data_p2d.shape == nvoxels
    assert reco_data_p2d.norm() > 0

    # Fanflat: forward
    discr_vol_data = discr_vol_space.element(phantom)
    proj_data_ff = astra_gpu_forward_projector_call(discr_vol_data, geom_ff,
                                                    discr_proj_space)
    assert proj_data_ff.shape == npixels
    assert proj_data_ff.norm() > 0

    # Fanflat: backward
    reco_data_ff = astra_gpu_backward_projector_call(proj_data_ff, geom_ff,
                                                     discr_vol_space)
    assert reco_data_ff.shape == nvoxels
    assert reco_data_ff.norm() > 0


@skip_if_no_astra_cuda
def test_astra_gpu_projector_call_3d():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    vol_shape = (4, 5, 6)
    discr_vol_space = uniform_discr([-4, -5, -6], [4, 5, 6],
                                    vol_shape, dtype='float32')

    # Phantom
    phan = np.zeros(vol_shape)
    phan[1, 1:3, 1:4] = 1

    # Create an element in the volume space
    discr_data = discr_vol_space.element(phan)

    # Angles
    angle_offset = 0
    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 9, as_midp=False)

    # Detector
    dparams = Rectangle([-7, -8], [7, 8])
    det_grid = uniform_sampling(dparams, (7, 8))

    # Parameter for cone beam geometries
    src_rad = 1000
    det_rad = 100
    spiral_pitch_factor = 0.5

    # Create geometries
    geom_p3d = Parallel3dGeometry(angle_intvl, dparams, angle_grid, det_grid)
    geom_ccf = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                        angle_grid, det_grid, angle_offset)
    geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                       det_rad, spiral_pitch_factor,
                                       angle_grid, det_grid, angle_offset)

    # Projection space
    proj_rect = angle_intvl.insert(dparams, 1)
    proj_space = FunctionSpace(proj_rect)

    # `DiscreteLp` projection space
    proj_shape = (angle_grid.ntotal, det_grid.shape[0], det_grid.shape[1])


    discr_proj_space = uniform_discr_fromspace(proj_space, proj_shape,
                                               dtype='float32')

    # Forward: Parallel 3D
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_p3d,
                                                 discr_proj_space)
    assert proj_data.norm() > 0

    # Backward: Parallel 3D
    rec_data = astra_gpu_backward_projector_call(proj_data, geom_p3d,
                                                 discr_vol_space)
    assert rec_data.norm() > 0

    # Forward: Circular Cone Flat
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_ccf,
                                                 discr_proj_space)
    assert proj_data.norm() > 0

    # Backward: Circular Cone Flat
    rec_data = astra_gpu_backward_projector_call(proj_data, geom_ccf,
                                                 discr_vol_space)
    assert rec_data.norm() > 0

    # Forward: Helical Cone Flat
    proj_data = astra_gpu_forward_projector_call(discr_data, geom_hcf,
                                                 discr_proj_space)
    assert proj_data.norm() > 0

    # Backward: Helical Cone Flat
    rec_data = astra_gpu_backward_projector_call(proj_data, geom_hcf,
                                                 discr_vol_space)
    assert rec_data.norm() > 0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
