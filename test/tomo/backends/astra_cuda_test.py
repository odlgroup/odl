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
import odl
from odl.tomo.backends.astra_cuda import ASTRA_CUDA_AVAILABLE
from odl.tomo.util.testutils import skip_if_no_astra_cuda


# TODO: test scaling of ASTRA projectors
# TODO: smarter tests

@skip_if_no_astra_cuda
def test_astra_cuda_projector_parallel2d():
    """Parallel 2D forward and backward projectors on the GPU."""

    # Create `DiscreteLp` space for volume data
    vol_shape = (4, 5)
    discr_vol_space = odl.uniform_discr([-4, -5], [4, 5], vol_shape,
                                        dtype='float32')

    # Create an element in the volume space
    vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 8)

    # Detector
    dparams = odl.Interval(-6, 6)
    det_grid = odl.uniform_sampling(dparams, 6)

    # Create geometry instances
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, angle_grid,
                                       det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(vol_data, geom,
                                                           discr_proj_space)
    assert proj_data.shape == proj_shape
    assert proj_data.norm() > 0

    # backward
    reco_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                            discr_vol_space)
    assert reco_data.shape == vol_shape
    assert reco_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_fanflat():
    """Fanflat 2D forward and backward projectors on the GPU."""

    # Create `DiscreteLp` space for volume data
    vol_shape = (4, 5)
    discr_vol_space = odl.uniform_discr([-4, -5], [4, 5], vol_shape,
                                        dtype='float32')

    # Create an element in the volume space
    vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 8)

    # Detector
    dparams = odl.Interval(-6, 6)
    det_grid = odl.uniform_sampling(dparams, 6)

    # Distances for fanflat geometry
    src_rad = 100
    det_rad = 10

    # Create geometry instances
    geom = odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                    angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(vol_data, geom,
                                                           discr_proj_space)
    assert proj_data.shape == proj_shape
    assert proj_data.norm() > 0

    # backward
    reco_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                            discr_vol_space)
    assert reco_data.shape == vol_shape
    assert reco_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_parallel3d():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    vol_shape = (4, 5, 6)
    discr_vol_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6],
                                        vol_shape, dtype='float32')

    # Create an element in the volume space
    vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 9)

    # Detector
    dparams = odl.Rectangle([-7, -8], [7, 8])
    det_grid = odl.uniform_sampling(dparams, (7, 8))

    # Create geometries
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, angle_grid,
                                       det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32')

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(vol_data, geom,
                                                           discr_proj_space)
    assert proj_data.norm() > 0

    # Backward
    rec_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)
    assert rec_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_gpu_projector_circular_conebeam():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    vol_shape = (4, 5, 6)
    discr_vol_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6],
                                        vol_shape, dtype='float32')

    # Phantom
    phan = np.zeros(vol_shape)
    phan[1, 1:3, 1:4] = 1

    # Create an element in the volume space
    discr_data = discr_vol_space.element(phan)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 9)

    # Detector
    dparams = odl.Rectangle([-7, -8], [7, 8])
    det_grid = odl.uniform_sampling(dparams, (7, 8))

    # Parameter for cone beam geometries
    src_rad = 1000
    det_rad = 100

    # Create geometries
    geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams, src_rad,
                                             det_rad, angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32')

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(discr_data, geom,
                                                           discr_proj_space)
    assert proj_data.norm() > 0

    # Backward
    rec_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)
    assert rec_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_helical_conebeam():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    vol_shape = (4, 5, 6)
    discr_vol_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6],
                                        vol_shape, dtype='float32')

    # Create an element in the volume space
    discr_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_intvl = odl.Interval(0, 2 * np.pi)
    angle_grid = odl.uniform_sampling(angle_intvl, 9)

    # Detector
    dparams = odl.Rectangle([-7, -8], [7, 8])
    det_grid = odl.uniform_sampling(dparams, (7, 8))

    # Parameter for cone beam geometries
    src_rad = 1000
    det_rad = 100
    pitch_factor = 0.5

    # Create geometries
    geom = odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams, src_rad,
                                            det_rad,
                                            pitch=pitch_factor,
                                            agrid=angle_grid, dgrid=det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32')

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector_call(discr_data, geom,
                                                           discr_proj_space)
    assert proj_data.norm() > 0

    # Backward
    rec_data = odl.tomo.astra_cuda_backward_projector_call(proj_data, geom,
                                                           discr_vol_space)
    assert rec_data.norm() > 0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
