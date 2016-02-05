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

"""Test ASTRA back-end using CUDA."""

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


# TODO: clean up and improve tests

@skip_if_no_astra_cuda
def test_astra_cuda_projector_parallel2d():
    """Parallel 2D forward and backward projectors on the GPU."""

    # Create `DiscreteLp` space for volume data
    discr_vol_space = odl.uniform_discr([-4, -5], [4, 5], [4, 5],
                                        dtype='float32')

    # Create an element in the volume space
    vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_grid = odl.uniform_sampling(0, 2 * np.pi, 8)

    # Detector
    det_grid = odl.uniform_sampling(-6, 6, 6)

    # Create geometry instances
    geom = odl.tomo.Parallel2dGeometry(angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # forward
    proj_data = odl.tomo.astra_cuda_forward_projector(vol_data, geom,
                                                      discr_proj_space)
    assert proj_data.shape == proj_shape
    assert proj_data.norm() > 0

    # backward
    reco_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                   discr_vol_space)
    assert reco_data.shape == discr_vol_space.shape
    assert reco_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_fanflat():
    """Fanflat 2D forward and backward projectors on the GPU."""

    # Create `DiscreteLp` space for volume data
    discr_vol_space = odl.uniform_discr([-4, -5], [4, 5], [4, 5],
                                        dtype='float32')

    # Create an element in the volume space
    vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_grid = odl.uniform_sampling(0, 2 * np.pi, 8)

    # Detector
    det_grid = odl.uniform_sampling(-6, 6, 6)

    # Distances for fanflat geometry
    src_rad = 100
    det_rad = 10

    # Create geometry instances
    geom = odl.tomo.FanFlatGeometry(angle_grid, det_grid, src_rad, det_rad)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    proj_shape = geom.grid.shape
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, proj_shape,
                                                   dtype='float32')

    # forward
    proj_data = odl.tomo.astra_cuda_forward_projector(vol_data, geom,
                                                      discr_proj_space)
    assert proj_data.shape == proj_shape
    assert proj_data.norm() > 0

    # backward
    reco_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                   discr_vol_space)
    assert reco_data.shape == discr_vol_space.shape
    assert reco_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_parallel3d():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    discr_vol_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], [4, 5, 6],
                                        dtype='float32')

    # Create an element in the volume space
    vol_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_grid = odl.uniform_sampling(0, 2 * np.pi, 9)

    # Detector
    det_grid = odl.uniform_sampling([-7, -8], [7, 8], (7, 8))

    # Create geometries
    geom = odl.tomo.Parallel3dGeometry(angle_grid, det_grid)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32')

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(vol_data, geom,
                                                      discr_proj_space)
    assert proj_data.norm() > 0

    # Backward
    rec_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                  discr_vol_space)
    assert rec_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_gpu_projector_circular_conebeam():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    discr_vol_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], [4, 5, 6],
                                        dtype='float32')

    # Create an element in the volume space
    discr_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_grid = odl.uniform_sampling(0, 2 * np.pi, 9)

    # Detector
    det_grid = odl.uniform_sampling([-7, -8], [7, 8], (7, 8))

    # Parameter for cone beam geometries
    src_rad = 1000
    det_rad = 100

    # Create geometries
    geom = odl.tomo.CircularConeFlatGeometry(angle_grid, det_grid,
                                             src_rad, det_rad)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32')

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data, geom,
                                                      discr_proj_space)
    assert proj_data.norm() > 0

    # Backward
    rec_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                  discr_vol_space)
    assert rec_data.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_helical_conebeam():
    """Test 3D forward and backward projection functions on the GPU."""

    # `DiscreteLp` volume space
    discr_vol_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], [4, 5, 6],
                                        dtype='float32')

    # Create an element in the volume space
    discr_data = odl.util.phantom.cuboid(discr_vol_space, 0.5, 1)

    # Angles
    angle_grid = odl.uniform_sampling(0, 2 * np.pi, 9)

    # Detector
    det_grid = odl.uniform_sampling([-7, -8], [7, 8], (7, 8))

    # Parameter for cone beam geometries
    src_rad = 1000
    det_rad = 100
    pitch = 0.5

    # Create geometries
    geom = odl.tomo.HelicalConeFlatGeometry(angle_grid, det_grid,
                                            src_rad, det_rad, pitch)

    # Projection space
    proj_space = odl.FunctionSpace(geom.params)

    # `DiscreteLp` projection space
    discr_proj_space = odl.uniform_discr_fromspace(proj_space, geom.grid.shape,
                                                   dtype='float32')

    # Forward
    proj_data = odl.tomo.astra_cuda_forward_projector(discr_data, geom,
                                                      discr_proj_space)
    assert proj_data.norm() > 0

    # Backward
    rec_data = odl.tomo.astra_cuda_back_projector(proj_data, geom,
                                                  discr_vol_space)
    assert rec_data.norm() > 0


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
