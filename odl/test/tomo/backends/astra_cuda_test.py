# Copyright 2014-2016 The ODL development group
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
import sys

# Internal
import odl
from odl.tomo.backends.astra_cuda import (
    astra_cuda_forward_projector, astra_cuda_back_projector)
from odl.tomo.util.testutils import skip_if_no_astra_cuda


# TODO: test with CUDA implemented uniform_discr


@skip_if_no_astra_cuda
def test_astra_cuda_projector_parallel2d():
    """Parallel 2D forward and backward projectors on the GPU."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5), dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0], max_pt=[4, 5])

    # Create parallel geometry
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition(-6, 6, 6)
    geom = odl.tomo.Parallel2dGeometry(angle_part, det_part)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cuda_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cuda_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_fanflat():
    """Fanflat 2D forward and backward projectors on the GPU."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5), dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0], max_pt=[4, 5])

    # Create fan beam geometry with flat detector
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition(-6, 6, 6)
    src_rad = 100
    det_rad = 10
    geom = odl.tomo.FanFlatGeometry(angle_part, det_part, src_rad, det_rad)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cuda_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cuda_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


@pytest.mark.xfail(sys.platform == 'win32', strict=True,
                   reason="Fails on windows")
@skip_if_no_astra_cuda
def test_astra_cuda_projector_parallel3d():
    """Test 3D forward and backward projection functions on the GPU."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                   dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0, 0],
                                 max_pt=[4, 5, 6])

    # Create parallel geometry
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
    geom = odl.tomo.Parallel3dAxisGeometry(angle_part, det_part)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cuda_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cuda_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


@skip_if_no_astra_cuda
def test_astra_gpu_projector_circular_conebeam():
    """Test 3D forward and backward projection functions on the GPU."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                   dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0, 0],
                                 max_pt=[4, 5, 6])

    # Create circular cone beam geometry with flat detector
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
    src_rad = 1000
    det_rad = 100
    geom = odl.tomo.CircularConeFlatGeometry(angle_part, det_part, src_rad,
                                             det_rad)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cuda_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cuda_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


@skip_if_no_astra_cuda
def test_astra_cuda_projector_helical_conebeam():
    """Test 3D forward and backward projection functions on the GPU."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                   dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0, 0],
                                 max_pt=[4, 5, 6])

    # Create circular cone beam geometry with flat detector
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
    src_rad = 1000
    det_rad = 100
    pitch = 0.5
    geom = odl.tomo.HelicalConeFlatGeometry(angle_part, det_part, src_rad,
                                            det_rad, pitch)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cuda_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cuda_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
