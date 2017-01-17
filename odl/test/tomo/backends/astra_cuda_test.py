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
    AstraCudaProjectorImpl, AstraCudaBackProjectorImpl)
from odl.tomo.util.testutils import skip_if_no_astra_cuda
from odl.util.testutils import simple_fixture


# TODO: test with CUDA implemented uniform_discr


use_cache = simple_fixture('use_cache', [False, True])

# Find the valid projectors
projectors = [skip_if_no_astra_cuda('par2d'),
              skip_if_no_astra_cuda('cone2d'),
              skip_if_no_astra_cuda('par3d'),
              skip_if_no_astra_cuda('cone3d'),
              skip_if_no_astra_cuda('helical')]


space_and_geometry_ids = ['geom = {}'.format(p.args[1]) for p in projectors]


@pytest.fixture(scope="module", params=projectors, ids=space_and_geometry_ids)
def space_and_geometry(request):
    dtype = 'float32'
    geom = request.param

    apart = odl.uniform_partition(0, 2 * np.pi, 8)

    if geom == 'par2d':
        reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5),
                                       dtype=dtype)
        dpart = odl.uniform_partition(-6, 6, 6)
        geom = odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geom == 'par3d':
        reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                       dtype=dtype)
        dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
        geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    elif geom == 'cone2d':
        reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5),
                                       dtype=dtype)
        dpart = odl.uniform_partition(-6, 6, 6)
        geom = odl.tomo.FanFlatGeometry(apart, dpart, src_radius=100,
                                        det_radius=10)
    elif geom == 'cone3d':
        reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                       dtype=dtype)
        dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))

        geom = odl.tomo.CircularConeFlatGeometry(apart, dpart, src_radius=200,
                                                 det_radius=100)
    elif geom == 'helical':
        reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                       dtype=dtype)

        # overwrite angle
        apart = odl.uniform_partition(0, 2 * 2 * np.pi, 18)
        dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
        geom = odl.tomo.HelicalConeFlatGeometry(apart, dpart, pitch=1.0,
                                                src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')

    return reco_space, geom


def test_astra_cuda_projector(space_and_geometry, use_cache):
    """Test ASTRA CUDA projector."""

    # Create reco space and a phantom
    reco_space, geom = space_and_geometry
    phantom = odl.phantom.cuboid(reco_space)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype=reco_space.dtype)

    # Forward evaluation
    projector = AstraCudaProjectorImpl(geom, reco_space, proj_space,
                                       use_cache=use_cache)
    proj_data = projector.call_forward(phantom)
    assert proj_data in proj_space
    assert proj_data.norm() > 0
    assert np.all(proj_data.asarray() >= 0)

    # Backward evaluation
    back_projector = AstraCudaBackProjectorImpl(geom, reco_space, proj_space,
                                                use_cache=use_cache)
    backproj = back_projector.call_backward(proj_data)
    assert backproj in reco_space
    assert backproj.norm() > 0
    assert np.all(proj_data.asarray() >= 0)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
