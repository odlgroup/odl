# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test ASTRA back-end using CUDA."""

from __future__ import division
import numpy as np
import pytest

import odl
from odl.tomo.backends.astra_cuda import (
    AstraCudaProjectorImpl, AstraCudaBackProjectorImpl)
from odl.tomo.util.testutils import skip_if_no_astra_cuda
from odl.util.testutils import simple_fixture

# TODO: test with CUDA implemented uniform_discr


# --- pytest fixtures --- #


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


# --- CUDA projector tests --- #


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
