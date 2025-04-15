# Copyright 2014-2019 The ODL contributors
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
import torch 

import odl
from odl.tomo.backends.astra_cuda import AstraCudaImpl
from odl.tomo.util.testutils import skip_if_no_astra_cuda, skip_if_no_astra 
from odl.tomo.util.testfixtures import (projector, in_place, geometry, 
        parse_geometry, parse_angular_partition, 
        geometry_type, ray_trafo_impl, 
        consistent_geometry,
        PARALLEL_2D_PROJECTORS_CPU,
        PARALLEL_2D_PROJECTORS, PARALLEL_3D_PROJECTORS,
        CONE_2D_PROJECTORS, CONE_3D_PROJECTORS,
        HELICAL_PROJECTORS)


projectors = []
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_astra)
     for proj_cfg in PARALLEL_2D_PROJECTORS_CPU)
)
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_astra_cuda)
     for proj_cfg in  
        PARALLEL_2D_PROJECTORS + PARALLEL_3D_PROJECTORS + \
        CONE_2D_PROJECTORS + CONE_3D_PROJECTORS + \
        HELICAL_PROJECTORS)
)


projector_ids = [
    " geometry='{}' - dimension='{}' - ray_trafo_impl='{}' - reco_space_impl='{}' - angles='{}' - device='{}'".format(*p.values[0].split())
    for p in projectors
]

projector = pytest.fixture(fixture_function=projector, params=projectors, ids=projector_ids)

# def space_and_geometry(request):
#     dtype = 'float32'
#     geom = request.param

#     apart = odl.uniform_partition(0, 2 * np.pi, 8)

#     if geom == 'par2d':
#         reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5),
#                                        dtype=dtype, impl='pytorch')
#         dpart = odl.uniform_partition(-6, 6, 6)
#         geom = odl.tomo.Parallel2dGeometry(apart, dpart)
#     elif geom == 'par3d':
#         reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
#                                        dtype=dtype, impl='pytorch')
#         dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
#         geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
#     elif geom == 'cone2d':
#         reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5),
#                                        dtype=dtype, impl='pytorch')
#         dpart = odl.uniform_partition(-6, 6, 6)
#         geom = odl.tomo.FanBeamGeometry(apart, dpart, src_radius=100,
#                                         det_radius=10)
#     elif geom == 'cone3d':
#         reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
#                                        dtype=dtype, impl='pytorch')
#         dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))

#         geom = odl.tomo.ConeBeamGeometry(apart, dpart,
#                                          src_radius=200, det_radius=100)
#     elif geom == 'helical':
#         reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
#                                        dtype=dtype, impl='pytorch')

#         # overwrite angle
#         apart = odl.uniform_partition(0, 2 * 2 * np.pi, 18)
#         dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
#         geom = odl.tomo.ConeBeamGeometry(apart, dpart, pitch=1.0,
#                                          src_radius=200, det_radius=100)
#     else:
#         raise ValueError('geom not valid')

#     return reco_space, geom


# --- CUDA projector tests --- #


def test_astra_cuda_projector(projector):
    """Test ASTRA CUDA projector."""

    phantom = odl.phantom.cuboid(projector.domain)

    # Forward evaluation
    proj_data = projector(phantom)
    assert proj_data in projector.adjoint.domain
    assert proj_data.norm() > 0
    data = proj_data.asarray()
    if isinstance(data, np.ndarray):
        assert np.all(data >= 0)
    else:
        assert torch.all(data >= 0)

    # Backward evaluation
    backproj = projector.adjoint(proj_data)
    assert backproj in projector.domain
    assert backproj.norm() > 0
    data = backproj.asarray()
    if isinstance(data, np.ndarray):
        assert np.all(data >= 0)
    else:
        assert torch.all(data >= 0)



if __name__ == '__main__':
    odl.util.test_file(__file__)
