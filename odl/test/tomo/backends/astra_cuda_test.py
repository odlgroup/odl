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

import odl
from odl.tomo.backends.astra_cuda import AstraCudaImpl
from odl.tomo.util.testutils import skip_if_no_astra_cuda


# --- pytest fixtures --- #


# Find the valid projectors
projectors = [
    pytest.param(value, marks=skip_if_no_astra_cuda)
    for value in ['par2d', 'cone2d', 'par3d', 'cone3d', 'helical']
]

space_and_geometry_ids = [
    " geom='{}' ".format(p.values[0]) for p in projectors
]


@pytest.fixture(scope="module", params=projectors, ids=space_and_geometry_ids)
def space_and_geometry(request, odl_impl_device_pairs):
    impl, device = odl_impl_device_pairs
    dtype = 'float32'
    geom = request.param

    apart = odl.uniform_partition(0, 2 * np.pi, 8)

    if geom == 'par2d':
        reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5),
                                       dtype=dtype, impl=impl, device=device)
        dpart = odl.uniform_partition(-6, 6, 6)
        geom = odl.tomo.Parallel2dGeometry(apart, dpart)
    elif geom == 'par3d':
        reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                       dtype=dtype, impl=impl, device=device)
        dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
        geom = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    elif geom == 'cone2d':
        reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5),
                                       dtype=dtype, impl=impl, device=device)
        dpart = odl.uniform_partition(-6, 6, 6)
        geom = odl.tomo.FanBeamGeometry(apart, dpart, src_radius=100,
                                        det_radius=10)
    elif geom == 'cone3d':
        reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                       dtype=dtype, impl=impl, device=device)
        dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))

        geom = odl.tomo.ConeBeamGeometry(apart, dpart,
                                         src_radius=200, det_radius=100)
    elif geom == 'helical':
        reco_space = odl.uniform_discr([-4, -5, -6], [4, 5, 6], (4, 5, 6),
                                       dtype=dtype, impl=impl, device=device)

        # overwrite angle
        apart = odl.uniform_partition(0, 2 * 2 * np.pi, 18)
        dpart = odl.uniform_partition([-7, -8], [7, 8], (7, 8))
        geom = odl.tomo.ConeBeamGeometry(apart, dpart, pitch=1.0,
                                         src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')

    return reco_space, geom


# --- CUDA projector tests --- #


def test_astra_cuda_projector(space_and_geometry):
    """Test ASTRA CUDA projector."""

    # Create reco space and a phantom
    vol_space, geom = space_and_geometry
    phantom = odl.phantom.cuboid(vol_space)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(
        geom.partition,
        dtype=vol_space.dtype_identifier, 
        impl=vol_space.impl,
        device=vol_space.device)

    # create RayTransform implementation
    astra_cuda = AstraCudaImpl(geom, vol_space, proj_space)

    out = astra_cuda.proj_space.zero()
    # Forward evaluation
    proj_data = astra_cuda.call_forward(phantom)
    assert proj_data in proj_space
    assert proj_data.norm() > 0
    assert odl.all(0 <= proj_data)

    # Backward evaluation
    backproj = astra_cuda.call_backward(proj_data)
    assert backproj in vol_space
    assert backproj.norm() > 0
    assert odl.all(0 <= backproj)


if __name__ == '__main__':
    odl.core.util.test_file(__file__)
