# Copyright 2014-2025 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tests for the Filtered Back-Projection."""

import numpy as np
import pytest

import odl
from odl.applications.tomo.util.testutils import (
    skip_if_no_astra, skip_if_no_astra_cuda, skip_if_no_skimage, skip_if_no_pytorch)
from odl.core.util.testutils import simple_fixture

# --- pytest fixtures --- #

### The threshold values of L2-norms and element-wise maximum differences
### are measured with the "m" and "n_angles" parameters below.
### If those are changed, the tests probably won't pass.
m = 500
n_angles = 1000

pytestmark = pytest.mark.suite('largescale')

impl = simple_fixture(
    name='impl',
    params=[pytest.param('astra_cpu', marks=skip_if_no_astra),
            pytest.param('astra_cuda', marks=skip_if_no_astra_cuda),
            pytest.param('skimage', marks=skip_if_no_skimage)]
)

geometry_params = ['par2d', 'par3d', 'cone2d', 'cone3d', 'helical']
geometry_ids = [" geometry='{}' ".format(p) for p in geometry_params]


@pytest.fixture(scope='module', ids=geometry_ids, params=geometry_params)
def geometry(request):
    geom = request.param

    if geom == 'par3d':
        apart = odl.uniform_partition(0, np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -30], [30, 30], (m, m))
        return odl.applications.tomo.Parallel3dAxisGeometry(apart, dpart)
    elif geom == 'cone3d':
        apart = odl.uniform_partition(0, 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-60, -60], [60, 60], (m, m))
        return odl.applications.tomo.ConeBeamGeometry(apart, dpart,
                                         src_radius=200, det_radius=100)
    elif geom == 'helical':
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], (m, m))
        return odl.applications.tomo.ConeBeamGeometry(apart, dpart, pitch=5.0,
                                         src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')
    
geometry_type = simple_fixture(
    'geometry_type',
    ['par2d', 'par3d', 'cone2d', 'cone3d']
)

### The last two numbers are "hand-picked" by looking the FBP reconstructions
### for different scenarios. The first number is the L2-norm of the difference
### between Shepp-Logan phantom and the FBP reconstruction.
### The second number is maximum distance between element-wise comparison of 
### the Shepp-Logan phantom and the FBP reconstruction.
projectors = []
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_astra_cuda)
     for proj_cfg in [
                      'par3d astra_cuda numpy cpu Ram-Lak 3.46 0.20',
                      'par3d astra_cuda numpy cpu Shepp-Logan 4.01 0.19',
                      'par3d astra_cuda numpy cpu Cosine 5.68 0.20',
                      'par3d astra_cuda numpy cpu Hamming 6.89 0.23',
                      'par3d astra_cuda numpy cpu Hann 7.25 0.23',
                      'cone3d astra_cuda numpy cpu Ram-Lak 5.41 0.23',
                      'cone3d astra_cuda numpy cpu Shepp-Logan 6.24 0.23',
                      'cone3d astra_cuda numpy cpu Cosine 8.10 0.26',
                      'cone3d astra_cuda numpy cpu Hamming 9.60 0.30',
                      'cone3d astra_cuda numpy cpu Hann 10.00 0.32',
                      'helical astra_cuda numpy cpu Ram-Lak 42.68 0.72',
                      'helical astra_cuda numpy cpu Shepp-Logan 41.50 0.69',
                      'helical astra_cuda numpy cpu Cosine 39.54 0.62',
                      'helical astra_cuda numpy cpu Hamming 38.45 0.58',
                      'helical astra_cuda numpy cpu Hann 38.16 0.56'
                      ])
)

projectors.extend(
    (pytest.param(proj_cfg, marks=[skip_if_no_astra, skip_if_no_pytorch])
     for proj_cfg in [
                      'par3d astra_cuda pytorch cuda:0 Ram-Lak 3.46 0.20',
                      'par3d astra_cuda pytorch cuda:0 Shepp-Logan 4.01 0.19',
                      'par3d astra_cuda pytorch cuda:0 Cosine 5.68 0.20',
                      'par3d astra_cuda pytorch cuda:0 Hamming 6.87 0.23',
                      'par3d astra_cuda pytorch cuda:0 Hann 7.25 0.23',
                      'cone3d astra_cuda pytorch cuda:0 Ram-Lak 5.41 0.23',
                      'cone3d astra_cuda pytorch cuda:0 Shepp-Logan 6.24 0.23',
                      'cone3d astra_cuda pytorch cuda:0 Cosine 8.01 0.26',
                      'cone3d astra_cuda pytorch cuda:0 Hamming 9.60 0.30',
                      'cone3d astra_cuda pytorch cuda:0 Hann 10.00 0.32',
                      'helical astra_cuda pytorch cuda:0 Ram-Lak 42.68 0.72',
                      'helical astra_cuda pytorch cuda:0 Shepp-Logan 41.50 0.69',
                      'helical astra_cuda pytorch cuda:0 Cosine 39.54 0.62',
                      'helical astra_cuda pytorch cuda:0 Hamming 38.45 0.58',
                      'helical astra_cuda pytorch cuda:0 Hann 38.16 0.56'])
)



projector_ids = [
    " geom='{}' - astra_impl='{}' - tspace_impl='{}' - tspace_device='{}' - filter='{}' - L2_value='{}' - max_dist='{}'".format(*p.values[0].split())
    for p in projectors
]


@pytest.fixture(scope='module', params=projectors, ids=projector_ids)
def projector(request):
    ### Change of the "n" variable can affect the passing requirements.
    n = 100
    dtype = 'float32'
    geom, astra_impl, tspace_impl, tspace_device, filter, L2_value, max_dist = request.param.split()
    
    apart = odl.uniform_partition(0, 2 * np.pi, n_angles)

    if geom == 'par2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype, impl=tspace_impl, device=tspace_device)
        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = odl.applications.tomo.Parallel2dGeometry(apart, dpart)

    elif geom == 'par3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype, impl=tspace_impl, device=tspace_device)

        # Geometry
        dpart = odl.uniform_partition([-30] * 2, [30] * 2, [m] * 2)
        geom = odl.applications.tomo.Parallel3dAxisGeometry(apart, dpart)

    elif geom == 'cone2d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 2, [20] * 2, [n] * 2,
                                       dtype=dtype, impl=tspace_impl, device=tspace_device)
        # Geometry
        dpart = odl.uniform_partition(-30, 30, m)
        geom = odl.applications.tomo.FanBeamGeometry(apart, dpart, src_radius=200,
                                        det_radius=100)

    elif geom == 'cone3d':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20] * 3, [20] * 3, [n] * 3,
                                       dtype=dtype, impl=tspace_impl, device=tspace_device)
        # Geometry
        dpart = odl.uniform_partition([-60] * 2, [60] * 2, [m] * 2)
        geom = odl.applications.tomo.ConeBeamGeometry(apart, dpart,
                                         src_radius=200, det_radius=100)

    elif geom == 'helical':
        # Reconstruction space
        reco_space = odl.uniform_discr([-20, -20, 0], [20, 20, 40],
                                       [n] * 3, dtype=dtype, impl=tspace_impl, device=tspace_device)
        # Geometry, overwriting angle partition
        apart = odl.uniform_partition(0, 8 * 2 * np.pi, n_angles)
        dpart = odl.uniform_partition([-30, -3], [30, 3], [m] * 2)
        geom = odl.applications.tomo.ConeBeamGeometry(apart, dpart, pitch=5.0,
                                         src_radius=200, det_radius=100)
    else:
        raise ValueError('geom not valid')
    
    # Ray transform
    return {'ray_trafo': odl.applications.tomo.RayTransform(reco_space, geom, impl=astra_impl, use_cache=False),
            'filter_type': filter,
            'L2_norm': L2_value,
            'max_distance': max_dist}
    # return (odl.applications.tomo.RayTransform(reco_space, geom, impl=astra_impl, use_cache=False), filter, L2_value, max_dist)


@pytest.fixture(scope='module',
                params=[True, False],
                ids=[' in-place ', ' out-of-place '])
def in_place(request):
    return request.param

def test_fpb(in_place, projector):
    """Test Filtered Back-Projection"""
    
    ray_trafo = projector['ray_trafo']
    filter = projector['filter_type']
    L2_value = float(projector['L2_norm'])
    max_dist = float(projector['max_distance'])

    rtol_norm = 0.05
    abs_dist = 0.05
    
    phantom = odl.core.phantom.shepp_logan(ray_trafo.domain, modified=True)
    # Filtered Back-Projection with different filters.
    fbp = odl.applications.tomo.fbp_op(ray_trafo, filter_type=filter, frequency_scaling=1.0)

    if in_place:
        proj = ray_trafo.range.zero()
        backproj = fbp.range.zero()
        fbp(ray_trafo(phantom, out=proj), out=backproj)
    else:
        proj = ray_trafo(phantom)
        backproj = fbp(proj)
    
    L2Norm = odl.functionals.default_functionals.L2Norm(fbp.range)
    assert L2Norm(phantom-backproj) == pytest.approx(L2_value, rel=L2_value*rtol_norm)
    assert odl.max(phantom-backproj) == pytest.approx(max_dist, abs=abs_dist)


if __name__ == '__main__':
    odl.core.util.test_file(__file__, ['-S', 'largescale'])
