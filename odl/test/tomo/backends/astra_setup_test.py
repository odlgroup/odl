# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test astra setup functions."""

from __future__ import division
import numpy as np
import pytest
try:
    import astra
except ImportError:
    pass

import odl
from odl.tomo.backends.astra_setup import astra_supports
from odl.util.testutils import is_subdict


pytestmark = pytest.mark.skipif("not odl.tomo.ASTRA_AVAILABLE")


def _discrete_domain(ndim, interp):
    """Create `DiscreteLp` space with isotropic grid stride.

    Parameters
    ----------
    ndim : `int`
        Number of space dimensions
    interp : `str`
        Interpolation scheme

    Returns
    -------
    space : `DiscreteLp`
        Returns a `DiscreteLp` instance
    """
    max_pt = np.arange(1, ndim + 1)
    min_pt = - max_pt
    shape = np.arange(1, ndim + 1) * 10

    return odl.uniform_discr(min_pt, max_pt, shape=shape, interp=interp,
                             dtype='float32')


def _discrete_domain_anisotropic(ndim, interp):
    """Create `DiscreteLp` space with anisotropic grid stride.

    Parameters
    ----------
    ndim : `int`
        Number of space dimensions
    interp : `str`
        Interpolation scheme

    Returns
    -------
    space : `DiscreteLp`
        Returns a `DiscreteLp` instance
    """
    min_pt = [-1] * ndim
    max_pt = [1] * ndim
    shape = np.arange(1, ndim + 1) * 10

    return odl.uniform_discr(min_pt, max_pt, shape=shape, interp=interp,
                             dtype='float32')


def test_vol_geom_2d():
    """Check correctness of ASTRA 2D volume geometries."""
    x_pts = 10  # x_pts = Rows
    y_pts = 20  # y_pts = Columns

    # Isotropic voxel case
    discr_dom = _discrete_domain(2, 'nearest')
    correct_dict = {
        'GridColCount': y_pts,
        'GridRowCount': x_pts,
        'option': {
            'WindowMinX': -2.0,  # y_min
            'WindowMaxX': 2.0,  # y_max
            'WindowMinY': -1.0,  # x_min
            'WindowMaxY': 1.0}}  # x_amx

    vol_geom = odl.tomo.astra_volume_geometry(discr_dom)
    assert vol_geom == correct_dict

    # Anisotropic voxel case
    discr_dom = _discrete_domain_anisotropic(2, 'nearest')
    correct_dict = {
        'GridColCount': y_pts,
        'GridRowCount': x_pts,
        'option': {
            'WindowMinX': -1.0,  # y_min
            'WindowMaxX': 1.0,  # y_max
            'WindowMinY': -1.0,  # x_min
            'WindowMaxY': 1.0}}  # x_amx

    if astra_supports('anisotropic_voxels_2d'):
        vol_geom = odl.tomo.astra_volume_geometry(discr_dom)
        assert vol_geom == correct_dict
    else:
        with pytest.raises(NotImplementedError):
            odl.tomo.astra_volume_geometry(discr_dom)


def test_vol_geom_3d():
    """Check correctness of ASTRA 3D volume geometies."""
    x_pts = 10
    y_pts = 20
    z_pts = 30

    # Isotropic voxel case
    discr_dom = _discrete_domain(3, 'nearest')
    # x = columns, y = rows, z = slices
    correct_dict = {
        'GridColCount': z_pts,
        'GridRowCount': y_pts,
        'GridSliceCount': x_pts,
        'option': {
            'WindowMinX': -3.0,  # z_min
            'WindowMaxX': 3.0,  # z_max
            'WindowMinY': -2.0,  # y_min
            'WindowMaxY': 2.0,  # y_amx
            'WindowMinZ': -1.0,  # x_min
            'WindowMaxZ': 1.0}}  # x_amx

    vol_geom = odl.tomo.astra_volume_geometry(discr_dom)
    assert vol_geom == correct_dict

    discr_dom = _discrete_domain_anisotropic(3, 'nearest')
    # x = columns, y = rows, z = slices
    correct_dict = {
        'GridColCount': z_pts,
        'GridRowCount': y_pts,
        'GridSliceCount': x_pts,
        'option': {
            'WindowMinX': -1.0,  # z_min
            'WindowMaxX': 1.0,  # z_max
            'WindowMinY': -1.0,  # y_min
            'WindowMaxY': 1.0,  # y_amx
            'WindowMinZ': -1.0,  # x_min
            'WindowMaxZ': 1.0}}  # x_amx

    if astra_supports('anisotropic_voxels_3d'):
        vol_geom = odl.tomo.astra_volume_geometry(discr_dom)
        assert vol_geom == correct_dict
    else:
        with pytest.raises(NotImplementedError):
            odl.tomo.astra_volume_geometry(discr_dom)


def test_proj_geom_parallel_2d():
    """Create ASTRA 2D projection geometry."""

    apart = odl.uniform_partition(0, 2, 5)
    dpart = odl.uniform_partition(-1, 1, 10)
    geom = odl.tomo.Parallel2dGeometry(apart, dpart)

    proj_geom = odl.tomo.astra_projection_geometry(geom)

    correct_subdict = {
        'type': 'parallel',
        'DetectorCount': 10, 'DetectorWidth': 0.2}

    assert is_subdict(correct_subdict, proj_geom)
    assert 'ProjectionAngles' in proj_geom


def test_astra_projection_geometry():
    """Create ASTRA projection geometry from geometry objects."""

    with pytest.raises(TypeError):
        odl.tomo.astra_projection_geometry(None)

    apart = odl.uniform_partition(0, 2 * np.pi, 5)
    dpart = odl.uniform_partition(-40, 40, 10)

    # motion sampling grid, detector sampling grid but not uniform
    dpart_0 = odl.RectPartition(odl.IntervalProd(0, 3),
                                odl.RectGrid([0, 1, 3]))
    geom_p2d = odl.tomo.Parallel2dGeometry(apart, dpart=dpart_0)
    with pytest.raises(ValueError):
        odl.tomo.astra_projection_geometry(geom_p2d)

    # detector sampling grid, motion sampling grid
    geom_p2d = odl.tomo.Parallel2dGeometry(apart, dpart)
    odl.tomo.astra_projection_geometry(geom_p2d)

    # Parallel 2D geometry
    geom_p2d = odl.tomo.Parallel2dGeometry(apart, dpart)
    astra_geom = odl.tomo.astra_projection_geometry(geom_p2d)
    assert astra_geom['type'] == 'parallel'

    # Fan flat
    src_rad = 10
    det_rad = 5
    geom_ff = odl.tomo.FanFlatGeometry(apart, dpart, src_rad, det_rad)
    astra_geom = odl.tomo.astra_projection_geometry(geom_ff)
    assert astra_geom['type'] == 'fanflat_vec'

    dpart = odl.uniform_partition([-40, -3], [40, 3], (10, 5))

    # Parallel 3D geometry
    geom_p3d = odl.tomo.Parallel3dAxisGeometry(apart, dpart)
    odl.tomo.astra_projection_geometry(geom_p3d)
    astra_geom = odl.tomo.astra_projection_geometry(geom_p3d)
    assert astra_geom['type'] == 'parallel3d_vec'

    # Circular conebeam flat
    geom_ccf = odl.tomo.ConeFlatGeometry(apart, dpart, src_rad, det_rad)
    astra_geom = odl.tomo.astra_projection_geometry(geom_ccf)
    assert astra_geom['type'] == 'cone_vec'

    # Helical conebeam flat
    pitch = 1
    geom_hcf = odl.tomo.ConeFlatGeometry(apart, dpart, src_rad, det_rad,
                                         pitch=pitch)
    astra_geom = odl.tomo.astra_projection_geometry(geom_hcf)
    assert astra_geom['type'] == 'cone_vec'


VOL_GEOM_2D = {
    'GridColCount': 20, 'GridRowCount': 10,
    'option': {'WindowMinX': -2.0, 'WindowMaxX': 2.0,
               'WindowMinY': -1.0, 'WindowMaxY': 1.0}}


def test_volume_data_2d():
    """Create ASTRA data structure in 2D."""

    # From scratch
    data_id = odl.tomo.astra_data(VOL_GEOM_2D, 'volume', ndim=2)
    data_out = astra.data2d.get_shared(data_id)
    assert data_out.shape == (10, 20)

    # From existing
    discr_dom = _discrete_domain(2, 'nearest')
    data_in = discr_dom.element(np.ones(10 * 20, dtype='float32'))
    data_id = odl.tomo.astra_data(VOL_GEOM_2D, 'volume', data=data_in)
    data_out = astra.data2d.get_shared(data_id)
    assert data_out.shape == (10, 20)


VOL_GEOM_3D = {
    'GridColCount': 30, 'GridRowCount': 20, 'GridSliceCount': 10,
    'option': {}}


def test_volume_data_3d():
    """Create ASTRA data structure in 2D."""

    # From scratch
    data_id = odl.tomo.astra_data(VOL_GEOM_3D, 'volume', ndim=3)
    data_out = astra.data3d.get_shared(data_id)
    assert data_out.shape == (10, 20, 30)

    # From existing
    discr_dom = _discrete_domain(3, 'nearest')
    data_in = discr_dom.element(np.ones(10 * 20 * 30, dtype='float32'))
    data_id = odl.tomo.astra_data(VOL_GEOM_3D, 'volume', data=data_in)
    data_out = astra.data3d.get_shared(data_id)
    assert data_out.shape == (10, 20, 30)


PROJ_GEOM_2D = {
    'type': 'parallel',
    'DetectorCount': 15, 'DetectorWidth': 1.5,
    'ProjectionAngles': np.linspace(0, 2, 5)}

PROJ_GEOM_3D = {
    'type': 'parallel3d',
    'DetectorColCount': 15, 'DetectorRowCount': 25,
    'DetectorSpacingX': 1.5, 'DetectorSpacingY': 2.5,
    'ProjectionAngles': np.linspace(0, 2, 5)}


def test_parallel_2d_projector():
    """Create ASTRA 2D projectors."""

    # We can just test if it runs
    odl.tomo.astra_projector('nearest', VOL_GEOM_2D, PROJ_GEOM_2D,
                             ndim=2, impl='cpu')
    odl.tomo.astra_projector('linear', VOL_GEOM_2D, PROJ_GEOM_2D,
                             ndim=2, impl='cpu')


def test_parallel_3d_projector():
    """Create ASTRA 3D projectors."""

    odl.tomo.astra_projector('nearest', VOL_GEOM_3D, PROJ_GEOM_3D, ndim=3,
                             impl='cuda')

    # Run as a real test once ASTRA supports this construction
    with pytest.raises(ValueError):
        odl.tomo.astra_projector('nearest', VOL_GEOM_3D, PROJ_GEOM_3D,
                                 ndim=3, impl='cpu')

    with pytest.raises(ValueError):
        odl.tomo.astra_projector('linear', VOL_GEOM_3D, PROJ_GEOM_3D,
                                 ndim=3, impl='cpu')


def test_astra_algorithm():
    """Create ASTRA algorithm object."""

    direction = 'forward'
    ndim = 2
    impl = 'cpu'
    vol_id = odl.tomo.astra_data(VOL_GEOM_2D, 'volume', ndim=ndim)
    rec_id = odl.tomo.astra_data(VOL_GEOM_2D, 'volume', ndim=ndim)
    sino_id = odl.tomo.astra_data(PROJ_GEOM_2D, 'projection', ndim=ndim)
    proj_id = odl.tomo.astra_projector('nearest', VOL_GEOM_2D, PROJ_GEOM_2D,
                                       ndim=ndim, impl=impl)

    # Checks
    with pytest.raises(ValueError):
        odl.tomo.astra_algorithm('none', ndim, vol_id, sino_id, proj_id, impl)
    with pytest.raises(ValueError):
        odl.tomo.astra_algorithm(direction, 0, vol_id, sino_id, proj_id, impl)
    with pytest.raises(ValueError):
        odl.tomo.astra_algorithm('none', ndim, vol_id, sino_id, proj_id,
                                 'none')
    with pytest.raises(ValueError):
        odl.tomo.astra_algorithm('backward', ndim, vol_id, sino_id,
                                 proj_id=None, impl='cpu')
    alg_id = odl.tomo.astra_algorithm(direction, ndim, vol_id, sino_id,
                                      proj_id, impl)
    astra.algorithm.delete(alg_id)

    # 2D CPU
    ndim = 2
    impl = 'cpu'
    for direction in {'forward', 'backward'}:
        alg_id = odl.tomo.astra_algorithm(direction, ndim, vol_id, sino_id,
                                          proj_id, impl)
        astra.algorithm.delete(alg_id)

    # 2D CUDA
    proj_id = odl.tomo.astra_projector('nearest', VOL_GEOM_2D, PROJ_GEOM_2D,
                                       ndim=ndim, impl='cuda')

    # 2D CUDA FP
    alg_id = odl.tomo.astra_algorithm('forward', ndim, vol_id, sino_id,
                                      proj_id=proj_id, impl='cuda')
    astra.algorithm.delete(alg_id)

    # 2D CUDA BP
    alg_id = odl.tomo.astra_algorithm('backward', ndim, rec_id, sino_id,
                                      proj_id=proj_id, impl='cuda')
    astra.algorithm.delete(alg_id)

    ndim = 3
    vol_id = odl.tomo.astra_data(VOL_GEOM_3D, 'volume', ndim=ndim)
    sino_id = odl.tomo.astra_data(PROJ_GEOM_3D, 'projection', ndim=ndim)
    proj_id = odl.tomo.astra_projector('nearest', VOL_GEOM_3D, PROJ_GEOM_3D,
                                       ndim=ndim, impl='cuda')

    with pytest.raises(NotImplementedError):
        odl.tomo.astra_algorithm(direction, ndim, vol_id, sino_id,
                                 proj_id=proj_id, impl='cpu')

    for direction in {'forward', 'backward'}:
        odl.tomo.astra_algorithm(direction, ndim, vol_id, sino_id,
                                 proj_id=proj_id, impl='cuda')


def test_geom_to_vec():
    """Create ASTRA projection geometry vectors using ODL geometries."""

    apart = odl.uniform_partition(0, 2 * np.pi, 5)
    dpart = odl.uniform_partition(-40, 40, 10)

    # Fanbeam flat
    src_rad = 10
    det_rad = 5
    geom_ff = odl.tomo.FanFlatGeometry(apart, dpart, src_rad, det_rad)
    vec = odl.tomo.astra_conebeam_2d_geom_to_vec(geom_ff)

    assert vec.shape == (apart.size, 6)

    # Circular cone flat
    dpart = odl.uniform_partition([-40, -3], [40, 3], (10, 5))
    geom_ccf = odl.tomo.ConeFlatGeometry(apart, dpart, src_rad, det_rad)
    vec = odl.tomo.astra_conebeam_3d_geom_to_vec(geom_ccf)
    assert vec.shape == (apart.size, 12)

    # Helical cone flat
    pitch = 1
    geom_hcf = odl.tomo.ConeFlatGeometry(apart, dpart, src_rad, det_rad,
                                         pitch=pitch)
    vec = odl.tomo.astra_conebeam_3d_geom_to_vec(geom_hcf)
    assert vec.shape == (apart.size, 12)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
