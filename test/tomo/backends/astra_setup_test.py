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

"""Test astra setup functions."""

from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
from odl.set.domain import Interval, IntervalProd
from odl.discr.grid import TensorGrid, uniform_sampling
from odl.discr.lp_discr import uniform_discr
from odl.util.testutils import all_equal, is_subdict
from odl.tomo.geometry.parallel import (Parallel2dGeometry,
                                        Parallel3dGeometry)
from odl.tomo.geometry.fanbeam import FanFlatGeometry
from odl.tomo.geometry.conebeam import (CircularConeFlatGeometry,
                                        HelicalConeFlatGeometry)
from odl.tomo.util.testutils import skip_if_no_astra
from odl.tomo.backends.astra_setup import ASTRA_AVAILABLE
if ASTRA_AVAILABLE:
    import astra
if ASTRA_AVAILABLE:
    from odl.tomo.backends.astra_setup import (astra_projection_geometry,
                                               astra_volume_geometry,
                                               astra_data, astra_projector,
                                               astra_algorithm, astra_cleanup,
                                               astra_geom_to_vec)


# TODO: add test to check scaling of ASTRA projectors

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
    minpt = np.arange(-1, -ndim - 1, -1)  # -1, -2 [, -3]
    maxpt = np.arange(1, ndim + 1, 1)  # 1, 2 [, 3]
    nsamples = np.arange(1, ndim + 1, 1) * 10  # 10, 20 [, 30]

    return uniform_discr(minpt, maxpt, nsamples=nsamples, interp=interp,
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
    minpt = [-1] * ndim
    maxpt = [1] * ndim
    nsamples = np.arange(1, ndim + 1, 1) * 10  # 10, 20 [, 30]

    return uniform_discr(minpt, maxpt, nsamples=nsamples, interp=interp,
                         dtype='float32')


@skip_if_no_astra
def test_vol_geom_2d():
    """Create ASTRA 2D volume geometry."""

    discr_dom = _discrete_domain(2, 'nearest')
    vol_geom = astra_volume_geometry(discr_dom)

    x_pts = 10  # x_pts = Rows
    y_pts = 20  # y_pts = Columns
    assert discr_dom.grid.shape == (x_pts, y_pts)

    correct_dict = {
        'GridColCount': y_pts,
        'GridRowCount': x_pts,
        'option': {
            'WindowMinX': -2.0,  # y_min
            'WindowMaxX': 2.0,  # y_max
            'WindowMinY': -1.0,  # x_min
            'WindowMaxY': 1.0}}  # x_amx

    assert vol_geom == correct_dict

    # non-isotropic case should fail due to lacking ASTRA support
    discr_dom = _discrete_domain_anisotropic(2, 'nearest')
    with pytest.raises(NotImplementedError):
        astra_volume_geometry(discr_dom)


@skip_if_no_astra
def test_vol_geom_3d():
    """Create ASTRA 2D volume geometry."""

    discr_dom = _discrete_domain(3, 'nearest')
    vol_geom = astra_volume_geometry(discr_dom)

    x_pts = 10  # x_pts =
    y_pts = 20  # y_pts =
    z_pts = 30  # z_pts =
    assert discr_dom.grid.shape == (x_pts, y_pts, z_pts)

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

    assert vol_geom == correct_dict

    # non-isotropic case should fail due to lacking ASTRA support
    discr_dom = _discrete_domain_anisotropic(3, 'nearest')
    with pytest.raises(NotImplementedError):
        astra_volume_geometry(discr_dom)


@skip_if_no_astra
def test_proj_geom_parallel_2d():
    """Create ASTRA 2D projection geometry."""

    angles = Interval(0, 2)
    angle_grid = uniform_sampling(angles, 5)
    det_params = Interval(-1, 1)
    det_grid = uniform_sampling(det_params, 10)
    geom = Parallel2dGeometry(angles, det_params, angle_grid, det_grid)

    proj_geom = astra_projection_geometry(geom)

    correct_subdict = {
        'type': 'parallel',
        'DetectorCount': 10, 'DetectorWidth': 0.2}

    assert is_subdict(correct_subdict, proj_geom)
    assert 'ProjectionAngles' in proj_geom
    a0 = 0.5 * 2. / 5
    assert all_equal(proj_geom['ProjectionAngles'], np.linspace(a0, 2 - a0, 5))


@skip_if_no_astra
def test_astra_projection_geometry():
    """Create ASTRA projection geometry from `odl.tomo.geometry` objects."""

    with pytest.raises(TypeError):
        astra_projection_geometry(None)

    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 5)
    dparams = Interval(-40, 40)
    det_grid = uniform_sampling(dparams, 10)

    # no detector sampling grid, no motion sampling grid
    geom = Parallel2dGeometry(angle_intvl, dparams)
    with pytest.raises(ValueError):
        astra_projection_geometry(geom)

    # motion sampling grid, but no detector sampling grid
    geom = Parallel2dGeometry(angle_intvl, dparams, angle_grid)
    with pytest.raises(ValueError):
        astra_projection_geometry(geom)

    # detector sampling grid, but no motion sampling grid
    geom = Parallel2dGeometry(angle_intvl, dparams, dgrid=det_grid)
    with pytest.raises(ValueError):
        astra_projection_geometry(geom)

    # motion sampling grid, detector sampling grid but not RegularGrid
    geom = Parallel2dGeometry(angle_intvl=angle_intvl, dparams=dparams,
                              agrid=angle_grid, dgrid=TensorGrid([0]))
    with pytest.raises(TypeError):
        astra_projection_geometry(geom)

    # detector sampling grid, motion sampling grid
    geom = Parallel2dGeometry(angle_intvl, dparams, angle_grid, det_grid)
    astra_projection_geometry(geom)

    # PARALLEL 2D GEOMETRY
    geom = Parallel2dGeometry(angle_intvl, dparams, angle_grid, det_grid)
    ageom = astra_projection_geometry(geom)
    assert ageom['type'] == 'parallel'

    # FANFLAT
    src_rad = 10
    det_rad = 5
    geom_ff = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              agrid=angle_grid, dgrid=det_grid)
    ageom = astra_projection_geometry(geom_ff)
    assert ageom['type'] == 'fanflat'

    dparams = IntervalProd([-40, -3], [40, 3])
    det_grid = uniform_sampling(dparams, (10, 5))

    # PARALLEL 3D GEOMETRY
    geom_p3d = Parallel3dGeometry(angle_intvl, dparams, angle_grid, det_grid)
    astra_projection_geometry(geom_p3d)
    ageom = astra_projection_geometry(geom_p3d)
    assert ageom['type'] == 'parallel3d'

    # CIRCULAR CONEFLAT
    geom_ccf = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                        agrid=angle_grid, dgrid=det_grid)
    ageom = astra_projection_geometry(geom_ccf)
    assert ageom['type'] == 'cone'

    # HELICAL CONEFLAT
    spiral_pitch_factor = 1
    geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                       spiral_pitch_factor, agrid=angle_grid,
                                       dgrid=det_grid)
    ageom = astra_projection_geometry(geom_hcf)
    assert ageom['type'] == 'cone_vec'


vol_geom_2d = {
    'GridColCount': 20, 'GridRowCount': 10,
    'option': {'WindowMinX': -2.0, 'WindowMaxX': 2.0,
               'WindowMinY': -1.0, 'WindowMaxY': 1.0}}


@skip_if_no_astra
def test_volume_data_2d():
    """Create ASTRA data structure in 2D."""

    # From scratch
    data_id = astra_data(vol_geom_2d, 'volume', ndim=2)
    data_out = astra.data2d.get_shared(data_id)
    assert data_out.shape == (10, 20)

    # From existing
    discr_dom = _discrete_domain(2, 'nearest')
    data_in = discr_dom.element(np.ones(10 * 20, dtype='float32'))
    data_id = astra_data(vol_geom_2d, 'volume', data=data_in)
    data_out = astra.data2d.get_shared(data_id)
    assert data_out.shape == (10, 20)


vol_geom_3d = {
    'GridColCount': 30, 'GridRowCount': 20, 'GridSliceCount': 10,
    'option': {}}


@skip_if_no_astra
def test_volume_data_3d():
    """Create ASTRA data structure in 2D."""

    # From scratch
    data_id = astra_data(vol_geom_3d, 'volume', ndim=3)
    data_out = astra.data3d.get_shared(data_id)
    assert data_out.shape == (10, 20, 30)

    # From existing
    discr_dom = _discrete_domain(3, 'nearest')
    data_in = discr_dom.element(np.ones(10 * 20 * 30, dtype='float32'))
    data_id = astra_data(vol_geom_3d, 'volume', data=data_in)
    data_out = astra.data3d.get_shared(data_id)
    assert data_out.shape == (10, 20, 30)


proj_geom_2d = {
    'type': 'parallel',
    'DetectorCount': 15, 'DetectorWidth': 1.5,
    'ProjectionAngles': np.linspace(0, 2, 5)}

proj_geom_3d = {
    'type': 'parallel3d',
    'DetectorColCount': 15, 'DetectorRowCount': 25,
    'DetectorSpacingX': 1.5, 'DetectorSpacingY': 2.5,
    'ProjectionAngles': np.linspace(0, 2, 5)}


@skip_if_no_astra
def test_parallel_2d_projector():
    """Create ASTRA 2D projectors."""

    # We can just test if it runs
    astra_projector('nearest', vol_geom_2d, proj_geom_2d,
                    ndim=2, impl='cpu')
    astra_projector('linear', vol_geom_2d, proj_geom_2d,
                    ndim=2, impl='cpu')


@skip_if_no_astra
def test_parallel_3d_projector():
    """Create ASTRA 2D projectors."""

    astra_projector('nearest', vol_geom_3d, proj_geom_3d, ndim=3, impl='cuda')

    # Run as a real test once ASTRA supports this construction
    with pytest.raises(ValueError):
        astra_projector('nearest', vol_geom_3d, proj_geom_3d,
                        ndim=3, impl='cpu')

    with pytest.raises(ValueError):
        astra_projector('linear', vol_geom_3d, proj_geom_3d,
                        ndim=3, impl='cpu')


@skip_if_no_astra
def test_astra_algorithm():
    """Create ASTRA algorithm object."""

    direction = 'forward'
    ndim = 2
    impl = 'cpu'
    vol_id = astra_data(vol_geom_2d, 'volume', ndim=ndim)
    rec_id = astra_data(vol_geom_2d, 'volume', ndim=ndim)
    sino_id = astra_data(proj_geom_2d, 'projection', ndim=ndim)
    proj_id = astra_projector('nearest', vol_geom_2d, proj_geom_2d,
                              ndim=ndim, impl=impl)

    # Checks
    with pytest.raises(ValueError):
        astra_algorithm('none', ndim, vol_id, sino_id, proj_id, impl)
    with pytest.raises(ValueError):
        astra_algorithm(direction, 0, vol_id, sino_id, proj_id, impl)
    with pytest.raises(ValueError):
        astra_algorithm('none', ndim, vol_id, sino_id, proj_id, 'none')
    with pytest.raises(ValueError):
        astra_algorithm('backward', ndim, vol_id, sino_id, proj_id=None,
                        impl='cpu')
    alg_id = astra_algorithm(direction, ndim, vol_id, sino_id, proj_id, impl)
    astra.algorithm.delete(alg_id)

    ndim = 2
    # 2D CPU
    impl = 'cpu'
    for direction in {'forward', 'backward'}:
        alg_id = astra_algorithm(direction, ndim, vol_id, sino_id, proj_id,
                                 impl)
        astra.algorithm.delete(alg_id)

    # 2D CUDA
    proj_id = astra_projector('nearest', vol_geom_2d, proj_geom_2d, ndim=ndim,
                              impl='cuda')

    # 2D CUDA FP
    alg_id = astra_algorithm('forward', ndim, vol_id, sino_id,
                             proj_id=proj_id, impl='cuda')
    astra.algorithm.delete(alg_id)

    # 2D CUDA BP
    alg_id = astra_algorithm('backward', ndim, rec_id, sino_id,
                             proj_id=proj_id, impl='cuda')
    astra.algorithm.delete(alg_id)

    ndim = 3
    vol_id = astra_data(vol_geom_3d, 'volume', ndim=ndim)
    sino_id = astra_data(proj_geom_3d, 'projection', ndim=ndim)
    proj_id = astra_projector('nearest', vol_geom_3d, proj_geom_3d, ndim=ndim,
                              impl='cuda')

    with pytest.raises(NotImplementedError):
        astra_algorithm(direction, ndim, vol_id, sino_id, proj_id=proj_id,
                        impl='cpu')

    for direction in {'forward', 'backward'}:
        astra_algorithm(direction, ndim, vol_id, sino_id, proj_id=proj_id,
                        impl='cuda')


@skip_if_no_astra
def test_geom_to_vec():
    """Create ASTRA projection geometries vectors using ODL geometries."""

    angle_intvl = Interval(0, 2 * np.pi)
    angle_grid = uniform_sampling(angle_intvl, 5)
    dparams = Interval(-40, 40)
    det_grid = uniform_sampling(dparams, 10)

    geom = Parallel2dGeometry(angle_intvl, dparams, angle_grid, det_grid)

    with pytest.raises(ValueError):
        astra_geom_to_vec(geom)

    # FAN FLAT
    src_rad = 10
    det_rad = 5
    geom_ff = FanFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                              agrid=angle_grid, dgrid=det_grid)
    vec = astra_geom_to_vec(geom_ff)

    assert vec.shape == (angle_grid.ntotal, 6)

    # CIRCULAR CONE FLAT
    dparams = IntervalProd([-40, -3], [40, 3])
    det_grid = uniform_sampling(dparams, (10, 5))
    geom_ccf = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                        agrid=angle_grid, dgrid=det_grid)
    vec = astra_geom_to_vec(geom_ccf)
    assert vec.shape == (angle_grid.ntotal, 12)

    # HELICAL CONE FLAT
    spiral_pitch_factor = 1
    geom_hcf = HelicalConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad,
                                       spiral_pitch_factor, agrid=angle_grid,
                                       dgrid=det_grid)
    vec = astra_geom_to_vec(geom_hcf)
    assert vec.shape == (angle_grid.ntotal, 12)


@skip_if_no_astra
def test_astra_cleanup():
    """Clean up ASTRA memory."""
    astra_cleanup()


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
