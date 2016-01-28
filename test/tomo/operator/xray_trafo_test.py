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

"""Test for X-ray transforms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.util.testutils import almost_equal
from odl.tomo import ASTRA_CUDA_AVAILABLE
from odl.tomo.util.testutils import skip_if_no_astra_cuda


# TODO: improve tests
# TODO: increase test coverage, see largescale test section

# DiscreteLp volume / reconstruction space
xx = 5
nn = 5
# xx = 5.5
# nn = 11
# xx = 5.25
# nn = 21
discr_vol_space_2d = odl.uniform_discr([-xx] * 2, [xx] * 2, [nn] * 2,
                                       dtype='float32')
discr_vol_space_3d = odl.uniform_discr([-xx] * 3, [xx] * 3, [nn] * 3,
                                       dtype='float32')

# Angle grid
angle_intvl = odl.Interval(0, 2 * np.pi) - np.pi / 4
agrid = odl.uniform_sampling(angle_intvl, 4)
astride = float(agrid.stride)
num_angle = agrid.ntotal

# Detector grid
# yy = 11
# mm = 11
yy = 10.5
mm = 2 * 21
dparams_2d = odl.Interval(-yy, yy)
dgrid_2d = odl.uniform_sampling(dparams_2d, mm)
dparams_3d = odl.Rectangle([-yy, -yy], [yy, yy])
dgrid_3d = odl.uniform_sampling(dparams_3d, [mm] * 2)

# Distances
src_radius = 1000
det_radius = 500
mag = (src_radius + det_radius) / src_radius

# Slice index to print
z_vol = np.round(discr_vol_space_3d.shape[2] / 2)
y_proj = np.floor(dgrid_3d.shape[1] / 2)

# Precision for adjoint test
precision = 4

# Print section
print('\n')
print('angle grid = {}'.format(agrid.points().transpose() / np.pi))
print('magnification = ', mag)
print('2D:\n shape = ', discr_vol_space_2d.shape)
print(' size = ', discr_vol_space_2d.grid.size())
print(' dgrid = ', dgrid_2d.points().transpose()[0])
print(' vol_grid = ', discr_vol_space_2d.points()[::nn][:, 0])
print('3D:\n shape = ', discr_vol_space_3d.shape)
print(' size = ', discr_vol_space_3d.grid.size())
print(' dgrid = ', dgrid_3d.points()[:2 * nn][:, 1])
print(' vol_grid = ', discr_vol_space_3d.points()[:, 2][:nn])
print('\n vol z index = ', z_vol, '\n proj y index = ', y_proj)


@skip_if_no_astra_cuda
def test_xray_trafo_cpu_parallel2d():
    """2D parallel-beam discrete X-ray transform with ASTRA and CPU."""

    dparams = dparams_2d
    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, agrid, dgrid)

    # X-ray transform
    projector = odl.tomo.XrayTransform(discr_vol_space, geom,
                                       backend='astra_cpu')

    # Domain element
    vol_phantom = projector.domain.one()

    # Forward projection
    proj_data = projector(vol_phantom)

    # Range element
    proj_phantom = projector.range.one()

    # Back projection
    reco_data = projector.adjoint(proj_phantom)

    inner_proj = proj_data.inner(proj_phantom)
    inner_vol = vol_phantom.inner(reco_data)
    r = inner_vol / inner_proj

    # Adjoint matching
    assert almost_equal(inner_vol, inner_proj, precision)

    print('\n\nCPU PARALLEL')
    print('vol stride', projector.domain.grid.stride)
    print('proj stride', projector.range.grid.stride)
    print('forward')
    print(proj_data.asarray()[0])
    print('backward / angle_stride / num_angle')
    print(reco_data.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cpu_fanflat():
    """2D fanbeam discrete X-ray transform with ASTRA and CUDA."""

    dparams = dparams_2d
    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_radius,
                                    det_radius, agrid, dgrid)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom,
                               backend='astra_cpu')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    r = inner_vol / inner_proj

    # Adjoint matching
    assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCPU FANFLAT')
    print('vol stride', A.domain.grid.stride)
    print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_parallel2d():
    """2D parallel-beam discrete X-ray transform with ASTRA and CUDA."""

    dparams = dparams_2d
    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams, agrid, dgrid)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom,
                               backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCUDA PARALLEL 2D')
    print('vol stride', A.domain.grid.stride)
    print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_fanflat():
    """2D fanbeam discrete X-ray transform with ASTRA and CUDA."""

    dparams = dparams_2d
    dgrid = dgrid_2d
    discr_vol_space = discr_vol_space_2d

    # Geometry
    geom = odl.tomo.FanFlatGeometry(angle_intvl, dparams, src_radius,
                                    det_radius, agrid, dgrid)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom,
                               backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCUDA FANFLAT')
    print('vol stride', A.domain.grid.stride)
    print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray() / astride / num_angle)
    print('<A f,g> =  ', inner_proj, '\n<f,Ad g> = ', inner_vol)
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_parallel3d():
    """3D parallel-beam discrete X-ray transform with ASTRA CUDA."""

    dparams = dparams_3d
    dgrid = dgrid_3d
    discr_vol_space = discr_vol_space_3d

    # Geometry
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, agrid, dgrid)

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom,
                               backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    assert almost_equal(inner_vol, inner_proj, precision)

    print('\nCUDA PARALLEL 3D')
    print('vol stride', A.domain.grid.stride)
    print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0, :, y_proj])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray()[:, :, z_vol] / astride / num_angle)
    print('<A f,g> = ', inner_proj, '\n<f,Ad g> =', inner_vol)
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@pytest.mark.skipif("not odl.tomo.ASTRA_CUDA_AVAILABLE")
def test_xray_trafo_cuda_conebeam_circular():
    """Cone-beam trafo with circular acquisition and ASTRA CUDA back-end."""

    dparams = dparams_3d
    dgrid = dgrid_3d

    # Geometry
    geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams,
                                             src_radius, det_radius,
                                             agrid, dgrid,
                                             axis=[0, 0, 1])

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space_3d, geom,
                               backend='astra_cuda')
    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    assert almost_equal(inner_vol, inner_proj, precision - 1)

    print('\nCUDA CONE CIRCULAR')
    print('vol stride', A.domain.grid.stride)
    print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0, :, y_proj])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray()[:, :, z_vol] / astride / num_angle)
    print('<A f,g>: ', Af.inner(g), '\n<f,Ad g>', f.inner(Adg))
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


@skip_if_no_astra_cuda
def test_xray_trafo_cuda_conebeam_helical():
    """Cone-beam trafo with helical acquisition and ASTRA CUDA back-end."""

    dparams = dparams_3d
    dgrid = dgrid_3d
    discr_vol_space = discr_vol_space_3d

    # Geometry
    geom = odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams,
                                            src_radius, det_radius, pitch=2,
                                            agrid=agrid, dgrid=dgrid,
                                            axis=[0, 0, 1])

    # X-ray transform
    A = odl.tomo.XrayTransform(discr_vol_space, geom,
                               backend='astra_cuda')
    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    assert geom.pitch != 0

    # Test adjoint matching
    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    assert almost_equal(inner_proj, inner_vol, precision - 2)

    print('\nCUDA CONE HELICAL')
    print('vol stride', A.domain.grid.stride)
    print('proj stride', A.range.grid.stride)
    print('forward')
    print(Af.asarray()[0, :, y_proj])
    print('backward / angle_stride / num_angle')
    print(Adg.asarray()[:, :, z_vol] / astride / num_angle)
    print('<A f,g>: ', Af.inner(g), '\n<f,Ad g>', f.inner(Adg))
    r = inner_vol / inner_proj
    print('ratio: v/p = {:f}, p/v = {:f}'.format(r, 1 / r))


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
