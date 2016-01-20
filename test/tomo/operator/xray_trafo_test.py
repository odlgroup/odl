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

# Discrete reconstruction space
xx = 5
nn = 8
discr_vol_space3 = odl.uniform_discr([-xx] * 3, [xx] * 3, [nn] * 3,
                                     dtype='float32')
discr_vol_space2 = odl.uniform_discr([-xx] * 2, [xx] * 2, [nn] * 2,
                                     dtype='float32')

# Angle
angle_intvl = odl.Interval(0, 2 * np.pi) - np.pi/4
agrid = odl.uniform_sampling(angle_intvl, 4)

# Detector
yy = 2 * xx
mm = 18
dparams1 = odl.Interval(-yy, yy)
dgrid1 = odl.uniform_sampling(dparams1, mm)
dparams2 = odl.Rectangle([-yy, -yy], [yy, yy])
dgrid2 = odl.uniform_sampling(dparams2, [mm] * 2)

# Distances
src_radius = 1000
det_radius = 500

@skip_if_no_astra_cuda
def test_xray_trafo_parallel2d():
    """Test discrete X-ray transform with ASTRA CUDA and parallel 3D beam."""

    # `DiscreteLp` volume space
    discr_vol_space2 = odl.uniform_discr([0] * 2, [10] * 2, [5] * 2,
                                         dtype='float32')

    angle_intvl = odl.Interval(0, 2 * np.pi)
    agrid = odl.uniform_sampling(angle_intvl, 5)

    dparams1 = odl.Interval(0, 10)
    dgrid1 = odl.uniform_sampling(dparams1, 10)

    # Geometry
    geom = odl.tomo.Parallel2dGeometry(angle_intvl, dparams1, agrid, dgrid1)

    # X-ray transform
    A = odl.tomo.DiscreteXrayTransform(discr_vol_space2, geom,
                                       backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f) / float(A.range.grid.stride[1])

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    inner_proj = Af.inner(g)
    inner_vol = f.inner(Adg)
    r = inner_vol / inner_proj

    # assert almost_equal(Af.inner(g), f.inner(Adg), 2)
    print('\nvol stride :', A.domain.grid.stride)
    print('proj stride:', A.range.grid.stride)

    print('\ninner vol :', inner_vol)
    print('inner proj:', inner_proj)
    print('ratios: {:.4f}, {:.4f}'.format(r, 1 / r))
    print('ratios-1: {:.4f}, {:.4f}'.format(abs(r - 1), abs(1 / r - 1)))


@skip_if_no_astra_cuda
def test_xray_trafo_parallel3d():
    """Parallel-beam X-ray transform with ASTRA CUDA."""

    # Geometry
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams2, agrid, dgrid2)

    # X-ray transform
    A = odl.tomo.DiscreteXrayTransform(discr_vol_space3, geom,
                                       backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Assure not to use unit cell sizes
    assert discr_vol_space3.grid.cell_volume != 1
    assert geom.grid.cell_volume != 1

    # Test adjoint
    assert almost_equal(Af.inner(g), f.inner(Adg), 2)

    print('\n')
    print(Af.inner(g)/f.inner(Adg))
    print(agrid.points()/np.pi)
    Af.show(indices=np.s_[0, :, :], show=True)


@pytest.mark.skipif("not odl.tomo.ASTRA_CUDA_AVAILABLE")
def test_xray_trafo_conebeam_circular():
    """Cone-beam trafo with circular acquisition and ASTRA CUDA backend."""

    # Geometry
    geom = odl.tomo.CircularConeFlatGeometry(angle_intvl, dparams2,
                                             src_radius, det_radius,
                                             agrid, dgrid2,
                                             axis=[0, 0, 1])

    # X-ray transform
    A = odl.tomo.DiscreteXrayTransform(discr_vol_space3, geom,
                                       backend='astra_cuda')
    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Assure not to use unit cell sizes
    assert discr_vol_space3.grid.cell_volume != 1
    assert geom.grid.cell_volume != 1

    # Test adjoint
    assert almost_equal(Af.inner(g), f.inner(Adg), 1)


# @pytest.mark.xfail  # Expected to fail since scaling of adjoint is wrong.
@skip_if_no_astra_cuda
def test_xray_trafo_conebeam_helical():
    """Cone-beam trafo with helical acquisition and ASTRA CUDA backend."""

    # Geometry
    geom = odl.tomo.HelicalConeFlatGeometry(angle_intvl, dparams2,
                                            src_radius, det_radius, pitch=2,
                                            agrid=agrid, dgrid=dgrid2,
                                            axis=[0, 0, 1])

    # X-ray transform
    A = odl.tomo.DiscreteXrayTransform(discr_vol_space3, geom,
                                       backend='astra_cuda')
    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    # Assure not to use trivial pitch or cell sizes
    assert discr_vol_space3.grid.cell_volume != 1
    assert geom.grid.cell_volume != 1
    assert geom.pitch != 0

    # Test adjoint
    assert almost_equal(Af.inner(g), f.inner(Adg), 2)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
