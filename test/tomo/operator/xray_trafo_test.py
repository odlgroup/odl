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

# Internal
import odl
from odl.util.testutils import almost_equal


def test_xray_trafo():
    """Test discrete X-ray transformt using ASTRA with CUDA."""

    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-10, -10, -10],
                                         [10, 10, 10],
                                         [10, 10, 10], dtype='float32')

    # Geometry
    angle_intvl = odl.Interval(0, 2 * np.pi)
    dparams = odl.Rectangle([-20, -20], [20, 20])
    agrid = odl.uniform_sampling(angle_intvl, 10, as_midp=False)
    dgrid = odl.uniform_sampling(dparams, [20, 20])
    geom = odl.tomo.Parallel3dGeometry(angle_intvl, dparams, agrid, dgrid)

    # X-ray transform
    A = odl.tomo.DiscreteXrayTransform(discr_reco_space, geom,
                                       backend='astra_cuda')

    # Domain element
    f = discr_reco_space.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)


    vol_stride = discr_reco_space.grid.stride[0]
    ang_stride = agrid.stride[0]
    left = Af.inner(g) / ang_stride
    right = f.inner(Adg) / vol_stride

    assert almost_equal(left, right, 3)


