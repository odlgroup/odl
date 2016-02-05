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


@pytest.mark.xfail  # Expected to fail since scaling of adjoint is wrong.
@pytest.mark.skipif("not odl.tomo.ASTRA_CUDA_AVAILABLE")
def test_xray_trafo():
    """Test discrete X-ray transformt using ASTRA with CUDA."""

    # Discrete reconstruction space
    discr_reco_space = odl.uniform_discr([-10, -10, -10],
                                         [10, 10, 10],
                                         [10, 10, 10], dtype='float32')

    # Geometry
    agrid = odl.uniform_sampling(0, 2 * np.pi, 10)
    dgrid = odl.uniform_sampling([-20, -20], [20, 20], [20, 20])
    geom = odl.tomo.Parallel3dGeometry(agrid, dgrid)

    # X-ray transform
    A = odl.tomo.DiscreteXrayTransform(discr_reco_space, geom,
                                       backend='astra_cuda')

    # Domain element
    f = A.domain.one()

    # Forward projection
    Af = A(f)

    # Range element
    g = A.range.one()

    # Back projection
    Adg = A.adjoint(g)

    assert almost_equal(Af.inner(g), f.inner(Adg), 3)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
