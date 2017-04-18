# Copyright 2014-2016 The ODL development group
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

"""Test skimage back-end."""

from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np
import pytest

# Internal
import odl
from odl.tomo.backends.skimage_radon import (
    skimage_radon_forward, skimage_radon_back_projector)
from odl.tomo.util.testutils import skip_if_no_skimage


@skip_if_no_skimage
def test_skimage_radon_projector_parallel2d():
    """Parallel 2D forward and backward projectors with skimage."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-5, -5], [5, 5], (5, 5))
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0], max_pt=[5, 5])

    # Create parallel geometry
    angle_part = odl.uniform_partition(0, np.pi, 5)
    det_part = odl.uniform_partition(-6, 6, 6)
    geom = odl.tomo.Parallel2dGeometry(angle_part, det_part)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition)

    # Forward evaluation
    proj_data = skimage_radon_forward(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = skimage_radon_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
