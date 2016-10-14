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

"""Test ODL geometry objects for SPECT."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
from odl.util.testutils import all_equal
from odl.tomo.geometry.spect import ParallelHoleCollimatorGeometry


def test_spect():
    det_nx_pix = 64
    det_ny_pix = 64
    det_nx_mm = 4
    det_radius = 200
    n_proj = 180
    det_param = det_nx_mm * det_nx_pix
    dpart = odl.uniform_partition([-det_param, -det_param],
                                  [det_param, det_param],
                                  [det_nx_pix, det_ny_pix])

    apart = odl.uniform_partition(0, 2 * np.pi, n_proj)
    geom = ParallelHoleCollimatorGeometry(
        apart, dpart, det_rad=det_radius)
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)
    assert all_equal(geom.det_radius, det_radius)

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-vs'])
