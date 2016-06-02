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

"""Tests ODL geometry opjects for the positron emission tomography."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest

# Internal
import odl
from odl.util.testutils import all_equal
from odl.tomo.geometry.pet import CylindricalPetGeom


def test_pet_geom():

    # Scanner specifications
    det_nx_mm = 4
    det_ny_mm = 4
    det_nx_pix = 13
    det_ny_pix = 13
    num_rings = 4
    num_blocks = 48
    det_radius = 40

    # Detector parameters
    a = det_nx_mm * det_nx_pix
    b = det_ny_mm
    dpart = odl.uniform_partition([-b, -a], [b, a], [1, det_ny_pix])

    # Axial (z-axis) movement parameters.
    axial_pos_num = det_nx_pix * num_rings + (num_rings - 1)
    axialpart = odl.uniform_partition(0, axial_pos_num, axial_pos_num)

    geom = CylindricalPetGeom(num_rings, num_blocks, dpart, axialpart,
                              det_radius)

    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)
    assert all_equal(geom.det_radius, det_radius)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
