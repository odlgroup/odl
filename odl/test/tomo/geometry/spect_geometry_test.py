# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test ODL geometry objects for SPECT."""

from __future__ import division

import numpy as np

import odl
from odl.tomo.geometry.spect import ParallelHoleCollimatorGeometry
from odl.util.testutils import all_equal


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
    geom = ParallelHoleCollimatorGeometry(apart, dpart, det_radius)
    assert isinstance(geom.detector, odl.tomo.Flat2dDetector)
    assert all_equal(geom.det_radius, det_radius)


if __name__ == '__main__':
    odl.util.test_file(__file__)
