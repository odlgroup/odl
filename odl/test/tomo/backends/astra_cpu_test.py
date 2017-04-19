# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test ASTRA backend using CPU."""

from __future__ import division
import numpy as np
import pytest
import sys

import odl
from odl.tomo.backends.astra_cpu import (
    astra_cpu_forward_projector, astra_cpu_back_projector)
from odl.tomo.util.testutils import skip_if_no_astra

# TODO: clean up and improve tests


@pytest.mark.xfail(sys.platform == 'win32', run=False,
                   reason="Crashes on windows")
@skip_if_no_astra
def test_astra_cpu_projector_parallel2d():
    """ASTRA CPU forward and back projection for 2d parallel geometry."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5), dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0], max_pt=[4, 5])

    # Create parallel geometry
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition(-6, 6, 6)
    geom = odl.tomo.Parallel2dGeometry(angle_part, det_part)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cpu_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cpu_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


@skip_if_no_astra
def test_astra_cpu_projector_fanflat():
    """ASTRA CPU forward and back projection for fanflat geometry."""

    # Create reco space and a phantom
    reco_space = odl.uniform_discr([-4, -5], [4, 5], (4, 5), dtype='float32')
    phantom = odl.phantom.cuboid(reco_space, min_pt=[0, 0], max_pt=[4, 5])

    # Create fan beam geometry with flat detector
    angle_part = odl.uniform_partition(0, 2 * np.pi, 8)
    det_part = odl.uniform_partition(-6, 6, 6)
    src_rad = 100
    det_rad = 10
    geom = odl.tomo.FanFlatGeometry(angle_part, det_part, src_rad, det_rad)

    # Make projection space
    proj_space = odl.uniform_discr_frompartition(geom.partition,
                                                 dtype='float32')

    # Forward evaluation
    proj_data = astra_cpu_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = astra_cpu_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
