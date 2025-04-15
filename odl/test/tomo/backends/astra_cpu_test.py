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
from odl.tomo.util.testfixtures import PARALLEL_2D_PROJECTORS_CPU, projector

# TODO: clean up and improve tests

projectors = []
projectors.extend(
    (pytest.param(proj_cfg, marks=skip_if_no_astra)
     for proj_cfg in PARALLEL_2D_PROJECTORS_CPU)
)

projector_ids = [
    " geometry='{}' - dimension='{}' - ray_trafo_impl='{}' - reco_space_impl='{}' - angles='{}' - device='{}'".format(*p.values[0].split())
    for p in projectors
]
projector = pytest.fixture(fixture_function=projector, params=projectors, ids=projector_ids)

@pytest.mark.xfail(sys.platform == 'win32', run=False,
                   reason="Crashes on windows")
@skip_if_no_astra
def test_astra_cpu_projector_parallel2d(projector):
    """ASTRA CPU forward and back projection for 2d parallel geometry."""
    phantom = odl.phantom.cuboid(projector.domain)

    # Forward evaluation
    proj_data = projector(phantom)
    assert proj_data.shape == projector.range.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = projector.adjoint(proj_data)
    assert backproj.shape == projector.domain.shape
    assert backproj.norm() > 0

if __name__ == '__main__':
    odl.util.test_file(__file__)
