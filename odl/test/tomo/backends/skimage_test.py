# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test skimage back-end."""

from __future__ import division
import numpy as np

import odl
from odl.tomo.backends.skimage_radon import (
    skimage_radon_forward_projector, skimage_radon_back_projector)
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
    proj_data = skimage_radon_forward_projector(phantom, geom, proj_space)
    assert proj_data.shape == proj_space.shape
    assert proj_data.norm() > 0

    # Backward evaluation
    backproj = skimage_radon_back_projector(proj_data, geom, reco_space)
    assert backproj.shape == reco_space.shape
    assert backproj.norm() > 0


if __name__ == '__main__':
    odl.util.test_file(__file__)
