# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test reconstruction with STIR."""

from __future__ import division
import os.path as pth
import odl
import pytest
from odl.tomo.backends.stir_bindings import stir_projector_from_file


pytestmark = odl.util.skip_if_no_largescale


@odl.util.skip_if_no_stir
def test_from_file():
    # Set path to input files
    base = pth.join(pth.dirname(pth.abspath(__file__)), 'data', 'stir')
    volume_file = str(pth.join(base, 'initial.hv'))
    projection_file = str(pth.join(base, 'small.hs'))

    # Create a STIR projector from file data.
    proj = stir_projector_from_file(volume_file, projection_file)

    # Create SPECT phantom
    vol = odl.phantom.derenzo_sources(proj.domain)

    # Project data
    projections = proj(vol)

    # Calculate operator norm for landweber
    op_norm_est_squared = proj.adjoint(projections).norm() / vol.norm()
    omega = 0.5 / op_norm_est_squared

    # Reconstruct using ODL
    recon = proj.domain.zero()
    odl.solvers.landweber(proj, recon, projections, niter=100, omega=omega)

    # Make sure the result is somewhat close to the actual result.
    assert recon.dist(vol) < vol.norm() / 2.0


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--largescale'])
