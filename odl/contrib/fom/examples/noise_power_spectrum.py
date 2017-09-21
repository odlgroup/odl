"""Simple example of noise power spectrum (NPS)."""

import numpy as np
import odl
from odl.contrib import fom

# Discrete space: discretized functions on the rectangle
# [-20, 20]^2 with 100 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[512, 512])

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show('phantom')

# Create some data with noise
op = odl.tomo.RayTransform(space,
                           odl.tomo.parallel_beam_geometry(space))
fbp_op = odl.tomo.fbp_op(op, filter_type='Hann', frequency_scaling=0.5)
noisy_data = op(phantom) + odl.phantom.white_noise(op.range)
reconstruction = fbp_op(noisy_data)
reconstruction.show('reconstruction')

# Estimate NPS
nps = fom.noise_power_spectrum(reconstruction, phantom)
np.log(nps).show('log(NPS)')

# Estimate radial NPS
radial_nps = fom.noise_power_spectrum(reconstruction, phantom, radial=True)
radial_nps.show('radial NPS')
