"""Simple example of implemented figures-of-merit (FOMs).

Numerical test of a few implemented FOMs (mean square error, mean
absolute error, mean value difference, standard deviation difference,
range difference, blurring, false structures and structural similarity)
as a function of increasing noise level.

"""

import odl
from odl.contrib import fom
import numpy as np
import matplotlib.pyplot as plt

# Discrete space: discretized functions on the rectangle
# [-20, 20]^2 with 100 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100])

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

mse = []
mae = []
mvd = []
std_diff = []
range_diff = []
blur = []
false_struct = []
ssim = []

# Create mask for ROI to evaluate blurring and false structures. Arbitrarily
# chosen as bone in Shepp-Logan phantom.
mask = (np.asarray(phantom) == 1)

for stddev in np.linspace(0.1, 10, 100):
    phantom_noisy = phantom + odl.phantom.white_noise(reco_space,
                                                      stddev=stddev)
    mse.append(fom.mean_squared_error(phantom_noisy,
                                      phantom,
                                      normalized=True))

    mae.append(fom.mean_absolute_error(phantom_noisy,
                                       phantom,
                                       normalized=True))

    mvd.append(fom.mean_value_difference(phantom_noisy,
                                         phantom,
                                         normalized=True))

    std_diff.append(fom.standard_deviation_difference(phantom_noisy,
                                                      phantom,
                                                      normalized=True))

    range_diff.append(fom.range_difference(phantom_noisy,
                                           phantom,
                                           normalized=True))

    blur.append(fom.blurring(phantom_noisy,
                             phantom,
                             mask,
                             normalized=True,
                             smoothness_factor=30))

    false_struct.append(fom.false_structures(phantom_noisy,
                                             phantom,
                                             mask,
                                             normalized=True,
                                             smoothness_factor=30))

    ssim.append(fom.ssim(phantom_noisy,
                         phantom,
                         normalized=True))

plt.figure('Figures of merit')
plt.plot(mse, label='Mean squared error')
plt.plot(mae, label='Mean absolute error')
plt.plot(mvd, label='Mean value difference')
plt.plot(std_diff, label='Standard deviation difference')
plt.plot(range_diff, label='Range difference')
plt.plot(blur, label='Blurring')
plt.plot(false_struct, label='False structures')
plt.plot(ssim, label='Structural similarity')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fancybox=True, shadow=True, ncol=1)
plt.xlabel('Noise level')
plt.ylabel('FOM')
