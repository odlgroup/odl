"""Simple example of implemented figures-of-merit (FOMs).

Numerical test of a few implemented FOMs (mean square error, mean
absolute error, mean value difference, standard deviation difference,
range difference, blurring, false structures and structural similarity)
as a function of increasing noise level.

"""

import matplotlib.pyplot as plt
import numpy as np

import odl
from odl.contrib import fom

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
psnr = []
haarpsi = []

# Create mask for ROI to evaluate blurring and false structures. Arbitrarily
# chosen as bone in Shepp-Logan phantom.
mask = (np.asarray(phantom) == 1)

for stddev in np.linspace(0.1, 10, 100):
    phantom_noisy = phantom + odl.phantom.white_noise(reco_space,
                                                      stddev=stddev)
    mse.append(
        fom.mean_squared_error(phantom_noisy, phantom, normalized=True))

    mae.append(
        fom.mean_absolute_error(phantom_noisy, phantom, normalized=True))

    mvd.append(
        fom.mean_value_difference(phantom_noisy, phantom, normalized=True))

    std_diff.append(
        fom.standard_deviation_difference(phantom_noisy, phantom,
                                          normalized=True))

    range_diff.append(
        fom.range_difference(phantom_noisy, phantom, normalized=True))

    blur.append(
        fom.blurring(phantom_noisy, phantom, mask, normalized=True,
                     smoothness_factor=30))

    false_struct.append(
        fom.false_structures(phantom_noisy, phantom, mask, normalized=True,
                             smoothness_factor=30))

    ssim.append(
        fom.ssim(phantom_noisy, phantom))

    psnr.append(
        fom.psnr(phantom_noisy, phantom, normalized=True))

    haarpsi.append(
        fom.haarpsi(phantom_noisy, phantom))

fig, ax = plt.subplots()
ax.plot(mse, label='MSE')
ax.plot(mae, label='MAE')
ax.plot(mvd, label='MVD')
ax.plot(std_diff, label='SDD')
ax.plot(range_diff, label='RD')
ax.plot(blur, label='BLUR')
ax.plot(false_struct, label='FS')
ax.plot(ssim, label='SSIM')
ax.plot(haarpsi, label='HaarPSI')
plt.legend(loc='center right', fancybox=True, shadow=True, ncol=1)
ax.set_xlabel('Noise level')
ax.set_ylabel('FOM')
plt.title('Figures of merit')
fig.show()
