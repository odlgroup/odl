"""Demonstration of the components of the HaarPSI figure of merit."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from odl.contrib import fom

# --- Generate images --- #

true_image = scipy.misc.ascent().astype('float32')

# Image with false structures
corrupt1 = true_image.copy()
corrupt1[100:110, 50:350] = 0  # long narrow hole
corrupt1[200:204, 60:64] = 180  # small square corrupted
corrupt1[350:400, 200:250] = 200  # large square corrupted

# Noisy image, using a predictable outcome
np.random.seed(123)
corrupt2 = true_image + np.random.uniform(0, 40, true_image.shape)
corrupt2 = corrupt2.astype('float32')

# Show images
fig, ax = plt.subplots(ncols=3)
ax[0].imshow(true_image, cmap='gray')
ax[0].set_title('original')
ax[1].imshow(corrupt1, cmap='gray')
ax[1].set_title('corrupted')
ax[2].imshow(corrupt2, cmap='gray')
ax[2].set_title('noisy')
fig.suptitle('Input images for comparison')
fig.tight_layout()
fig.show()

# --- Similarity maps --- #

# Compute similarity maps
a = 4.2
c = 100
sim1_ax0 = fom.util.haarpsi_similarity_map(corrupt1, true_image, axis=0,
                                           a=a, c=c)
sim1_ax1 = fom.util.haarpsi_similarity_map(corrupt1, true_image, axis=1,
                                           a=a, c=c)
sim2_ax0 = fom.util.haarpsi_similarity_map(corrupt2, true_image, axis=0,
                                           a=a, c=c)
sim2_ax1 = fom.util.haarpsi_similarity_map(corrupt2, true_image, axis=1,
                                           a=a, c=c)

# Show similarity images
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(sim1_ax0)
ax[0].set_title('axis 0')
ax[1].imshow(sim1_ax1)
ax[1].set_title('axis 1')
fig.suptitle('Similarity maps for corrupted image')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(sim2_ax0)
ax[0].set_title('axis 0')
ax[1].imshow(sim2_ax1)
ax[1].set_title('axis 1')
fig.suptitle('Similarity maps for noisy image')
fig.tight_layout()
fig.show()

# --- Weight maps --- #

# Compute similarity weight maps
wmap1_ax0 = fom.util.haarpsi_weight_map(corrupt1, true_image, axis=0)
wmap1_ax1 = fom.util.haarpsi_weight_map(corrupt1, true_image, axis=1)
wmap2_ax0 = fom.util.haarpsi_weight_map(corrupt2, true_image, axis=0)
wmap2_ax1 = fom.util.haarpsi_weight_map(corrupt2, true_image, axis=1)

# Show weight maps
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(wmap1_ax0)
ax[0].set_title('axis 0')
ax[1].imshow(wmap1_ax1)
ax[1].set_title('axis 1')
fig.suptitle('Weight maps for corrupted image')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(wmap2_ax0)
ax[0].set_title('axis 0')
ax[1].imshow(wmap2_ax1)
ax[1].set_title('axis 1')
fig.suptitle('Weight maps for noisy image')
fig.tight_layout()
fig.show()

# --- HaarPSI scores --- #

# Compute HaarPSI scores
score1 = fom.haarpsi(corrupt1, true_image, a, c)
score2 = fom.haarpsi(corrupt2, true_image, a, c)

print('Similarity score (a = {}, c = {}) for corrupted image: {}'
      ''.format(a, c, score1))
print('Similarity score (a = {}, c = {}) for noisy image: {}'
      ''.format(a, c, score2))
