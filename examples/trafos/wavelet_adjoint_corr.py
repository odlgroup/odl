# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:02:24 2016

@author: hkohr
"""

import numpy as np
import pywt
from scipy.signal import convolve

import odl


# Problem size
n = 11
# data = np.ones(n)
data = np.random.randn(n)

# Filter
wavelet = pywt.Wavelet('db4')
filter = wavelet.dec_lo

# Padded length, i.e. size needed for application of the correction
padded_len = data.shape[0] + len(filter) - 1

# Full padded length, i.e. len(filter) added to both sides so mode='valid'
# yields padded_len
full_len = data.shape[0] + 2 * (len(filter) - 1)

# Zero-padded array for the correction step. Ensure that the extra node
# ends up on the right. (HACK)
data_zp = odl.util.resize_array(data, (padded_len,), frac_left=0.499)

# Symmetrically padded array of full size, so all nodes of interest are valid
data_symm = odl.util.resize_array(data, (full_len,), frac_left=0.499,
                                  pad_mode='symmetric')

# Convolution using the fully padded array, this is the ground truth
conv_symm = convolve(data_symm, filter, mode='valid')

# Zero-padded convolution as basis for correction
conv_zp = convolve(data_zp, filter, mode='same')


def symm_matrices(filter):
    """Return the correction matrices for symmetric padding."""
    filter = np.asarray(filter)
    k = filter.size
    nrows, ncols = k - 1, k - 1
    top = np.zeros((nrows, ncols))
    bot = top.copy()

    for row in range(nrows):
        top[row, :ncols - row] = filter[row + 1:]
        bot[row, ::-1][:row + 1] = filter[:row + 1][::-1]

    return top, bot


top, bot = symm_matrices(filter)

# Apply the correction, starting off at the zero-padded convolution
conv_symm_corr = conv_zp.copy()
conv_symm_corr[:top.shape[0]] += top.dot(data[:top.shape[1]])
conv_symm_corr[-bot.shape[0]:] += bot.dot(data[-bot.shape[1]:])

print(conv_symm)
print(conv_symm_corr)
