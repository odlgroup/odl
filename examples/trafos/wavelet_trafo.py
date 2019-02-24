"""Simple example on the usage of the Wavelet Transform."""

import odl

# Discretized space: discretized functions on the rectangle [-1, 1] x [-1, 1]
# with 512 samples per dimension.
space = odl.uniform_discr([-1, -1], [1, 1], (256, 256))

# Make the Wavelet transform operator on this space. The range is calculated
# automatically. The default backend is PyWavelets (pywt).
wavelet_op = odl.trafos.WaveletTransform(space, wavelet='Haar', nlevels=2)

# Create a phantom and its wavelet transfrom and display them
phantom = odl.phantom.shepp_logan(space, modified=True)
space.show(phantom, title='Shepp-Logan Phantom')

# Note that the wavelet transform is a vector in R^n
phantom_wt = wavelet_op(phantom)
wavelet_op.range.show(phantom_wt, title='Wavelet Transform')

# It may however (for some choices of wbasis) be interpreted as an element
# of the transformation domain
phantom_wt_2d = phantom_wt.reshape(space.shape)
space.show(phantom_wt_2d, 'Wavelet Transform in 2D')

# Calculate the inverse transform
phantom_wt_inv = wavelet_op.inverse(phantom_wt)
space.show(phantom_wt_inv, title='Wavelet Transform Inverted', force_show=True)
