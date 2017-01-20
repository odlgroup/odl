"""Simple example on the usage of the Wavelet Transform."""

import odl

# Discretized space: discretized functions on the rectangle [-1, 1] x [-1, 1]
# with 512 samples per dimension.
space = odl.uniform_discr([-1, -1], [1, 1], (256, 256))

# Make the Wavelet transform operator on this space. The range is calculated
# automatically. The default backend is PyWavelets (pywt).
wavelet_op = odl.trafos.WaveletTransform(space, wavelet='Haar', nlevels=2)

# Create a phantom and its wavelet transfrom and display them.
phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show(title='Shepp-Logan phantom')

# Note that the wavelet transform is a vector in rn.
phantom_wt = wavelet_op(phantom)
phantom_wt.show(title='wavelet transform')

# It may however (for some choices of wbasis) be interpreted as a vector in the
# domain of the transformation
phantom_wt_2d = space.element(phantom_wt)
phantom_wt_2d.show('wavelet transform in 2d')

# Calculate the inverse transform.
phantom_wt_inv = wavelet_op.inverse(phantom_wt)
phantom_wt_inv.show(title='wavelet transform inverted')
