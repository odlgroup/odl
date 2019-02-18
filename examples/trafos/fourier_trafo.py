"""Simple example on the usage of the Fourier Transform."""

import odl


# Discretized space: discretized functions on the rectangle [-1, 1] x [-1, 1]
# with 512 samples per dimension and complex data type (for full FT).
space = odl.uniform_discr([-1, -1], [1, 1], (512, 512), dtype='complex')

# Make the Fourier transform operator on this space. The range is calculated
# automatically. The default backend is numpy.fft.
ft_op = odl.trafos.FourierTransform(space)

# Create a phantom and its Fourier transfrom and display them.
phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show(title='Shepp-Logan Phantom')
phantom_ft = ft_op(phantom)
phantom_ft.show(title='Full Fourier Transform')

# Calculate the inverse transform.
phantom_ft_inv = ft_op.inverse(phantom_ft)
phantom_ft_inv.show(title='Full Fourier Transform Inverted')

# Calculate the FT only along the first axis.
ft_op_axis0 = odl.trafos.FourierTransform(space, axes=0)
phantom_ft_axis0 = ft_op_axis0(phantom)
phantom_ft_axis0.show(title='Fourier transform Along Axis 0')

# If a real space is used, the Fourier transform can be calculated in the
# "half-complex" mode. This means that along the last axis of the transform,
# only the negative half of the spectrum is stored since the other half is
# its complex conjugate. This is faster and more memory efficient.
real_space = space.real_space
ft_op_halfc = odl.trafos.FourierTransform(real_space, halfcomplex=True)
phantom_real = odl.phantom.shepp_logan(real_space, modified=True)
phantom_real.show(title='Shepp-Logan Phantom, Real Version')
phantom_real_ft = ft_op_halfc(phantom_real)
phantom_real_ft.show(title='Half-complex Fourier Transform')

# If the space is real, the inverse also gives a real result.
phantom_real_ft_inv = ft_op_halfc.inverse(phantom_real_ft)
phantom_real_ft_inv.show(title='Half-complex Fourier Transform Inverted',
                         force_show=True)

# The FT operator itself has no option of (zero-)padding, but it can be
# composed with a `ResizingOperator` which does exactly that. Note that the
# FT needs to be redefined on the enlarged space.
padding_op = odl.ResizingOperator(space, ran_shp=(768, 768))
ft_op = odl.trafos.FourierTransform(padding_op.range)
padded_ft_op = ft_op * padding_op
phantom_ft_padded = padded_ft_op(phantom)
phantom_ft_padded.show('Padded FT of the Phantom', force_show=True)
