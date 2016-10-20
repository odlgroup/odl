# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Utility functions for Fourier transforms on regularly sampled data."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range

import numpy as np

from odl.discr import (RegularGrid, DiscreteLp, uniform_partition_fromgrid,
                       uniform_discr_frompartition)
from odl.set import RealNumbers
from odl.util import (
    fast_1d_tensor_mult,
    is_real_dtype, is_scalar_dtype, is_real_floating_dtype,
    is_complex_floating_dtype, complex_dtype, dtype_repr,
    conj_exponent,
    normalized_scalar_param_list, normalized_axes_tuple)


__all__ = ('reciprocal_grid', 'realspace_grid',
           'reciprocal_space',
           'dft_preprocess_data', 'dft_postprocess_data')


def reciprocal_grid(grid, shift=True, axes=None, halfcomplex=False):
    """Return the reciprocal of the given regular grid.

    This function calculates the reciprocal (Fourier/frequency space)
    grid for a given regular grid defined by the nodes::

        x[k] = x[0] + k * s,

    where ``k = (k[0], ..., k[d-1])`` is a ``d``-dimensional index in
    the range ``0 <= k < N`` (component-wise). The multi-index
    ``N`` is the shape of the input grid.
    This grid's reciprocal is then given by the nodes::

        xi[j] = xi[0] + j * sigma,

    with the reciprocal grid stride ``sigma = 2*pi / (s * N)``.
    The minimum frequency ``xi[0]`` can in principle be chosen
    freely, but usually it is chosen in a such a way that the reciprocal
    grid is centered around zero. For this, there are two possibilities:

    1. Make the grid point-symmetric around 0.

    2. Make the grid "almost" point-symmetric around zero by shifting
       it to the left by half a reciprocal stride.

    In the first case, the minimum frequency (per axis) is given as::

        xi_1[0] = -pi/s + pi/(s*n) = -pi/s + sigma/2.

    For the second case, it is::

        xi_1[0] = -pi / s.

    Note that the zero frequency is contained in case 1 for an odd
    number of points, while for an even size, the second option
    guarantees that 0 is contained.

    If a real-to-complex (half-complex) transform is to be computed,
    the reciprocal grid has the shape ``M[i] = floor(N[i]/2) + 1``
    in the last transform axis ``i``.

    Parameters
    ----------
    grid : `RegularGrid`
        Original sampling grid
    shift : bool or sequence of bools, optional
        If ``True``, the grid is shifted by half a stride in the negative
        direction. With a sequence, this option is applied separately on
        each axis.
    axes : int or sequence of ints, optional
        Dimensions in which to calculate the reciprocal. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence. ``None`` means all axes in ``grid``.
    halfcomplex : bool, optional
        If ``True``, return the half of the grid with last coordinate
        less than zero. This is related to the fact that for real-valued
        functions, the other half is the mirrored complex conjugate of
        the given half and therefore needs not be stored.

    Returns
    -------
    reciprocal_grid : `RegularGrid`
        The reciprocal grid.
    """
    if axes is None:
        axes = list(range(grid.ndim))
    else:
        try:
            axes = [int(axes)]
        except TypeError:
            axes = list(axes)

    # List indicating shift or not per "active" axis, same length as axes
    shift_list = normalized_scalar_param_list(shift, length=len(axes),
                                              param_conv=bool)

    # Full-length vectors
    stride = grid.stride
    shape = np.array(grid.shape)
    rmin = grid.min_pt.copy()
    rmax = grid.max_pt.copy()
    rshape = list(shape)

    # Shifted axes (full length to avoid ugly double indexing)
    shifted = np.zeros(grid.ndim, dtype=bool)
    shifted[axes] = shift_list
    rmin[shifted] = -np.pi / stride[shifted]
    # Length min->max increases by double the shift, so we
    # have to compensate by a full stride
    rmax[shifted] = (-rmin[shifted] -
                     2 * np.pi / (stride[shifted] * shape[shifted]))

    # Non-shifted axes
    not_shifted = np.zeros(grid.ndim, dtype=bool)
    not_shifted[axes] = np.logical_not(shift_list)
    rmin[not_shifted] = ((-1.0 + 1.0 / shape[not_shifted]) *
                         np.pi / stride[not_shifted])
    rmax[not_shifted] = -rmin[not_shifted]

    # Change last axis shape and max if halfcomplex
    if halfcomplex:
        rshape[axes[-1]] = shape[axes[-1]] // 2 + 1

        # - Odd and shifted: - stride / 2
        # - Even and not shifted: + stride / 2
        # - Otherwise: 0
        last_odd = shape[axes[-1]] % 2 == 1
        last_shifted = shift_list[-1]
        half_rstride = np.pi / (shape[axes[-1]] * stride[axes[-1]])

        if last_odd and last_shifted:
            rmax[axes[-1]] = -half_rstride
        elif not last_odd and not last_shifted:
            rmax[axes[-1]] = half_rstride
        else:
            rmax[axes[-1]] = 0

    return RegularGrid(rmin, rmax, rshape)


def realspace_grid(recip_grid, x0, axes=None, halfcomplex=False,
                   halfcx_parity='even'):
    """Return the real space grid from the given reciprocal grid.

    Given a reciprocal grid::

        xi[j] = xi[0] + j * sigma,

    with a multi-index ``j = (j[0], ..., j[d-1])`` in the range
    ``0 <= j < M``, this function calculates the original grid::

        x[k] = x[0] + k * s

    by using a provided ``x[0]`` and calculating the stride ``s``.

    If the reciprocal grid is interpreted as coming from a usual
    complex-to-complex FFT, it is ``N == M``, and the stride is::

        s = 2*pi / (sigma * N)

    For a reciprocal grid from a real-to-complex (half-complex) FFT,
    it is ``M[i] = floor(N[i]/2) + 1`` in the last transform axis ``i``.
    To resolve the ambiguity regarding the parity of ``N[i]``, the
    it must be specified if the output shape should be even or odd,
    resulting in::

        odd : N[i] = 2 * M[i] - 1
        even: N[i] = 2 * M[i] - 2

    The output stride is calculated with this ``N`` as above in this
    case.

    Parameters
    ----------
    recip_grid : `RegularGrid`
        Sampling grid in reciprocal space.
    x0 : `array-like`
        Desired minimum point of the real space grid.
    axes : int or sequence of ints, optional
        Dimensions in which to calculate the real space grid. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence. ``None`` means "all axes".
    halfcomplex : bool, optional
        If ``True``, interpret the given grid as the reciprocal as used
        in a half-complex FFT (see above). Otherwise, the grid is
        regarded as being used in a complex-to-complex transform.
    halfcx_parity : {'even', 'odd'}
        Use this parity for the shape of the returned grid in the
        last axis of ``axes`` in the case ``halfcomplex=True``

    Returns
    -------
    irecip : `RegularGrid`
        The inverse reciprocal grid
    """
    if axes is None:
        axes = list(range(recip_grid.ndim))
    else:
        try:
            axes = [int(axes)]
        except TypeError:
            axes = list(axes)

    rstride = recip_grid.stride
    rshape = recip_grid.shape

    # Calculate shape of the output grid by adjusting in axes[-1]
    irshape = list(rshape)
    if halfcomplex:
        if str(halfcx_parity).lower() == 'even':
            irshape[axes[-1]] = 2 * rshape[axes[-1]] - 2
        elif str(halfcx_parity).lower() == 'odd':
            irshape[axes[-1]] = 2 * rshape[axes[-1]] - 1
        else:
            raise ValueError("`halfcomplex` parity '{}' not understood"
                             "".format(halfcx_parity))

    irmin = np.asarray(x0)
    irshape = np.asarray(irshape)
    irstride = np.copy(rstride)
    irstride[axes] = 2 * np.pi / (irshape[axes] * rstride[axes])
    irmax = irmin + (irshape - 1) * irstride

    return RegularGrid(irmin, irmax, irshape)


def dft_preprocess_data(arr, shift=True, axes=None, sign='-', out=None):
    """Pre-process the real-space data before DFT.

    This function multiplies the given data with the separable
    function::

        p(x) = exp(+- 1j * dot(x - x[0], xi[0]))

    where ``x[0]`` and ``xi[0]`` are the minimum coodinates of
    the real-space and reciprocal grids, respectively. The sign of
    the exponent depends on the choice of ``sign``. In discretized
    form, this function becomes an array::

        p[k] = exp(+- 1j * k * s * xi[0])

    If the reciprocal grid is not shifted, i.e. symmetric around 0,
    it is ``xi[0] =  pi/s * (-1 + 1/N)``, hence::

        p[k] = exp(-+ 1j * pi * k * (1 - 1/N))

    For a shifted grid, we have :math:``xi[0] =  -pi/s``, thus the
    array is given by::

        p[k] = (-1)**k

    Parameters
    ----------
    arr : `array-like`
        Array to be pre-processed. If its data type is a real
        non-floating type, it is converted to 'float64'.
    shift : bool or or sequence of bools, optional
        If ``True``, the grid is shifted by half a stride in the negative
        direction. With a sequence, this option is applied separately on
        each axis.
    axes : int or sequence of ints, optional
        Dimensions in which to calculate the reciprocal. The sequence
        must have the same length as ``shift`` if the latter is given
        as a sequence.
        Default: all axes.
    sign : {'-', '+'}, optional
        Sign of the complex exponent.
    out : `numpy.ndarray`, optional
        Array in which the result is stored. If ``out is arr``,
        an in-place modification is performed. For real data type,
        this is only possible for ``shift=True`` since the factors are
        complex otherwise.

    Returns
    -------
    out : `numpy.ndarray`
        Result of the pre-processing. If ``out`` was given, the returned
        object is a reference to it.

    Notes
    -----
    If ``out`` is not specified, the data type of the returned array
    is the same as that of ``arr`` except when ``arr`` has real data
    type and ``shift`` is not ``True``. In this case, the return type
    is the complex counterpart of ``arr.dtype``.
    """
    arr = np.asarray(arr)
    if not is_scalar_dtype(arr.dtype):
        raise ValueError('array has non-scalar data type {}'
                         ''.format(dtype_repr(arr.dtype)))
    elif is_real_dtype(arr.dtype) and not is_real_floating_dtype(arr.dtype):
        arr = arr.astype('float64')

    if axes is None:
        axes = list(range(arr.ndim))
    else:
        try:
            axes = [int(axes)]
        except TypeError:
            axes = list(axes)

    shape = arr.shape
    shift_list = normalized_scalar_param_list(shift, length=len(axes),
                                              param_conv=bool)

    # Make a copy of arr with correct data type if necessary, or copy values.
    if out is None:
        if is_real_dtype(arr.dtype) and not all(shift_list):
            out = np.array(arr, dtype=complex_dtype(arr.dtype), copy=True)
        else:
            out = arr.copy()
    else:
        out[:] = arr

    if is_real_dtype(out.dtype) and not shift:
        raise ValueError('cannot pre-process real input in-place without '
                         'shift')

    if sign == '-':
        imag = -1j
    elif sign == '+':
        imag = 1j
    else:
        raise ValueError("`sign` '{}' not understood".format(sign))

    def _onedim_arr(length, shift):
        if shift:
            # (-1)^indices
            factor = np.ones(length, dtype=out.dtype)
            factor[1::2] = -1
        else:
            factor = np.arange(length, dtype=out.dtype)
            factor *= -imag * np.pi * (1 - 1.0 / length)
            np.exp(factor, out=factor)
        return factor.astype(out.dtype, copy=False)

    onedim_arrs = []
    for axis, shift in zip(axes, shift_list):
        length = shape[axis]
        onedim_arrs.append(_onedim_arr(length, shift))

    fast_1d_tensor_mult(out, onedim_arrs, axes=axes, out=out)
    return out


def _interp_kernel_ft(norm_freqs, interp):
    """Scaled FT of a one-dimensional interpolation kernel.

    For normalized frequencies ``-1/2 <= xi <= 1/2``, this
    function returns::

        sinc(pi * xi)**k / sqrt(2 * pi)

    where ``k=1`` for 'nearest' and ``k=2`` for 'linear' interpolation.

    Parameters
    ----------
    norm_freqs : `numpy.ndarray`
        Normalized frequencies between -1/2 and 1/2
    interp : {'nearest', 'linear'}
        Type of interpolation kernel

    Returns
    -------
    ker_ft : `numpy.ndarray`
        Values of the kernel FT at the given frequencies
    """
    # Numpy's sinc(x) is equal to the 'math' sinc(pi * x)
    ker_ft = np.sinc(norm_freqs)
    interp_ = str(interp).lower()
    if interp_ == 'nearest':
        pass
    elif interp_ == 'linear':
        ker_ft **= 2
    else:
        raise ValueError("`interp` '{}' not understood".format(interp))

    ker_ft /= np.sqrt(2 * np.pi)
    return ker_ft


def dft_postprocess_data(arr, real_grid, recip_grid, shift, axes,
                         interp, sign='-', op='multiply', out=None):
    """Post-process the Fourier-space data after DFT.

    This function multiplies the given data with the separable
    function::

        q(xi) = exp(+- 1j * dot(x[0], xi)) * s * phi_hat(xi_bar)

    where ``x[0]`` and ``s`` are the minimum point and the stride of
    the real-space grid, respectively, and ``phi_hat(xi_bar)`` is the FT
    of the interpolation kernel. The sign of the exponent depends on the
    choice of ``sign``. Note that for ``op='divide'`` the
    multiplication with ``s * phi_hat(xi_bar)`` is replaced by a
    division with the same array.

    In discretized form on the reciprocal grid, the exponential part
    of this function becomes an array::

        q[k] = exp(+- 1j * dot(x[0], xi[k]))

    and the arguments ``xi_bar`` to the interpolation kernel
    are the normalized frequencies::

        for 'shift=True'  : xi_bar[k] = -pi + pi * (2*k) / N
        for 'shift=False' : xi_bar[k] = -pi + pi * (2*k+1) / N

    See [Pre+2007]_, Section 13.9 "Computing Fourier Integrals Using
    the FFT" for a similar approach.

    Parameters
    ----------
    arr : `array-like`
        Array to be pre-processed. An array with real data type is
        converted to its complex counterpart.
    real_grid : `RegularGrid`
        Real space grid in the transform
    recip_grid : `RegularGrid`
        Reciprocal grid in the transform
    shift : bool or sequence of bools
        If ``True``, the grid is shifted by half a stride in the negative
        direction in the corresponding axes. The sequence must have the
        same length as ``axes``.
    axes : int or sequence of ints
        Dimensions along which to take the transform. The sequence must
        have the same length as ``shifts``.
    interp : string or sequence of strings
        Interpolation scheme used in the real-space.
    sign : {'-', '+'}, optional
        Sign of the complex exponent.
    op : {'multiply', 'divide'}
        Operation to perform with the stride times the interpolation
        kernel FT
    out : `numpy.ndarray`, optional
        Array in which the result is stored. If ``out is arr``, an
        in-place modification is performed.

    Returns
    -------
    out : `numpy.ndarray`
        Result of the post-processing. If ``out`` was given, the returned
        object is a reference to it.
    """
    arr = np.asarray(arr)
    if is_real_floating_dtype(arr.dtype):
        arr = arr.astype(complex_dtype(arr.dtype))
    elif not is_complex_floating_dtype(arr.dtype):
        raise ValueError('array data type {} is not a complex floating point '
                         'data type'.format(dtype_repr(arr.dtype)))

    if out is None:
        out = arr.copy()
    elif out is not arr:
        out[:] = arr

    if axes is None:
        axes = list(range(arr.ndim))
    else:
        try:
            axes = [int(axes)]
        except TypeError:
            axes = list(axes)

    shift_list = normalized_scalar_param_list(shift, length=len(axes),
                                              param_conv=bool)

    if sign == '-':
        imag = -1j
    elif sign == '+':
        imag = 1j
    else:
        raise ValueError("`sign` '{}' not understood".format(sign))

    op, op_in = str(op).lower(), op
    if op not in ('multiply', 'divide'):
        raise ValueError("kernel `op` '{}' not understood".format(op_in))

    # Make a list from interp if that's not the case already
    try:
        # Duck-typed string check
        interp + ''
    except TypeError:
        pass
    else:
        interp = [str(interp).lower()] * arr.ndim

    onedim_arrs = []
    for ax, shift, intp in zip(axes, shift_list, interp):
        x = real_grid.min_pt[ax]
        xi = recip_grid.coord_vectors[ax]

        # First part: exponential array
        onedim_arr = np.exp(imag * x * xi)

        # Second part: interpolation kernel
        len_dft = recip_grid.shape[ax]
        len_orig = real_grid.shape[ax]
        halfcomplex = (len_dft < len_orig)
        odd = len_orig % 2

        fmin = -0.5 if shift else -0.5 + 1.0 / (2 * len_orig)
        if halfcomplex:
            # maximum lies around 0, possibly half a cell left or right of it
            if shift and odd:
                fmax = - 1.0 / (2 * len_orig)
            elif not shift and not odd:
                fmax = 1.0 / (2 * len_orig)
            else:
                fmax = 0.0

        else:  # not halfcomplex
            # maximum lies close to 0.5, half or full cell left of it
            if shift:
                # -0.5 + (N-1)/N = 0.5 - 1/N
                fmax = 0.5 - 1.0 / len_orig
            else:
                # -0.5 + 1/(2*N) + (N-1)/N = 0.5 - 1/(2*N)
                fmax = 0.5 - 1.0 / (2 * len_orig)

        freqs = np.linspace(fmin, fmax, num=len_dft)
        stride = real_grid.stride[ax]

        if op == 'multiply':
            onedim_arr *= stride * _interp_kernel_ft(freqs, intp)
        else:
            onedim_arr /= stride * _interp_kernel_ft(freqs, intp)

        onedim_arrs.append(onedim_arr.astype(out.dtype, copy=False))

    fast_1d_tensor_mult(out, onedim_arrs, axes=axes, out=out)
    return out


def reciprocal_space(space, axes=None, halfcomplex=False, shift=True,
                     **kwargs):
    """Return the range of the Fourier transform on ``space``.

    Parameters
    ----------
    space : `DiscreteLp`
        Real space whose reciprocal is calculated. It must be
        uniformly discretized.
    axes : sequence of ints, optional
        Dimensions along which the Fourier transform is taken.
        Default: all axes
    halfcomplex : bool, optional
        If ``True``, take only the negative frequency part along the last
        axis for. For ``False``, use the full frequency space.
        This option can only be used if ``space`` is a space of
        real-valued functions.
    shift : bool or sequence of bools, optional
        If ``True``, the reciprocal grid is shifted by half a stride in
        the negative direction. With a boolean sequence, this option
        is applied separately to each axis.
        If a sequence is provided, it must have the same length as
        ``axes`` if supplied. Note that this must be set to ``True``
        in the halved axis in half-complex transforms.
        Default: ``True``
    impl : string, optional
        Implementation back-end for the created space.
        Default: ``'numpy'``
    exponent : float, optional
        Create a space with this exponent. By default, the conjugate
        exponent ``q = p / (p - 1)`` of the exponent of ``space`` is
        used, where ``q = inf`` for ``p = 1`` and vice versa.
    dtype : optional
        Complex data type of the created space. By default, the
        complex counterpart of ``space.dtype`` is used.

    Returns
    -------
    rspace : `DiscreteLp`
        Reciprocal of the input ``space``. If ``halfcomplex=True``, the
        upper end of the domain (where the half space ends) is chosen to
        coincide with the grid node.
    """
    if not isinstance(space, DiscreteLp):
        raise TypeError('`space` {!r} is not a `DiscreteLp` instance'
                        ''.format(space))
    if not space.is_uniform:
        raise ValueError('`space` is not uniformly discretized')

    if axes is None:
        axes = tuple(range(space.ndim))

    axes = normalized_axes_tuple(axes, space.ndim)

    if halfcomplex and space.field != RealNumbers():
        raise ValueError('`halfcomplex` option can only be used with real '
                         'spaces')

    exponent = kwargs.pop('exponent', None)
    if exponent is None:
        exponent = conj_exponent(space.exponent)

    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = complex_dtype(space.dtype)
    else:
        if not is_complex_floating_dtype(dtype):
            raise ValueError('{} is not a complex data type'
                             ''.format(dtype_repr(dtype)))

    impl = kwargs.pop('impl', 'numpy')

    # Calculate range
    recip_grid = reciprocal_grid(space.grid, shift=shift,
                                 halfcomplex=halfcomplex, axes=axes)

    # Make a partition with nodes on the boundary in the last transform axis
    # if `halfcomplex == True`, otherwise a standard partition.
    if halfcomplex:
        max_pt = {axes[-1]: recip_grid.max_pt[axes[-1]]}
        part = uniform_partition_fromgrid(recip_grid, max_pt=max_pt)
    else:
        part = uniform_partition_fromgrid(recip_grid)

    recip_spc = uniform_discr_frompartition(part, exponent=exponent,
                                            dtype=dtype, impl=impl)

    return recip_spc


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
