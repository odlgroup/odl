# Copyright 2014-2016 The ODL development group
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

"""Bindings to the PyWavelets backend for wavelet transforms.

`PyWavelets <https://pywavelets.readthedocs.io/>`_ is a Python library
for wavelet transforms in arbitrary dimensions, featuring a large number
of built-in wavelet filters.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from itertools import product
import numpy as np

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


__all__ = ('PAD_MODES_ODL2PYWT', 'PYWT_SUPPORTED_MODES', 'PYWT_AVAILABLE',
           'pywt_wavelet', 'pywt_pad_mode', 'pywt_coeff_shapes',
           'pywt_flat_coeff_size',
           'pywt_flat_array_from_coeffs', 'pywt_coeffs_from_flat_array',
           'pywt_single_level_decomp', 'pywt_single_level_recon',
           'pywt_multi_level_decomp', 'pywt_multi_level_recon')


PAD_MODES_ODL2PYWT = {'constant': 'zero',
                      'periodic': 'periodic',
                      'symmetric': 'symmetric',
                      'order0': 'constant',
                      'order1': 'smooth',
                      'pywt_periodic': 'periodization',
                      # Upcoming version of Pywavelets adds this
                      # 'reflect': 'reflect'
                      }
PYWT_SUPPORTED_MODES = PAD_MODES_ODL2PYWT.values()


def pywt_wavelet(wavelet):
    """Convert ``wavelet`` to a `pywt.Wavelet` instance."""
    if isinstance(wavelet, pywt.Wavelet):
        return wavelet
    else:
        return pywt.Wavelet(wavelet)


def pywt_pad_mode(pad_mode, pad_const=0):
    """Convert ODL-style padding mode to pywt-style padding mode."""
    pad_mode = str(pad_mode).lower()
    if pad_mode == 'constant' and pad_const != 0.0:
        raise ValueError('constant padding with constant != 0 not supported '
                         'for `pywt` back-end')
    try:
        return PAD_MODES_ODL2PYWT[pad_mode]
    except KeyError:
        raise ValueError("`pad_mode` '{}' not understood".format(pad_mode))


def pywt_coeff_shapes(shape, wavelet, nlevels, mode):
    """Return a list of coefficient shapes in the specified transform.

    The wavelet transform specified by ``wavelet``, ``nlevels`` and
    ``mode`` produces a sequence of approximation and detail
    coefficients. This function computes the shape of those
    coefficient arrays and returns them as a list.

    Parameters
    ----------
    shape : sequence
        Shape of an input to the transform.
    wavelet : string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
        Use `pywt.wavelist` to get a list of available wavelets.
    nlevels : positive int
        Number of scaling levels to be used in the decomposition. The
        maximum number of levels can be calculated with
        `pywt.dwt_max_level`.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes`_
        for available options.

    Returns
    -------
    shapes : list
        The shapes of the approximation and detail coefficients at
        different scaling levels in the following order:

            ``[shape_aN, shape_DN, ..., shape_D1]``

        Here, ``shape_aN`` is the shape of the N-th level approximation
        coefficient array, and ``shape_Di`` is the shape of all
        i-th level detail coefficients.

    See Also
    --------
    pywt.dwt_coeff_len : function used to determine coefficient sizes at
        each scaling level

    Examples
    --------
    Determine the coefficient shapes for 2 scaling levels in 3 dimensions
    with zero-padding. Approximation coefficient shape comes first, then
    the level-2 detail coefficient and last the level-1 detail coefficient:

    >>> pywt_coeff_shapes(shape=(16, 17, 18), wavelet='db2', nlevels=2,
    ...                   mode='zero')
    [(6, 6, 6), (6, 6, 6), (9, 10, 10)]

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html
    """
    shape = tuple(shape)
    if any(int(s) != s for s in shape):
        raise ValueError('`shape` may only contain integers, got {}'
                         ''.format(shape))
    wavelet = pywt_wavelet(wavelet)

    nlevels, nlevels_in = int(nlevels), nlevels
    if nlevels_in != nlevels:
        raise ValueError('`nlevels` must be integer, got {}'
                         ''.format(nlevels_in))
    # TODO: adapt for axes
    for i, n in enumerate(shape):
        max_levels = pywt.dwt_max_level(n, wavelet.dec_len)
        if max_levels == 0:
            raise ValueError('in axis {}: data size {} too small for '
                             'transform, results in maximal `nlevels` of 0'
                             ''.format(i, n))
        if not 0 < nlevels <= max_levels:
            raise ValueError('in axis {}: `nlevels` must satisfy 0 < nlevels '
                             '<= {}, got {}'
                             ''.format(i, max_levels, nlevels))

    mode, mode_in = str(mode).lower(), mode
    if mode not in PYWT_SUPPORTED_MODES:
        raise ValueError("mode '{}' not understood".format(mode_in))

    # Use pywt.dwt_coeff_len to determine the coefficient lengths at each
    # scale recursively. Start with the image shape and use the last created
    # shape for the next step.
    shape_list = [shape]
    for _ in range(nlevels):
        shape = tuple(pywt.dwt_coeff_len(n, filter_len=wavelet.dec_len,
                                         mode=mode)
                      for n in shape_list[-1])
        shape_list.append(shape)

    # Add a duplicate of the last entry for the approximation coefficients
    shape_list.append(shape_list[-1])

    # We created the list in reversed order, reverse it. Remove also the
    # superfluous image shape at the end.
    shape_list.reverse()
    shape_list.pop()
    return shape_list


def pywt_flat_coeff_size(shape, wavelet, nlevels, mode):
    """Return the size of a flat array containing all coefficients.

    The wavelet transform specified by ``wavelet``, ``nlevels`` and
    ``mode`` produces a sequence of approximation and detail
    coefficients. This function computes the total size of those
    coefficients when stored in a flat vector.

    Parameters
    ----------
    shape : sequence
        Shape of an input to the transform.
    wavelet : string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
        Use `pywt.wavelist` to get a list of available wavelets.
    nlevels : positive int
        Number of scaling levels to be used in the decomposition. The
        maximum number of levels can be calculated with
        `pywt.dwt_max_level`.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes`_
        for available options.

    Returns
    -------
    flat_size : int
        Size of a flat array containing all approximation and detail
        coefficients for the specified transform.

    See Also
    --------
    pywt_coeff_shapes : calculate the shapes of coefficients in a
        specified wavelet transform

    Examples
    --------
    Determine the total size of a flat coefficient array for 2 scaling
    levels in 3 with zero-padding:

    >>> pywt_flat_coeff_size(shape=(16, 17, 18), wavelet='db2', nlevels=2,
    ...                      mode='zero')
    8028
    >>> 16 * 17 * 18  # original size, smaller -> redundancy in transform
    4896

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html
    """
    shapes = pywt_coeff_shapes(shape, wavelet, nlevels, mode)
    # 1 x approx coeff and (2**n - 1) x detail coeff
    ndim = len(shapes[0])
    return (np.prod(shapes[0]) +
            (2 ** ndim - 1) * sum(np.prod(shape)
                                  for shape in shapes[1:]))


def pywt_flat_array_from_coeffs(coeffs):
    """Return a flat array from a Pywavelets coefficient sequence.

    Parameters
    ----------
    coeffs : ordered sequence
        Sequence of approximation and detail coefficients in the format

            ``[aN, DN, ... D1]``,

        where ``aN`` is the N-th level approximation coefficient array and
        ``Di`` the tuple of i-th level detail coefficient arrays.

    Returns
    -------
    arr : `numpy.ndarray`
        Flat coefficient vector containing approximation and detail
        coefficients in the same order as they appear in ``coeffs``.

    See Also
    --------
    pywt_coeffs_from_flat_array : Conversion from flat array to
        coefficient list, the inverse of this function.

    Examples
    --------
    Flatten a list of 1 approximation and ``2 ** 2 - 1 = 3`` detail
    coefficients (sizes not representative):

    >>> approx = [[1, 1]]  # shape (1, 2)
    >>> details = ([[2, 2]], [[3, 3]], [[4, 4]])  # 3 * shape (1, 2)
    >>> pywt_flat_array_from_coeffs([approx, details])
    array([1, 1, 2, 2, 3, 3, 4, 4])
    """
    # We convert to numpy array since we need the sizes anyway, and np.size
    # is as expensive as np.asarray().size.
    approx = np.asarray(coeffs[0])
    details_list = []
    flat_sizes = [approx.size]

    for details in coeffs[1:]:
        if isinstance(details, tuple):
            detail_arrs = tuple(np.asarray(detail) for detail in details)
        else:
            # Single detail coefficient array
            detail_arrs = (np.asarray(details),)

        flat_sizes.append(detail_arrs[0].size)
        details_list.append(detail_arrs)

    ndim = approx.ndim
    dtype = approx.dtype
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size, dtype=dtype)

    start = 0
    stop = flat_sizes[0]
    flat_coeff[start:stop] = approx.ravel()

    for fsize, details in zip(flat_sizes[1:], details_list):
        for detail in details:
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = detail.ravel()

    return flat_coeff


def pywt_coeffs_from_flat_array(arr, shapes):
    """Convert a flat array into a ``pywt`` coefficient list.

    Parameters
    ----------
    arr : `array-like`
        A flat coefficient vector containing approximation and detail
        coefficients with order and sizes determined by ``shapes``.
    shapes : sequence
        The shapes of the approximation and detail coefficients at
        different scaling levels in the following order:

            ``[shape_aN, shape_DN, ..., shape_D1]``

        Here, ``shape_aN`` is the shape of the N-th level approximation
        coefficient array, and ``shape_Di`` is the shape of all
        i-th level detail coefficients.

    Returns
    -------
    coeff_list : structured list
        List of approximation and detail coefficients in the format

            ``[aN, DN, ... D1]``,

        where ``aN`` is the N-th level approximation coefficient array and
        ``Di`` the tuple of i-th level detail coefficient arrays. Each of
        the ``Di`` tuples has length ``2 ** ndim - 1``, where ``ndim`` is
        the number of dimensions of ``arr``.

    See Also
    --------
    pywt_flat_array_from_coeffs : Conversion from coefficient list to
        flat array, i.e. the inverse of this function.

    Examples
    --------
    Turn the flat array from the example in `pywt_flat_array_from_coeffs`
    back into a coefficient list:

    >>> arr = [1, 1, 2, 2, 3, 3, 4, 4]
    >>> # approximation and detail coefficient shapes
    >>> shapes = [(1, 2), (1, 2)]
    >>>
    >>> coeffs = pywt_coeffs_from_flat_array(arr, shapes)
    >>> approx, details = coeffs
    >>> print(approx)
    [[1 1]]
    >>> print(details)
    (array([[2, 2]]), array([[3, 3]]), array([[4, 4]]))
    """
    arr = np.asarray(arr)
    flat_sizes = [np.prod(shp) for shp in shapes]
    start = 0
    stop = flat_sizes[0]
    coeff_list = [arr[start:stop].reshape(shapes[0])]
    ndim = len(shapes[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    for fsize, shape in zip(flat_sizes[1:], shapes[1:]):
        start, stop = stop, stop + dcoeffs_per_scale * fsize

        detail_coeffs = tuple(
            c.reshape(shape)
            for c in np.split(arr[start:stop], dcoeffs_per_scale))

        coeff_list.append(detail_coeffs)

    return coeff_list


def pywt_dict_keys(length):
    """Return a list of coefficient dictionary keys as used in ``pywt``.

    The dictionary keys are composed of the characters ``'a'`` and ``'d'``
    and stand for "approximation" and "detail" coefficients. The letter
    in the i-th position of a key string signalizes which filtering
    operation the input undergoes in the i-th coordinate axis. For example,
    ``'ada'`` stands for low-pass filtering (-> approximation) in the
    first and last axes and high-pass filtering (-> detail) in the
    middle axis.

    Examples
    --------
    There are 8 variants for ``length=3``:

    >>> pywt_dict_keys(length=3)
    ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
    """
    return list(''.join(k) for k in product('ad', repeat=length))


def pywt_single_level_decomp(arr, wavelet, mode):
    """Return single level wavelet decomposition coefficients from ``arr``.

    Parameters
    ----------
    arr : `array-like`
        Input array to the wavelet decomposition.
    wavelet : string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
        Use `pywt.wavelist` to get a list of available wavelets.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes`_
        for available options.

    Returns
    -------
    approx : `numpy.ndarray`
        Approximation coefficients, a single array.
    details : tuple of `numpy.ndarray`'s
        Detail coefficients, ``2 ** ndim - 1`` arrays, where ``ndim``
        is the number of dimensions of ``arr``.

    See Also
    --------
    pywt_single_level_recon : Single-level reconstruction, i.e. the
        inverse of this function.
    pywt_multi_level_decomp : Multi-level version of the decompostion.

    Examples
    --------
    Decomposition of a small two-dimensional array using a Haar wavelet
    and zero padding:

    >>> arr = [[1, 1, 1],
    ...        [1, 0, 0],
    ...        [0, 1, 1]]
    >>> coeffs = pywt_single_level_decomp(arr, wavelet='haar', mode='zero')
    >>> approx, details = coeffs
    >>> print(approx)
    [[ 1.5  0.5]
     [ 0.5  0.5]]
    >>> for detail in details:
    ...     print(detail)
    [[ 0.5  0.5]
     [-0.5  0.5]]
    [[ 0.5  0.5]
     [ 0.5  0.5]]
    [[-0.5  0.5]
     [-0.5  0.5]]

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html
    """
    # Handle input
    arr = np.asarray(arr)
    wavelet = pywt_wavelet(wavelet)
    mode, mode_in = str(mode).lower(), mode
    if mode not in PYWT_SUPPORTED_MODES:
        raise ValueError("mode '{}' not understood".format(mode_in))

    # Compute single level DWT using pywt.dwtn and pick the approximation
    # and detail coefficients from the dictionary
    coeff_dict = pywt.dwtn(arr, wavelet, mode)
    dict_keys = pywt_dict_keys(arr.ndim)
    approx = coeff_dict[dict_keys[0]]
    details = tuple(coeff_dict[k] for k in dict_keys[1:])
    return approx, details


def pywt_single_level_recon(approx, details, wavelet, mode, recon_shape=None):
    """Return single level wavelet reconstruction from given coefficients.

    Parameters
    ----------
    approx : `array-like`
        Approximation coefficients.
    details : sequence of `array-like`'s
        Detail coefficients. The length of the sequence must be
        ``2 ** ndim - 1``, where ``ndim`` is the number of dimensions
        in ``approx``.
    wavelet :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        Use `pywt.wavelist` to get a list of available wavelets.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes`_
        for available options.
    recon_shape : sequence of ints, optional
        Shape of the array to be reconstructed. Without this parameter,
        the reconstructed array always has even shape due to upsampling
        by a factor of 2. To get reconstructions with odd shapes, this
        parameter is required.

    Returns
    -------
    recon : `numpy.ndarray`
        The single-level wavelet reconstruction.

    See Also
    --------
    pywt_single_level_decomp : Single-level decomposition, i.e. the
        inverse of this function.
    pywt_multi_level_recon : Multi-level version of the reconstruction.

    Examples
    --------
    Take the coefficients from the example in `pywt_single_level_decomp`
    and reconstruct the original array. Without ``recon_shape``, we get
    a ``(4, 4)`` array as reconstruction:

    >>> approx = [[1.5, 0.5],
    ...           [0.5, 0.5]]
    >>>
    >>> details = ([[ 0.5, 0.5],
    ...             [-0.5, 0.5]],
    ...            [[ 0.5, 0.5],
    ...             [ 0.5, 0.5]],
    ...            [[-0.5, 0.5],
    ...             [-0.5, 0.5]])
    >>> # Gives even shape by default
    >>> pywt_single_level_recon(approx, details, wavelet='haar', mode='zero')
    array([[ 1.,  1.,  1.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.]])
    >>> # Original shape can only be recovered if given explicitly
    >>> # in this case
    >>> pywt_single_level_recon(approx, details, wavelet='haar', mode='zero',
    ...                         recon_shape=(3, 3))
    array([[ 1.,  1.,  1.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  1.]])

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html
    """
    # Handle input
    approx = np.asarray(approx)
    if len(details) != 2 ** approx.ndim - 1:
        raise ValueError('`details` must be a sequence of length {}, got '
                         'length {}'
                         .format(2 ** approx.ndim - 1, len(details)))
    details = tuple(np.asarray(detail) for detail in details)

    if recon_shape is not None:
        recon_shape, recon_shape_in = tuple(recon_shape), recon_shape
        if any(int(s) != s for s in recon_shape):
            raise ValueError('`recon_shape` may only contain integers, got {}'
                             ''.format(recon_shape_in))

    wavelet = pywt_wavelet(wavelet)
    mode, mode_in = str(mode).lower(), mode
    if mode not in PYWT_SUPPORTED_MODES:
        raise ValueError("mode '{}' not understood".format(mode_in))

    coeff_dict = {}
    dict_keys = pywt_dict_keys(np.ndim(approx))
    if len(details) != len(dict_keys) - 1:
        raise ValueError('wrong number of detail coefficients: expected {}, '
                         'got {}'.format(len(dict_keys) - 1, len(details)))
    coeff_dict[dict_keys[0]] = approx
    coeff_dict.update(zip(dict_keys[1:], details))
    recon = pywt.idwtn(coeff_dict, wavelet, mode)

    if recon_shape is not None:
        recon_slc = []
        for i, (n_recon, n_intended) in enumerate(zip(recon.shape,
                                                      recon_shape)):
            if n_recon == n_intended + 1:
                # Upsampling added one entry too much in this axis, drop
                # last one
                recon_slc.append(slice(-1))
            elif n_recon == n_intended:
                recon_slc.append(slice(None))
            else:
                raise ValueError('in axis {}: expected size {} or {} in '
                                 '`recon_shape`, got {}'
                                 ''.format(i, n_recon - 1, n_recon,
                                           n_intended))

        recon = recon[tuple(recon_slc)]

    return recon


def pywt_multi_level_decomp(arr, wavelet, nlevels, mode):
    """Return multi-level wavelet decomposition coefficients from ``arr``.

    Parameters
    ----------
    arr : `array-like`
        Input array to the wavelet decomposition.
    wavelet :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        Use `pywt.wavelist` to get a list of available wavelets.
    nlevels : positive int
        Number of scaling levels to be used in the decomposition. The
        maximum number of levels can be calculated with
        `pywt.dwt_max_level`.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes`_
        for available options.

    Returns
    -------
    coeff_list : structured list
        List of approximation and detail coefficients in the format

            ``[aN, DN, ... D1]``,

        where ``aN`` is the N-th level approximation coefficient array and
        ``Di`` the tuple of i-th level detail coefficient arrays. Each of
        the ``Di`` tuples has length ``2 ** ndim - 1``, where ``ndim`` is
        the number of dimensions of the input array.
        See the documentation for the `multilevel decomposition`_ in
        PyWavelets for more information.

    See Also
    --------
    pywt_single_level_decomp : Single-level version of this function.
    pywt_multi_level_recon : Multi-level reconstruction, i.e. the inverse
        of this function.

    Examples
    --------
    Decomposition of a ``(4, 4)`` array using Haar wavelet, zero-padding
    and 2 scaling levels:

    >>> arr = [[1, 1, 0, 0],
    ...        [0, 0, 0, 1],
    ...        [1, 1, 1, 1],
    ...        [0, 1, 1, 0]]
    >>> coeffs = pywt_multi_level_decomp(arr, 'haar', 2, 'zero')
    >>> # "2" -> scaling level 2, "1" -> scaling level 1
    >>> approx2, details2, details1 = coeffs
    >>> print(approx2)
    [[ 2.25]]
    >>> for detail in details2:
    ...     print(detail)
    [[ 0.25]]
    [[-0.75]]
    [[ 0.25]]
    >>> for detail in details1:
    ...     print(detail)
    [[ 0.  -0.5]
     [-0.5  0.5]]
    [[ 1.  -0.5]
     [ 0.5  0.5]]
    [[ 0.   0.5]
     [ 0.5 -0.5]]

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html

    .. _multilevel decomposition:
       https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-\
wavelet-transform.html#multilevel-decomposition-using-wavedec
    """
    # Handle input
    arr = np.asarray(arr)
    wavelet = pywt_wavelet(wavelet)

    # TODO: adapt for axes
    nlevels, nlevels_in = int(nlevels), nlevels
    if nlevels_in != nlevels:
        raise ValueError('`nlevels` must be integer, got {}'
                         ''.format(nlevels_in))
    for i, n in enumerate(arr.shape):
        max_levels = pywt.dwt_max_level(n, wavelet.dec_len)
        if max_levels == 0:
            raise ValueError('in axis {}: data size {} too small for '
                             'transform, results in maximal `nlevels` of 0'
                             ''.format(i, n))
        if not 0 < nlevels <= max_levels:
            raise ValueError('in axis {}: `nlevels` must satisfy 0 < nlevels '
                             '<= {}, got {}'
                             ''.format(i, max_levels, nlevels))

    mode, mode_in = str(mode).lower(), mode
    if mode not in PYWT_SUPPORTED_MODES:
        raise ValueError("mode '{}' not understood".format(mode_in))

    # Fill the list with detail coefficients from coarsest to finest level,
    # by recursively applying the single-level transform to the approximation
    # coefficients, starting with the input array. Append the final
    # approximation coefficients.
    coeff_list = []
    approx, details = pywt_single_level_decomp(arr, wavelet, mode)
    coeff_list.append(details)

    for _ in range(1, nlevels):
        approx, details = pywt_single_level_decomp(approx, wavelet, mode)
        coeff_list.append(details)

    coeff_list.append(approx)
    coeff_list.reverse()

    return coeff_list


def pywt_multi_level_recon(coeff_list, wavelet, mode, recon_shape=None):
    """Return multi-level wavelet decomposition coefficients from ``arr``.

    Parameters
    ----------
    coeff_list : structured list
        List of approximation and detail coefficients in the format

            ``[aN, DN, ... D1]``,

        where ``aN`` is the N-th level approximation coefficient array and
        ``Di`` the tuple of i-th level detail coefficient arrays. Each of
        the ``Di`` tuples must have length ``2 ** ndim - 1``, where ``ndim``
        is the number of dimensions of the input array.
    wavelet :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        Use `pywt.wavelist` to get a list of available wavelets.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes`_
        for available options.
    recon_shape : sequence of ints, optional
        Shape of the array to be reconstructed. Without this parameter,
        the reconstructed array always has even shape due to upsampling
        by a factor of 2. To get reconstructions with odd shapes, this
        parameter is required.

    Returns
    -------
    recon : `numpy.ndarray`
        Wavelet reconstruction from the given coefficients.

    See Also
    --------
    pywt_single_level_recon : Single-level version of this function.
    pywt_multi_level_decomp : Multi-level decomposition, i.e. the inverse
        of this function.

    Examples
    --------
    Reconstruct the original array from the decomposition in the example
    in `pywt_multi_level_decomp`:

    >>> orig_arr = [[1, 1, 0, 0],
    ...             [0, 0, 0, 1],
    ...             [1, 1, 1, 1],
    ...             [0, 1, 1, 0]]
    >>> approx2 = [[2.25]]
    >>> details2 = ([[0.25]], [[-0.75]], [[0.25]])
    >>> details1 = ([[0, -0.5],
    ...              [-0.5, 0.5]],
    ...             [[1, -0.5],
    ...              [0.5, 0.5]],
    ...             [[0, 0.5],
    ...              [0.5, -0.5]])
    >>> coeffs = [approx2, details2, details1]
    >>> recon = pywt_multi_level_recon(coeffs, wavelet='haar', mode='zero')
    >>> np.allclose(recon, orig_arr)
    True

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html
    """
    if recon_shape is not None:
        recon_shape, recon_shape_in = tuple(recon_shape), recon_shape
        if any(int(s) != s for s in recon_shape):
            raise ValueError('`recon_shape` may only contain integers, got {}'
                             ''.format(recon_shape_in))

    recon = np.asarray(coeff_list[0])

    for cur_details, next_details in zip(coeff_list[1:-1], coeff_list[2:]):
        if isinstance(next_details, tuple):
            next_shape = np.shape(next_details[0])
        else:
            # Single details array
            next_shape = np.shape(next_details)

        recon = pywt_single_level_recon(recon, cur_details, wavelet, mode,
                                        recon_shape=next_shape)

    # Last reco step uses `recon_shape` for shape correction
    return pywt_single_level_recon(recon, coeff_list[-1], wavelet, mode,
                                   recon_shape=recon_shape)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYWT_AVAILABLE)
