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

"""Bindings to the ``PyWavelets`` backend for wavelet transforms."""

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
           'pywt_wavelet', 'pywt_mode', 'pywt_coeff_shapes',
           'pywt_flat_coeff_size',
           'pywt_flat_array_from_coeffs', 'pywt_coeffs_from_flat_array',
           'pywt_single_level_decomp', 'pywt_single_level_recon',
           'pywt_multi_level_decomp', 'pywt_multi_level_recon')


PAD_MODES_ODL2PYWT = {'constant': 'zpd',
                      'periodic': 'ppd',
                      'symmetric': 'sym',
                      'order0': 'cpd',
                      'order1': 'sp1',
                      'pywt_periodic': 'per'}
PYWT_SUPPORTED_MODES = PAD_MODES_ODL2PYWT.values()


def pywt_wavelet(wavelet):
    """Convert ``wavelet`` to a ``pywt.Wavelet`` instance."""
    if isinstance(wavelet, pywt.Wavelet):
        return wavelet
    else:
        return pywt.Wavelet(wavelet)


def pywt_mode(pad_mode, pad_const=0):
    """Convert ODL-style padding mode to pywt-style mode."""
    pad_mode = str(pad_mode).lower()
    pad_const = float(pad_const)
    if pad_mode in PYWT_SUPPORTED_MODES:
        return pad_mode
    elif pad_mode == 'constant' and pad_const != 0.0:
        raise ValueError('constant padding with constant != 0 not supported '
                         'for `pywt` back-end')
    else:
        try:
            return PAD_MODES_ODL2PYWT[pad_mode]
        except KeyError:
            raise ValueError("`pad_mode` '{}' not understood".format(pad_mode))


def _check_nlevels(nlevels, data_len, wavelet):
    """Check if ``nlevels`` is valid for given data length and wavelet."""
    max_levels = pywt.dwt_max_level(data_len, wavelet.dec_len)
    if not 0 < nlevels <= max_levels:
        raise ValueError('`nlevels` must lie between 1 and {}, got {}'
                         ''.format(max_levels, nlevels))


def pywt_coeff_shapes(shape, wavelet, nlevels, mode):
    """Return a list of coefficient shapes in the specified transform.

    Parameters
    ----------
    shape : sequence
        Shape of an input to the transform.
    wavelet : string or ``pywt.Wavelet``
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a ``pywt.Wavelet``.
        Use `pywt.wavelist
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#\
built-in-wavelets-wavelist>`_ to get a list of available wavelets.
    nlevels : positive int
        Number of scaling levels to be used in the decomposition. The
        maximum number of levels can be calculated with
        `pywt.dwt_max_level
        <https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-\
wavelet-transform.html#maximum-decomposition-level-dwt-max-level>`_.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes
        <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html>`_ for available options.

    Returns
    -------
    shapes : list
        The shapes of the approximation and detail coefficients at
        different scaling levels in the following order:

            ``[shape_aN, shape_DN, ..., shape_D1]``

        Here, ``shape_aN`` is the shape of the N-th level approximation
        coefficient array, and ``shape_Di`` is the shape of all
        i-th level detail coefficients.
    """
    shape = tuple(shape)
    if any(int(s) != s for s in shape):
        raise ValueError('`shape` may only contain integers, got {}'
                         ''.format(shape))
    wavelet = pywt_wavelet(wavelet)

    nlevels, nlevels_in = int(nlevels), nlevels
    if float(nlevels_in) != nlevels:
        raise ValueError('`nlevels` must be integer, got {}'
                         ''.format(nlevels_in))
    # TODO: adapt for axes
    _check_nlevels(nlevels, min(shape), wavelet)

    mode, mode_in = str(mode).lower(), mode
    if mode not in PYWT_SUPPORTED_MODES:
        raise ValueError("mode '{}' not understood".format(mode_in))

    # Use pywt.dwt_coeff_len to determine the coefficient lengths at each
    # scale recursively. Start with the image shape and use the last created
    # shape for the next step.
    shape_list = [shape]
    for i in range(nlevels):
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

    Parameters
    ----------
    shape : sequence
        Shape of an input to the transform.
    wavelet : string or ``pywt.Wavelet``
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a ``pywt.Wavelet``.
        Use `pywt.wavelist
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#\
built-in-wavelets-wavelist>`_ to get a list of available wavelets.
    nlevels : positive int
        Number of scaling levels to be used in the decomposition. The
        maximum number of levels can be calculated with
        `pywt.dwt_max_level
        <https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-\
wavelet-transform.html#maximum-decomposition-level-dwt-max-level>`_.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes
        <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html>`_ for available options.

    Returns
    -------
    flat_size : int
        Size of a flat array containing all approximation and detail
        coefficients for the specified transform.
    """
    shapes = pywt_coeff_shapes(shape, wavelet, nlevels, mode)
    # 1 x approx coeff and (2**n - 1) * detail coeff
    ndim = len(shapes[0])
    return (np.prod(shapes[0]) +
            sum((2 ** ndim - 1) * np.prod(shape)
                for shape in shapes[1:]))


def pywt_flat_array_from_coeffs(coeffs):
    """Return a flat array from a Pywavelets coefficient sequence.

    Related to 1D, 2D and 3D multilevel discrete wavelet transforms.

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
    """
    flat_sizes = [np.size(coeffs[0])]
    for details in coeffs[1:]:
        if isinstance(details, tuple):
            flat_sizes.append(np.size(details[0]))
        else:
            # Single detail coefficient array
            flat_sizes.append(np.size(details))

    ndim = np.ndim(coeffs[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]
    flat_coeff[start:stop] = np.ravel(coeffs[0])

    for fsize, details in zip(flat_sizes[1:], coeffs[1:]):
        if isinstance(details, tuple):
            for detail in details:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = np.ravel(detail)
        else:
            # Single detail coefficient array
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = np.ravel(details)

    return flat_coeff


def pywt_coeffs_from_flat_array(arr, shapes):
    """Convert a flat array into a ``pywt`` coefficient list.

    For multilevel 1D, 2D and 3D discrete wavelet transforms.

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
    """Return a list of coefficient dictionary keys as used in ``pywt``."""
    return list(''.join(k) for k in product('ad', repeat=length))


def pywt_single_level_decomp(arr, wavelet, mode):
    """Return single level wavelet decomposition coefficients from ``arr``.

    Parameters
    ----------
    arr : `array-like`
        Input array to the wavelet decomposition.
    wavelet : string or ``pywt.Wavelet``
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a ``pywt.Wavelet``.
        Use `pywt.wavelist
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#\
built-in-wavelets-wavelist>`_ to get a list of available wavelets.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes
        <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html>`_ for available options.

    Returns
    -------
    approx : `numpy.ndarray`
        Approximation coefficients, a single array.
    details : tuple of `numpy.ndarray`'s
        Detail coefficients, ``2 ** ndim - 1`` arrays, where ``ndim``
        is the number of dimensions of ``arr``.
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


def pywt_single_level_recon(approx, details, wavelet, mode):
    """Return single level wavelet reconstruction from given coefficients.

    Parameters
    ----------
    approx : `array-like`
        Approximation coefficients.
    details : sequence of `array-like`'s
        Detail coefficients. The length of the sequence must be
        ``2 ** ndim - 1``, where ``ndim`` is the number of dimensions
        in ``approx``.
    wavelet :  string or ``pywt.Wavelet``
        Specification of the wavelet to be used in the transform.
        Use `pywt.wavelist
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#\
built-in-wavelets-wavelist>`_ to get a list of available wavelets.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes
        <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html>`_ for available options.

    Returns
    -------
    recon : `numpy.ndarray`
        The single-level wavelet reconstruction.
    """
    # Handle input
    approx = np.asarray(approx)
    if len(details) != 2 ** approx.ndim - 1:
        raise ValueError('`details` must be a sequence of length {}, got '
                         'length {}'
                         .format(2 ** approx.ndim - 1, len(details)))
    details = tuple(np.asarray(detail) for detail in details)
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
    return pywt.idwtn(coeff_dict, wavelet, mode)


def pywt_multi_level_decomp(arr, wavelet, nlevels, mode):
    """Return multi-level wavelet decomposition coefficients from ``arr``.

    Parameters
    ----------
    arr : `array-like`
        Input array to the wavelet decomposition.
    wavelet :  string or ``pywt.Wavelet``
        Specification of the wavelet to be used in the transform.
        Use `pywt.wavelist
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#\
built-in-wavelets-wavelist>`_ to get a list of available wavelets.
    nlevels : positive int
        Number of scaling levels to be used in the decomposition. The
        maximum number of levels can be calculated with
        `pywt.dwt_max_level
        <https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-\
wavelet-transform.html#maximum-decomposition-level-dwt-max-level>`_.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes
        <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html>`_ for available options.

    Returns
    -------
    coeff_list : structured list
        List of approximation and detail coefficients in the format

            ``[aN, DN, ... D1]``,

        where ``aN`` is the N-th level approximation coefficient array and
        ``Di`` the tuple of i-th level detail coefficient arrays. Each of
        the ``Di`` tuples has length ``2 ** ndim - 1``, where ``ndim`` is
        the number of dimensions of the input array.
        See `the documentation for the multilevel decomposition in
        PyWavelets
        <https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-\
wavelet-transform.html#multilevel-decomposition-using-wavedec>`_ for
        more information.
    """
    # Handle input
    arr = np.asarray(arr)
    wavelet = pywt_wavelet(wavelet)

    # TODO: adapt for axes
    nlevels, nlevels_in = int(nlevels), nlevels
    if float(nlevels_in) != nlevels:
        raise ValueError('`nlevels` must be integer, got {}'
                         ''.format(nlevels_in))
    max_levels = pywt.dwt_max_level(min(arr.shape), wavelet.dec_len)
    if not 0 < nlevels <= max_levels:
        raise ValueError('`nlevels` must lie between 1 and {}, got {}'
                         ''.format(max_levels, nlevels_in))

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


def pywt_multi_level_recon(coeff_list, recon_shape, wavelet, mode):
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
    recon_shape : sequence of ints
        Shape of the reconstructed array. This information is required since
        this shape is not uniquely determined by the length of the
        coefficients and the wavelet.
    wavelet :  string or ``pywt.Wavelet``
        Specification of the wavelet to be used in the transform.
        Use `pywt.wavelist
        <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#\
built-in-wavelets-wavelist>`_ to get a list of available wavelets.
    mode : string, optional
        PyWavelets style signal extension mode. See `signal extension modes
        <https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-\
modes.html>`_ for available options.

    Returns
    -------
    recon : `numpy.ndarray`
        Wavelet reconstruction from the given coefficients.
    """
    reco = coeff_list[0]

    for details in coeff_list[1:]:
        # Adjust shape of reco to match the detail coefficients' shapes.
        reco_slc = []
        try:
            details_shape = details[0].shape
        except AttributeError:
            # Single details array
            details_shape = details.shape

        for n_reco, n_detail in zip(reco.shape, details_shape):
            if n_reco == n_detail + 1:
                # Drop last element in this axis
                reco_slc.append(slice(-1))
            else:
                reco_slc.append(slice(None))

        reco = pywt_single_level_recon(reco[tuple(reco_slc)], details,
                                       wavelet, mode)

    reco_slc = []
    for n_reco, n_intended in zip(reco.shape, recon_shape):
        if n_reco == n_intended + 1:
            # Drop last element in this axis
            reco_slc.append(slice(-1))
        else:
            reco_slc.append(slice(None))

    return reco[tuple(reco_slc)]
