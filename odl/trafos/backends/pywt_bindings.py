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

import numpy as np

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


__all__ = ('pywt_wbasis', 'pywt_coeff_shape_list',
           'pywt_coeff_to_array', 'array_to_pywt_coeff',
           'PAD_MODES_ODL2PYWT', 'PAD_MODES_ODL2PYWT',
           'PYWT_SUPPORTED_PAD_MODES', 'PYWT_AVAILABLE')


PAD_MODES_ODL2PYWT = {'constant': 'zpd',
                      'periodic': 'ppd',
                      'symmetric': 'sym',
                      'order0': 'cpd',
                      'order1': 'sp1',
                      'pywt_periodic': 'per'}
PAD_MODES_PYWT2ODL = {v: k for k, v in PAD_MODES_ODL2PYWT.items()}
PYWT_SUPPORTED_PAD_MODES = PAD_MODES_ODL2PYWT.keys()


def pywt_wbasis(wbasis):
    """Convert ``wbasis`` a to `pywt.Wavelet` if it is not already one."""
    if isinstance(wbasis, pywt.Wavelet):
        return wbasis
    else:
        return pywt.Wavelet(wbasis)


def pywt_max_level(*args, **kwargs):
    """Return maximum number of scaling levels."""
    if not PYWAVELETS_AVAILABLE:
        raise NotImplementedError('PyWavelets not available')
    return pywt.dwt_max_level(*args, **kwargs)


def pywt_coeff_shape_list(shape, wbasis, nscales, pad_mode):
    """Return a list of coefficient shapes in the specified transform.

    Parameters
    ----------
    shape : tuple
        Number of pixels/voxels in the image. Its length must be 1, 2 or 3.
    wbasis : string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
        For more information see the `wavelets
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_
        page of the PyWavelets documentation.

        Possible wavelet families are:

        'Haar': Haar

        'db': Daubechies

        'sym' : Symlets

        'coif': Coiflets

        'bior': Biorthogonal

        'rbio': Reverse biorthogonal

        'dmey': Discrete FIR approximation of the Meyer wavelet

    nscales : int
        Number of scaling levels to be used in the transform.
        The maximum number of usable scales can be determined
        by `pywt.dwt_max_level`.
    pad_mode : string, optional
        Signal extention mode. Possible extension modes are:

        ``'constant'``: Fill with ``pad_const``.

        ``'symmetric'``: Reflect at the boundaries, not doubling the
        outmost values.

        ``'periodic'``: Fill in values from the other side, keeping
        the order.

        ``'order0'``: Extend constantly with the outmost values
        (ensures continuity).

        ``'order1'``: Extend with constant slope (ensures continuity of
        the first derivative). This requires at least 2 values along
        each axis where padding is applied.

        ``'pywt_periodic'``:  like ``'periodic'`` padding but gives
        the smallest possible number of decomposition coefficients.
        Only available with ``impl='pywt'``, See `pywt.MODES.modes`.

    Returns
    -------
    shape_list : list
        List containing the shapes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

        ``shape_list[0]`` = shape of approximation coefficients at
        the coarsest level,

        ``shape_list[1]`` = shape of the detailed coefficients at
        the coarsest level,

        ``shape_list[N]`` = shape of the detailed coefficients at
        the finest level,

        ``shape_list[N+1]`` = shape of the original image,

        ``N`` =  number of scaling levels
    """
    if len(shape) not in (1, 2, 3):
        raise ValueError('shape must have length 1, 2 or 3, got {}'
                         ''.format(len(shape)))

    wbasis = pywt_wbasis(wbasis)
    pywt_mode = PAD_MODES_ODL2PYWT[pad_mode]

    max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
    if nscales > max_level:
        raise ValueError('too many scaling levels, got {}, maximum useful '
                         'level is {}'
                         ''.format(nscales, max_level))

    # dwt_coeff_len calculates the number of coefficients at the next
    # scaling level given the input shape, the length of the filter and
    # the applied mode.
    # We use this in the following way (per dimension):
    # - length[0] = original data length
    # - length[n+1] = dwt_coeff_len(length[n], ...)
    # - until n = nscales
    shape_list = [shape]
    for scale in range(nscales):
        shp = tuple(pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len,
                                       mode=pywt_mode)
                    for n in shape_list[scale])
        shape_list.append(shp)

    # Add a duplicate of the last entry for the approximation coefficients
    shape_list.append(shape_list[-1])

    # We created the list in reversed order compared to what pywt expects
    shape_list.reverse()
    return shape_list


def pywt_coeff_to_array(coeff, shape_list):
    """Convert a Pywavelets coefficient list into a flat array.

    Related to 1D, 2D and 3D multilevel discrete wavelet transforms.

    Parameters
    ----------
    coeff : ordered `sequence`
        Sequence of coefficients organized in the following way:

        In 1D:

        ``[aN, (dN), ..., (d1)]``

        In 2D:

        ``[aaN, (adN, daN, ddN), ..., (ad1, da1, dd1)]``

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations are "a" for "approximation" and "d" for "detail".
        The letter position encodes the coefficient type in the respective
        axis. ``N`` stands for the number of scaling levels.

    shape_list : `sequence`
        Sequence containing the shapes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

        ``shape_list[0]`` = shape of approximation coefficients at
        the coarsest level,

        ``shape_list[1]`` = shape of the detailed coefficients at
        the coarsest level,

        ``shape_list[N]`` = shape of the detailed coefficients at
        the finest level,

        ``shape_list[N+1]`` = shape of original image,

        ``N`` =  number of scaling levels

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array.
        The length of the array depends on the shape of the input image to
        be transformed and on the chosen wavelet basis.
      """
    flat_sizes = [np.prod(shp) for shp in shape_list[:-1]]
    ndim = len(shape_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]
    flat_coeff[start:stop] = coeff[0].ravel()

    for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
        if dcoeffs_per_scale == 1:
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = detail_coeffs.ravel()
        else:
            for dcoeff in detail_coeffs:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def array_to_pywt_coeff(coeff, shape_list):
    """Convert a flat array into a ``pywt`` coefficient list.

    For multilevel 1D, 2D and 3D discrete wavelet transforms.

    Parameters
    ----------
    coeff : `array-like`
        A flat coefficient vector containing approximation and detail
        coefficients in the following order (3D):

            ``[aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN,
            aad1, ada1, add1, daa1, dad1, dda1, ddd1]``

    shape_list : `sequence`
        Sequence containing the shapes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

        ``shape_list[0]`` = shape of approximation coefficients at
        the coarsest level,

        ``shape_list[1]`` = shape of the detail coefficients at
        the coarsest level,

        ``shape_list[N]`` = shape of the detail coefficients at
        the finest level,

        ``size_list[N+1]`` = shape of the original image,

        ``N`` =  number of scaling levels

    Returns
    -------
    coeff_list : ordered list
        Sequence of coefficients organized in the following way:

        In 1D:

        ``[aN, (dN), ..., (d1)]``

        In 2D:

        ``[aaN, (adN, daN, ddN), ..., (ad1, da1, dd1)]``

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations are "a" for "approximation" and "d" for "detail".
        The letter position encodes the coefficient type in the respective
        axis. ``N`` stands for the number of scaling levels.
    """
    coeff_arr = np.asarray(coeff)
    flat_sizes = [np.prod(shp) for shp in shape_list[:-1]]
    start = 0
    stop = flat_sizes[0]
    coeff_list = [coeff_arr[start:stop].reshape(shape_list[0])]
    ndim = len(shape_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    for fsize, shape in zip(flat_sizes[1:], shape_list[1:]):
        start, stop = stop, stop + dcoeffs_per_scale * fsize
        if dcoeffs_per_scale == 1:
            detail_coeffs = coeff_arr[start:stop]
        else:
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(coeff_arr[start:stop],
                                           dcoeffs_per_scale))
        coeff_list.append(detail_coeffs)

    return coeff_list


def pywt_wavelet_decomp(arr, wbasis, pad_mode, nscales, shape_list):
    """Return the discrete wavelet transform of ``arr``.

    Parameters
    ----------
    arr : `array-like`
        Input array to the wavelet decomposition.
    wbasis :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
    pad_mode: string
        ODL style boundary condition.
    nscales : int
        Number of scaling levels to be used in the transform.
        The maximum number of usable scales can be determined
        by `pywt.dwt_max_level`.
    shape_list : `sequence`
        Sequence containing the shapes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

    Returns
    -------
    out : `numpy.ndarray`
        Flattened and concatenated coefficient array.
        The length of the array depends on the shape of the input image to
        be transformed and on the chosen wavelet basis.
    """
    arr = np.asarray(arr)
    wbasis = pywt_wbasis(wbasis)
    pywt_mode = PAD_MODES_ODL2PYWT[pad_mode]
    if arr.ndim == 1:
        coeff_list = pywt.wavedec(arr, wbasis, pywt_mode, nscales)
    elif arr.ndim == 2:
        coeff_list = pywt.wavedec2(arr, wbasis, pywt_mode, nscales)
    elif arr.ndim == 3:
        coeff_list = _wavedec3(arr, wbasis, pywt_mode, nscales)
    else:
        raise NotImplementedError('no transform available for {} dimensions'
                                  ''.format(arr.ndim))
    return pywt_coeff_to_array(coeff_list, shape_list)


def pywt_wavelet_recon(coeff, wbasis, pad_mode, shape_list):
    """Return the discrete wavelet reconstruction from ``coeff``.

    Parameters
    ----------
    coeff : `array-like`
        Flat coefficient array used as input to the wavelet
        reconstruction.
    wbasis :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
    pad_mode: string
        ODL style boundary condition.
    shape_list : `sequence`
        Sequence containing the shapes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

    Returns
    -------
    out : `numpy.ndarray`
        Flattened and concatenated coefficient array.
        The length of the array depends on the shape of the input image to
        be transformed and on the chosen wavelet basis.
    """
    coeff = np.asarray(coeff)
    wbasis = pywt_wbasis(wbasis)
    coeff_list = array_to_pywt_coeff(coeff, shape_list)
    pywt_mode = PAD_MODES_ODL2PYWT[pad_mode]
    ndim = len(shape_list[0])
    nscales = len(shape_list) - 1

    if ndim == 1:
        return pywt.waverec(coeff_list, wbasis, pywt_mode)
    elif ndim == 2:
        return pywt.waverec2(coeff_list, wbasis, pywt_mode)
    elif ndim == 3:
        return _waverec3(coeff_list, wbasis, pywt_mode, nscales)
    else:
        raise NotImplementedError('no transform available for {} dimensions'
                                  ''.format(ndim))


def _wavedec3(arr, wbasis, pywt_mode, nscales):
    """Discrete 3D multiresolution wavelet decomposition.

    Helper function for the 3D wavelet decomposition that constructs
    the same type of ordered coefficient list as the 1D and 2D variants
    of `pywt.wavedecn`. This is necessary since the 3D variant returns
    a dictionary instead of an ordered list.

    Parameters
    ----------
    arr : `array-like`
        Input array to the wavelet decomposition.
    wbasis :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
    pywt_mode: string
        `pywt` style boundary condition.
    nscales : int
        Number of scaling levels to be used in the transform.
        The maximum number of usable scales can be determined
        by `pywt.dwt_max_level`.

    Returns
    -------
    coeff_list : ordered list
        List of coefficients organized in the following way:

            ```[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...,
            (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]```

        The abbreviations are "a" for "approximation" and "d" for "detail".
        The letter position encodes the coefficient type in the respective
        axis. ``N`` stands for the number of scaling levels.
    """
    wbasis = pywt_wbasis(wbasis)
    coeff_list = []

    wcoeffs = pywt.dwtn(arr, wbasis, pywt_mode)
    aaa = wcoeffs['aaa']
    aad = wcoeffs['aad']
    ada = wcoeffs['ada']
    add = wcoeffs['add']
    daa = wcoeffs['daa']
    dad = wcoeffs['dad']
    dda = wcoeffs['dda']
    ddd = wcoeffs['ddd']

    details = (aad, ada, add, daa, dad, dda, ddd)
    coeff_list.append(details)

    for _ in range(1, nscales):
        wcoeffs = pywt.dwtn(aaa, wbasis, pywt_mode)
        aaa = wcoeffs['aaa']
        aad = wcoeffs['aad']
        ada = wcoeffs['ada']
        add = wcoeffs['add']
        daa = wcoeffs['daa']
        dad = wcoeffs['dad']
        dda = wcoeffs['dda']
        ddd = wcoeffs['ddd']
        details = (aad, ada, add, daa, dad, dda, ddd)
        coeff_list.append(details)

    coeff_list.append(aaa)
    coeff_list.reverse()

    return coeff_list


def _waverec3(coeff_list, wbasis, pywt_mode, nscales):
    """Discrete 3D multiresolution wavelet reconstruction.

    Helper function for the 3D wavelet reconstruction that uses
    the same type of ordered coefficient list as the 1D and 2D variants
    of `pywt.waverecn`. This is necessary since the 3D variant takes
    a dictionary instead of an ordered list.

    Parameters
    ----------
    coeff_list : ordered `sequence`
        Sequence of coefficients organized in the following way:

            ```[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...,
            (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]```

        The abbreviations are "a" for "approximation" and "d" for "detail".
        The letter position encodes the coefficient type in the respective
        axis. ``N`` stands for the number of scaling levels.

    wbasis :  string or `pywt.Wavelet`
        Specification of the wavelet to be used in the transform.
        If a string is given, it is converted to a `pywt.Wavelet`.
    pywt_mode: string
        `pywt` style boundary condition.
    nscales : int
        Number of scaling levels to be used in the transform.
        The maximum number of usable scales can be determined
        by `pywt.dwt_max_level`.

    Returns
    -------
    rec : `numpy.ndarray`
        Wavelet reconstruction from the coefficients.
    """
    wbasis = pywt_wbasis(wbasis)

    aaa = coeff_list[0]
    (aad, ada, add, daa, dad, dda, ddd) = coeff_list[1]
    coeff_dict = {'aaa': aaa, 'aad': aad, 'ada': ada, 'add': add,
                  'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}
    for tpl in coeff_list[2:]:
        aaa = pywt.idwtn(coeff_dict, wbasis, pywt_mode)
        (aad, ada, add, daa, dad, dda, ddd) = tpl
        coeff_dict = {'aaa': aaa, 'aad': aad, 'ada': ada, 'add': add,
                      'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}

    return pywt.idwtn(coeff_dict, wbasis, pywt_mode)
