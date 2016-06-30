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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str, super

import numpy as np
from itertools import product
try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

from odl.discr import DiscreteLp
from odl.operator import Operator


__all__ = ('WaveletTransform', 'WaveletTransformInverse',
           'PYWAVELETS_AVAILABLE')


_SUPPORTED_IMPL = ('pywt',)


def coeff_size_list(shape, wbasis, mode, nscales=None, axes=None):
    """Construct a size list from given wavelet coefficients.

    Related to 1D, 2D and 3D multidimensional wavelet transforms that utilize
    `PyWavelets
    <http://www.pybytes.com/pywavelets/>`_.

    Parameters
    ----------
    shape : `tuple`
        Number of pixels/voxels in the image. Its length must be 1, 2 or 3.

    wbasis : ``pywt.Wavelet``
        Selected wavelet basis. For more information see the
        `PyWavelets documentation on wavelet bases
        <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_.

    mode : `str`
        Signal extention mode. Possible extension modes are

        'zero': zero-padding -- signal is extended by adding zero samples

        'constant': constant padding -- border values are replicated

        'symmetric': symmetric padding -- signal extension by mirroring samples

        'periodic': periodic padding -- signal is trated as a periodic one

        'smooth': smooth padding -- signal is extended according to the
        first derivatives calculated on the edges (straight line)

        'periodization': periodization -- like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    nscales : `int`, optional
        Number of scales in the multidimensional wavelet
        transform.  This parameter is checked against the maximum number of
        scales returned by ``pywt.dwt_max_level``. For more information
        see the `PyWavelets documentation on the maximum level of scales
        <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_.
        If `nscales=None` `axes` has to be given

    axes : sequence of `int`, optional
         Dimensions in which to calculate the wavelet transform.
         If `axes=None` `nscales` has to be given

    Returns
    -------
    size_list : list
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels:

        ``size_list[0]`` = size of approximation coefficients at
        the coarsest level

        ``size_list[1]`` = size of the detail coefficients at the
        coarsest level

        ...

        ``size_list[N]`` = size of the detail coefficients at the
        finest level

        ``size_list[N+1]`` = size of the original image

        ``N`` = number of scaling levels = nscales
    """
    if len(shape) not in (1, 2, 3):
        raise ValueError('Shape must have length 1, 2 or 3, got {}.'
                         ''.format(len(shape)))

    if nscales is None and axes is None:
        raise ValueError('Either nscales or axes has to be defined')

    if axes is None:
        max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
        if nscales > max_level:
            raise ValueError('Too many scaling levels, got {}, maximum useful'
                             ' level is {}'
                             ''.format(nscales, max_level))

        # dwt_coeff_len calculates the number of coefficients at the next
        # scaling level given the input size, the length of the filter and
        # the applied mode.
        # We use this in the following way (per dimension):
        # - length[0] = original data length
        # - length[n+1] = dwt_coeff_len(length[n], ...)
        # - until n = nscales
        size_list = [shape]
        for scale in range(nscales):
            shp = tuple(pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len,
                                           mode=mode)
                        for n in size_list[scale])
            size_list.append(shp)

        # Add a duplicate of the last entry for the approximation coefficients
        size_list.append(size_list[-1])
        # We created the list in reversed order compared to what pywt expects
        size_list.reverse()

    if nscales is None:
        size_list = [shape]
        ndim = len(shape)
        axes_counts = [axes.count(i) for i in range(ndim)]
        reduced_shape = []
        for ax_len, ax_count in zip(shape, axes_counts):
            n = ax_len
            for _ in range(ax_count):
                n = pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len,
                                       mode=mode)
            reduced_shape.append(n)

        size_list.append(tuple(reduced_shape))
        size_list.append(size_list[-1])
        size_list.reverse()

    return size_list


def pywt_dict_to_array(coeffs, size_list, axes):
    """Convert a PyWavelet coefficient dictionary into a flat array.

    Related to 2D and 3D discrete wavelet transforms with `axes` option.
    Computing 1D wavelet transform multiple times along the axis
    corresponds to computing 1D multilevel wavelet transform.

    Parameters
    ----------
    coeff : ordered `dict`
        Coefficients are organized in the dictionary with the following
        appreviations in the key words:

        ``a`` = approximation,

        ``d`` = detail

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients when `axes` option is used.

    axes :
   Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed, on the chosen wavelet basis, on the used boundary
        condition and on the defined axes.
    """
    keys = list(coeffs.keys())
    keys.sort()
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    num_dcoeffs = len(keys) - 1

    flat_total_size = flat_sizes[0] + num_dcoeffs * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]

    details = tuple(coeffs[key] for key in keys if 'd' in key)
    coeff_list = []
    coeff_list.append(details)
    coeff_list.append(coeffs[keys[0]])
    coeff_list.reverse()
    flat_coeff[start:stop] = coeffs[keys[0]].ravel()
    for fsize, detail_coeffs in zip(flat_sizes[1:], coeff_list[1:]):
        for dcoeff in detail_coeffs:
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def pywt_list_to_array(coeff, size_list):
    """Convert a Pywavelets coefficient list into a flat array.

    Related to 1D, 2D and 3D multilevel discrete wavelet transforms.

    Parameters
    ----------
    coeff : ordered list
        Coefficient are organized in the list in the following way:

        In 1D:

        ``[aN, (dN), ..., (d1)]``

        The abbreviations refer to

        ``a`` = approximation,

        ``d`` = detail

        In 2D:

        ``[aaN, (adN, daN, ddN), ..., (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations refer to

        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        ``N`` refers to the number of scaling levels

    size_list : list
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels.

        ``size_list[0]`` = size of approximation coefficients at
        the coarsest level,

        ``size_list[1]`` = size of the detailed coefficients at
        the coarsest level,

        ``size_list[N]`` = size of the detailed coefficients at
        the finest level,

        ``size_list[N+1]`` = size of original image,

        ``N`` =  the number of scaling levels

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
      """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    flat_total_size = flat_sizes[0] + dcoeffs_per_scale * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    start = 0
    stop = flat_sizes[0]
    flat_coeff[start:stop] = coeff[0].ravel()

    if dcoeffs_per_scale == 1:
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = detail_coeffs.ravel()
    elif dcoeffs_per_scale == 3:
        for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
            for dcoeff in detail_coeffs:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()
    elif dcoeffs_per_scale == 7:
        for ind in range(1, len(size_list) - 1):
            detail_coeffs_dict = coeff[ind]
            keys = list(detail_coeffs_dict.keys())
            keys.sort()
            details = tuple(detail_coeffs_dict[key] for key in
                            keys if 'd' in key)
            fsize = flat_sizes[ind]
            for dcoeff in details:
                start, stop = stop, stop + fsize
                flat_coeff[start:stop] = dcoeff.ravel()

    return flat_coeff


def array_to_pywt_dict(coeff, size_list, axes):
    """Convert a flat array into a PyWavelet coefficient dictionary.

    For 2D and 3D discrete wavelet transform with `axes` option.
    Computing 1D wavelet transform multiple times along the axis
    correspond to computing 1D multilevel wavelet transform

    Parameters
    ----------
    coeff : `DiscreteLpVector`
        A flat coefficient vector containing the approximation,
        and detail coefficients

    size_list : list
       A list of wavelet coefficient sizes.

    Returns
    -------
    coeff_dict : an ordered `dict` . In the key words following
        appreviations are used

        ``a`` = approximation,

        ``d`` = detail
    """
    rep = len(axes)
    keys = list(''.join(k) for k in product('ad', repeat=rep))
    num_coeffs = len(keys)
    shape = size_list[1]
    values = tuple(c.reshape(shape) for c in
                   np.split(np.asarray(coeff), num_coeffs))
    coeff_dict = {key: value for key, value in zip(keys, values)}

    return coeff_dict


def array_to_pywt_list(coeff, size_list):
    """Convert a flat array into a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient list.

    For multilevel 1D, 2D and 3D discrete wavelet transforms.

    Parameters
    ----------
    coeff : `DiscreteLpElement`
        A flat coefficient vector containing the approximation,
        and detail coefficients in the following order
        [aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN, ...
        aad1, ada1, add1, daa1, dad1, dda1, ddd1]

    size_list : list
       A list of coefficient sizes such that,

       ``size_list[0]`` = size of approximation coefficients at the coarsest
                          level,

       ``size_list[1]`` = size of the detailedetails at the coarsest level,

       ``size_list[N]`` = size of the detailed coefficients at the finest
                          level,

       ``size_list[N+1]`` = size of original image,

       ``N`` =  the number of scaling levels

    Returns
    -------
    coeff : ordered list
        Coefficient are organized in the list in the following way:

        In 1D:

        ``[aN, (dN), ... (d1)]``

        The abbreviations refer to

        ``a`` = approximation,

        ``d`` = detail,

        In 2D:

        ``[aaN, (adN, daN, ddN), ... (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        In 3D:

        ``[aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)]``

        The abbreviations refer to

        ``aaa`` = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``aad`` = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``ada`` = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        ``add`` = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        ``daa`` = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        ``dad`` = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ``dda`` = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ``ddd`` = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        ``N`` refers to the number of scaling levels

    """
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]
    start = 0
    stop = flat_sizes[0]
    coeff_list = [np.asarray(coeff)[start:stop].reshape(size_list[0])]
    ndim = len(size_list[0])
    dcoeffs_per_scale = 2 ** ndim - 1

    if dcoeffs_per_scale == 1:
        for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = np.asarray(coeff)[start:stop]
            coeff_list.append(detail_coeffs)
    elif ndim == 2:
        for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(np.asarray(coeff)[start:stop],
                                           dcoeffs_per_scale))
            coeff_list.append(detail_coeffs)
    elif ndim == 3:
        for ind in range(1, len(size_list) - 1):
            fsize = flat_sizes[ind]
            shape = size_list[ind]
            start, stop = stop, stop + dcoeffs_per_scale * fsize
            detail_coeffs = tuple(c.reshape(shape) for c in
                                  np.split(np.asarray(coeff)[start:stop],
                                           dcoeffs_per_scale))
            (aad, ada, add, daa, dad, dda, ddd) = detail_coeffs
            coeff_dict = {'aad': aad, 'ada': ada, 'add': add,
                          'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}
            coeff_list.append(coeff_dict)

    return coeff_list


class WaveletTransform(Operator):

    """Discrete wavelet transform between discrete Lp spaces."""

    def __init__(self, domain, wbasis, pad_mode, nscales=None, axes=None):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.

        wbasis :  {`str`, ``pywt.Wavelet``}
            If a string is given, converts to a ``pywt.Wavelet``.
            Describes properties of a selected wavelet basis.
            See PyWavelet `documentation
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_

            Possible wavelet families are:

            Haar (``haar``)

            Daubechies (``db``)

            Symlets (``sym``)

            Coiflets (``coif``)

            Biorthogonal (``bior``)

            Reverse biorthogonal (``rbio``)

            Discrete FIR approximation of Meyer wavelet (``dmey``)

        pad_mode : string
             Signal extention modes as defined by ``pywt.MODES.modes``
             http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

             Possible extension modes are:

            'zero': zero-padding -- signal is extended by adding zero samples

            'constant': constant padding -- border values are replicated

            'symmetric': symmetric padding -- signal extension by
            mirroring samples

            'periodic': periodic padding -- signal is trated as a periodic one

            'smooth': smooth padding -- signal is extended according to the
            first derivatives calculated on the edges (straight line)

            'periodization': periodization -- like periodic-padding but gives
            the smallest possible number of decomposition coefficients.

        nscales : `int`, optional
            Number of scales in the coefficient list.
            The maximum number of usable scales can be determined
            by ``pywt.dwt_max_level``. For more information see
            the corresponding `documentation of PyWavelets
            <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .
            `nscales` option cannot be combined with `axes` option:
            if `nscales=None` `axes` has to be given.

        axes : sequence of `int`, optional
            Dimensions in which to calculate the wavelet transform.
            The sequence's length has to be equal to dimension of the ``grid``
            `None` means traditional transform along the axes in ``grid``.
            `axes` option cannot be combined with `nscales` option:
            if `axes=None` `nscales` has to be given.

        Examples
        --------
        >>> import odl, pywt
        >>> wbasis = pywt.Wavelet('db1')
        >>> discr_domain = odl.uniform_discr([0, 0], [1, 1], (16, 16))
        >>> op = WaveletTransform(discr_domain, nscales=1,
        ...                       wbasis=wbasis, pad_mode='per')
        >>> op.is_biorthogonal
        True
        """
        self.pad_mode = str(pad_mode).lower()

        if isinstance(wbasis, pywt.Wavelet):
            self.wbasis = wbasis
        else:
            self.wbasis = pywt.Wavelet(wbasis)

        if not isinstance(domain, DiscreteLp):
            raise TypeError('`domain` {!r} is not a `DiscreteLp` instance.'
                            ''.format(domain))

        if domain.exponent != 2.0:
            raise ValueError('`domain` Lp exponent is {} instead of 2.0.'
                             ''.format(domain.exponent))

        if domain.ndim not in [1, 2, 3]:
            raise NotImplementedError('Dimension of the domain {} not 1, '
                                      '2 or 3'.format(len(domain.ndim)))

        if axes is not None:
            if nscales is not None:
                raise ValueError('Cannot use both nscales and axes options '
                                 'at the same time, set other to None')

            else:
                self.nscales = None
            if domain.ndim == 1:
                raise ValueError('Wavelet transform in 1D multiple times '
                                 'along the axis corresponds to 1D multilevel '
                                 'wavelet transform. Set axes to  None and '
                                 'nscales to {}.'.format(len(axes)))

            self.axes = tuple(int(ax) for ax in axes)
            max_level = pywt.dwt_max_level(domain.shape[0],
                                           filter_len=self.wbasis.dec_len)

            axes_counts = [axes.count(i) for i in range(domain.ndim)]
            for i in range(len(axes_counts)):
                if axes_counts[i] > max_level:
                    raise ValueError('Wavelet transforms per axes cannot be '
                                     'performed more than maximum useful '
                                     'level computed by pywt.dwt_max_level. '
                                     'Max level here is {}.'.format(max_level))

            self.size_list = coeff_size_list(domain.shape, self.wbasis,
                                             self.pad_mode, nscales=None,
                                             axes=self.axes)

            rep = len(axes)
            keys = list(''.join(k) for k in product('ad', repeat=rep))
            num_coeffs = len(keys)
            ran_size = num_coeffs * np.prod(self.size_list[0])

        elif nscales is not None:
            self.axes = None
            self.nscales = int(nscales)
            max_level = pywt.dwt_max_level(domain.shape[0],
                                           filter_len=self.wbasis.dec_len)
            if self.nscales > max_level:
                raise ValueError('Cannot use more than {} scaling levels, '
                                 'got {}. Maximum useful number of levels '
                                 'can be computed using pywt.dwt_max_level '
                                 ''.format(max_level, self.nscales))
            self.size_list = coeff_size_list(
                domain.shape, self.wbasis, self.pad_mode, self.nscales,
                axes=None)

            multiplicity = {1: 1, 2: 3, 3: 7}
            ran_size = (np.prod(self.size_list[0]) +
                        sum(multiplicity[domain.ndim] * np.prod(shape)
                            for shape in self.size_list[1:-1]))

        else:
            raise ValueError('Either `nscales` or `axes` has to be given')

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        ran = domain.dspace_type(ran_size, dtype=domain.dtype)
        super().__init__(domain, ran, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, x):
        """Compute the discrete wavelet transform.

        Parameters
        ----------
        x : `domain` element

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
        """
        if self.axes is None:
            if self.domain.ndim == 1:
                coeff_list = pywt.wavedec(x, self.wbasis, self.pad_mode,
                                          self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)
                return self.range.element(coeff_arr)

            if self.domain.ndim == 2:
                coeff_list = pywt.wavedec2(x, self.wbasis, self.pad_mode,
                                           self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)
                return self.range.element(coeff_arr)

            if self.domain.ndim == 3:
                coeff_list = pywt.wavedecn(x, self.wbasis, self.pad_mode,
                                           self.nscales)
                coeff_arr = pywt_list_to_array(coeff_list, self.size_list)

                return self.range.element(coeff_arr)
        else:
            coeff_dict = pywt.dwtn(x, self.wbasis, self.pad_mode, self.axes)
            coeff_arr = pywt_dict_to_array(coeff_dict, self.size_list,
                                           self.axes)
            return self.range.element(coeff_arr)

    @property
    def adjoint(self):
        """Adjoint wavelet transform.

        Returns
        -------
        adjoint : `WaveletTransformInverse`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            If `is_orthogonal` is not true, the adjoint is not implemented.
        """
        if self.is_orthogonal:
            output = self.inverse
            output /= self.domain.cell_volume
            return output
            #return self.inverse
        elif self.wbasis.name.startswith('bior'):
            adjoint_name = self.wbasis.name.replace('bior', 'rbio')
            wbasis_adjoint = pywt.Wavelet(adjoint_name)
            output = WaveletTransformInverse(
                ran=self.domain, wbasis=wbasis_adjoint, mode=self.mode,
                nscales=self.nscales, axes=self.axes)
            output /= self.domain.cell_volume
            return output
        else:
            return super().adjoint

    @property
    def inverse(self):
        """Inverse wavelet transform.

        Returns
        -------
        inverse : `WaveletTransformInverse`

        See Also
        --------
        adjoint
        """
        return WaveletTransformInverse(
            range=self.domain, wbasis=self.wbasis, pad_mode=self.pad_mode,
            nscales=self.nscales, axes=self.axes)


class WaveletTransformInverse(Operator):

    """Discrete inverse wavelet tranform between discrete Lp spaces."""

    def __init__(self, range, wbasis, mode, nscales=None, axes=None):
        """Initialize a new instance.

         Parameters
        ----------
        range : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.

        wbasis :  ``pywt.Wavelet``
            Describes properties of a selected wavelet basis.
            See PyWavelet `documentation
            <http://www.pybytes.com/pywavelets/ref/wavelets.html>`_

            Possible wavelet families are:

            Haar (``haar``)

            Daubechies (``db``)

            Symlets (``sym``)

            Coiflets (``coif``)

            Biorthogonal (``bior``)

            Reverse biorthogonal (``rbio``)

            Discrete FIR approximation of Meyer wavelet (``dmey``)

        pad_mode : string
             Signal extention modes as defined by ``pywt.MODES.modes``
             http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

             Possible extension modes are:

            'zero': zero-padding -- signal is extended by adding zero samples

            'constant': constant padding -- border values are replicated

            'symmetric': symmetric padding -- signal extension by
            mirroring samples

            'periodic': periodic padding -- signal is trated as a periodic one

            'smooth': smooth padding -- signal is extended according to the
            first derivatives calculated on the edges (straight line)

            'periodization': periodization -- like periodic-padding but gives
            the smallest possible number of decomposition coefficients.

        nscales : `int`, optional
            Number of scales in the coefficient list.
            The maximum number of usable scales can be determined
            by ``pywt.dwt_max_level``. For more information see
            the corresponding `documentation of PyWavelets
            <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_ .
            `nscales` option cannot be combined with `axes` option:
            if `nscales=None` `axes` has to be given.

        axes : sequence of `int`, optional
            Dimensions in which to calculate the wavelet transform.
            The sequence's length has to be equal to dimension of the ``grid``
            `None` means traditional transform along the axes in ``grid``.
            `axes` option cannot be combined with `nscales` option:
            if `axes=None` `nscales` has to be given.
        """
        from builtins import range as range_seq

        if nscales is None and axes is None:
            raise ValueError('Either nscales or axes has to be defined')
        elif nscales is not None and axes is not None:
            raise ValueError('Cannot use both nscales and axes options '
                             ' at the same time, set other to None')

        self.pad_mode = str(mode).lower()

        if isinstance(wbasis, pywt.Wavelet):
            self.wbasis = wbasis
        else:
            self.wbasis = pywt.Wavelet(wbasis)

        if not isinstance(range, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(range))

        if range.exponent != 2.0:
            raise ValueError('domain Lp exponent is {} instead of 2.0.'
                             ''.format(range.exponent))

        if range.ndim not in [1, 2, 3]:
            raise NotImplementedError('Dimension of the domain {} not 1, '
                                      '2 or 3'.format(len(range.ndim)))

        if axes is not None:
            if nscales is not None:
                raise ValueError('Cannot use both nscales and axes options '
                                 'at the same time, set other to None')
            self.nscales = None
            if range.ndim == 1:
                raise ValueError('Wavelet transform in 1D multiple times '
                                 'along the axis corresponds to 1D multilevel '
                                 'wavelet transform. Set axes to None and '
                                 'nscales to {}.'.format(len(axes)))

            self.axes = tuple(int(ax) for ax in axes)
            max_level = pywt.dwt_max_level(range.shape[0],
                                           filter_len=self.wbasis.dec_len)

            axes_counts = [axes.count(i) for i in range_seq(range.ndim)]
            for i in range_seq(len(axes_counts)):
                if axes_counts[i] > max_level:
                    raise ValueError('Wavelet transforms per axes cannot be '
                                     'performed more than maximum useful '
                                     'level computed by pywt.dwt_max_level. '
                                     'Max level here is {}.'.format(max_level))

            self.size_list = coeff_size_list(range.shape, self.wbasis,
                                             self.pad_mode, nscales=None,
                                             axes=self.axes)

            rep = len(axes)
            keys = list(''.join(k) for k in product('ad', repeat=rep))
            num_coeffs = len(keys)
            dom_size = num_coeffs * np.prod(self.size_list[0])

        elif nscales is not None:
            self.axes = None
            self.nscales = int(nscales)
            max_level = pywt.dwt_max_level(range.shape[0],
                                           filter_len=self.wbasis.dec_len)
            if self.nscales > max_level:
                raise ValueError('Cannot use more than {} scaling levels, '
                                 'got {}. Maximum useful number of levels '
                                 'can be computed using pywt.dwt_max_level '
                                 ''.format(max_level, self.nscales))
            self.size_list = coeff_size_list(
                range.shape, self.wbasis, self.pad_mode, self.nscales,
                axes=None)

            multiplicity = {1: 1, 2: 3, 3: 7}
            dom_size = (np.prod(self.size_list[0]) +
                        sum(multiplicity[range.ndim] * np.prod(shape)
                            for shape in self.size_list[1:-1]))

        else:
            raise ValueError('Either `nscales` or `axes` has to be given')

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        domain = range.dspace_type(dom_size, dtype=range.dtype)
        super().__init__(domain, range, linear=True)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, coeff):
        """Compute the discrete 1D, 2D or 3D inverse wavelet transform."""
        if self.axes is None:
            if len(self.range.shape) == 1:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverec(coeff_list, self.wbasis, self.pad_mode)
                return self.range.element(x)
            elif len(self.range.shape) == 2:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverec2(coeff_list, self.wbasis, self.pad_mode)
                return self.range.element(x)
            elif len(self.range.shape) == 3:
                coeff_list = array_to_pywt_list(coeff, self.size_list)
                x = pywt.waverecn(coeff_list, self.wbasis, self.pad_mode)
                return x

        else:
            coeff_dict = array_to_pywt_dict(coeff, self.size_list, self.axes)
            x = pywt.idwtn(coeff_dict, self.wbasis, self.pad_mode, self.axes)
            return x

    @property
    def adjoint(self):
        """Adjoint of this operator.

        Returns
        -------
        adjoint : `WaveletTransform`
            If the transform is orthogonal, the adjoint is the inverse.

        Raises
        ------
        OpNotImplementedError
            If `is_orthogonal` is not true, the adjoint is not implemented.

        See Also
        --------
        inverse
        """
        if self.is_orthogonal:
            return self.inverse
        elif self.wbasis.name.startswith('bior'):
            adjoint_name = self.wbasis.name.replace('bior', 'rbio')
            wbasis_adjoint = pywt.Wavelet(adjoint_name)
            return WaveletTransform(dom=self.range, wbasis=wbasis_adjoint,
                                    mode=self.mode, nscales=self.nscales,
                                    axes=self.axes)
        else:
            return super().adjoint

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        return WaveletTransform(domain=self.range, wbasis=self.wbasis,
                                pad_mode=self.pad_mode, nscales=self.nscales,
                                axes=self.axes)

if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
