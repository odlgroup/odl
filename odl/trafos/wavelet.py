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

"""Discrete wavelet transformation on L2 spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range, str, super

# External
import numpy as np
import pywt

# Internal


__all__ = ('DiscreteWaveletTrafo', 'DiscreteWaveletTrafoAdjoint',
           'DiscreteWaveletTrafoInverse')


_SUPPORTED_IMPL = ('pywt',)


def list_of_coeff_sizes(shape, nscale, wbasis, mode):
    """Construct a size list from given wavelet coefficients.

    Construct a list containing the sizes of the wavelet approximation
    and detail coefficients. Related to 2D multidimensional wavelet transform.

    Parameters
    ----------
    shape : `tuple`
        Number of pixels in the image. Its lenght must be 2.
    nscale: int
        Number of scales in the multidimensional wavelet
        transform
    wbasis: ``_pywt.Wavelet``
        Describes properties of a selected wavelet basis
    mode: `str`
        Signal extention mode. Possible extension modes are
        'zpd' - zero-padding: signal is extended by adding zero samples
        'cpd' - constant padding: border values are replicated
        'sym' - symmetric padding: signal extension by mirroring samples
        'ppd' - periodic padding: signal is trated as a periodic one
        'sp1' - smooth padding: signal is extended according to the
        first derivatives calculated on the edges (straight line)
        'per' - periodization: like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    Returns
    -------
    S: `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels
        S[0] = size of approximation coefficients at the coarsest level
              size of cAn
        S[1] = size of the detailed coefficients at the coarsest level
              size of cHn/cVn/cDn
        S[n] = size of the detailed coefficients at the finest level
              size of cH1/cV1/cD1, n = nscales
        S[n+1] = size of original image

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    if len(shape) != 2:
        raise ValueError('shape must have length 2, got {}.'
                         ''.format(len(shape)))
    P = np.zeros(shape)
    max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
    if nscale >= max_level:
        raise ValueError('Too many scaling levels')

    W = pywt.wavedec2(P, wbasis, mode, level=nscale)
    S = []
    a = np.shape(W[0])
    S.append(a)
    for kk in range(1, nscale+1):
        (wave_horz, wave_vert, wave_diag) = W[kk]
        a = np.shape(wave_horz)
        S.append(a)
    S.append(shape)
    return S


def pywt_coeff_to_array2d(coeff, size_list, nscales):
    """Convert a pywt coefficient into a flat array.
    Related to 2D multilevel discrete wavelet transform.

    Parameters
    ----------
    coeff : `list`
        Coefficient are organized in the list in the following way
        [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)]
        where  cA, cH, cV, cD denote approximation, horizontal detail,
        vertical detail and diagonal detail coefficients respectively
        and n denotes the level of decomposition/number of scales.
    size_list: `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels
        size_list[0] = size of approximation coefficients at
            the coarsest level, i.e. size of cAn
        size_list[1] = size of the detailed coefficients at
            the coarsest level, i.e.  size of cHn/cVn/cDn
        size_list[n] = size of the detailed coefficients at
            the finest level, i.e. size of cH1/cV1/cD1,
            n = nscales
        size_list[n+1] = size of original image
    nscales : int
        Number of scales in the coefficient list

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
        If the size of the input image is 2^n x 2^n, the lenght of the
        wavelet coefficient array is the same.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    # TODO: outsource to an own helper?
    size_flatCoeff = np.prod(size_list[0])
    for kk in range(1, nscales+1):
        size_flatCoeff = size_flatCoeff + 3*np.prod(size_list[kk])

    flat_coeff = np.zeros((size_flatCoeff))
    approx = coeff[0]
    approx = approx.ravel()
    stop = np.prod(size_list[0])
    flat_coeff[0:stop] = approx
    start = stop
    for kk in range(1, nscales+1):
        (wave_horz, wave_vert, wave_diag) = coeff[kk]
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = wave_horz.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = wave_vert.ravel()
        start = stop
        stop = start + np.prod(size_list[kk])
        flat_coeff[start:stop] = wave_diag.ravel()
        start = stop
    return flat_coeff


def array_to_pywt_coeff2d(coeff, size_list, nscales):
    """Convert a flat array into a pywt coefficient list.
    For multilevel 2D discrete wavelet transform

    Parameters
    ----------
    coeff:  :class:`DiscreteLp.Vector`
        A flat coefficient vector containing the approximation,
        horizontal detail, vertical detail and
        diagonal detail coefficients in the following order
        [cAn, cHn, cVn, cDn, ... cH1, cV1, cD1]
    size_list: list
       A list of coefficient sizes such that
       size_list[0] = size of approximation coefficients at the coarsest level
       size_list[1] = size of the detailed coefficients at the coarsest level
       size_list[n] = size of the detailed coefficients at the finest level
                  n = nscales
       size_list[n+1] = size of original image
    nscales : int
        Number of scales in the coefficient list

    Returns
    -------
    :attr:`coeff_list` : `list`
        A list of coefficient organized in the following way
        [cAn, (cHn, cVn, cDn), ... (cH1, cV1, cD1)]
        where  cA, cH, cV, cD denote approximation, horizontal detail,
        vertical detail and diagonal detail coefficients respectively
        and n denotes the level of decomposition/number of scales.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    size1 = size_list[0][0] * size_list[0][1]
    approx_flat = coeff[0:size1]
    approx = np.asarray(approx_flat).reshape(
        [size_list[0][0], size_list[0][1]])
    kk = 1
    coeff_list = []
    coeff_list.append(approx)

    while kk <= nscales:
        detail_shape = size_list[kk]
        size2 = size_list[kk][0] * size_list[kk][1]
        wave_horz_flat = coeff[size1:size1+size2]
        wave_horz = np.asarray(wave_horz_flat).reshape(detail_shape)

        wave_vert_flat = coeff[size1+size2:size1+2*size2]
        wave_vert = np.asarray(wave_vert_flat).reshape(detail_shape)

        wave_diag_flat = coeff[size1+2*size2:size1 + 3*size1]
        wave_diag = np.asarray(wave_diag_flat).reshape(detail_shape)

        details = (wave_horz, wave_vert, wave_diag)
        coeff_list.append(details)

        kk = kk + 1
        size1 = size1 + 3*size2

    return coeff_list


class DiscreteWaveletTrafo(odl.Operator):

    """Discrete 2D wavelet trafo between discrete L2 spaces."""

    def __init__(self, dom, nscales, wbasis, mode):
        """Initialize a new instance.

        Parameters
        ----------
        dom : :class:`~odl.DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.
        nscales : `int`
            Number of scales in the coefficient list
        wbasis:  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode: `str`
            Signal extention mode. For possible extensions see
            ``pywt.MODES.modes``

        See also
        --------
        pywt : http://www.pybytes.com/pywavelets/
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.mode = str(mode).lower()

        max_level = pywt.dwt_max_level(dom.grid.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales >= max_level:
            raise ValueError('Cannot use more than {} scaling levels, '
                             'got {}.'.format(max_level, self.nscales))

        self.size_list = list_of_coeff_sizes(dom.grid.shape, self.nscales,
                                             self.wbasis, self.mode)

        ran_size = np.prod(self.size_list[0])
        ran_size += sum(3 * np.prod(shape) for shape in self.size_list[1:-1])
        #`pywt` does not handle complex values
        # >>ComplexWarning: Casting complex values to real discards the imaginary part

        # TODO: Maybe allow other ranges like Besov spaces (yet to be crated)
        ran = dom.dspace_type(ran_size, dtype=dom.dtype)
        super().__init__(dom, ran)

        if not isinstance(dom, odl.DiscreteL2):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

#        if dom.exponent != 2.0:
#            raise ValueError('domain Lp exponent is {} instead of 2.0.'
#                             ''.format(dom.exponent))
#        if not np.all(dom.grid.stride == 1):
#            raise NotImplementedError('non-uniform grid cell sizes not yet '
#                                      'supported.')

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, x):
        """Compute the discrete 2D multiresolution wavelet transform
        Parameters
        ----------
        x: `DiscreteLp.Vector`

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
            If the size of the input image is 2^n x 2^n, the lenght of the
            wavelet coefficient array is the same.
        """
        # TODO: check if one needs to write x.asarray()
        coeff_list = pywt.wavedec2(x, self.wbasis, self.mode, self.nscales)
        coeff_arr = pywt_coeff_to_array2d(coeff_list, self.size_list,
                                          self.nscales)
        return self.range.element(coeff_arr)

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return None

    @property
    def inverse(self):
        """The inverse wavelet transform."""

        return DiscreteWaveletTrafoInverse(ran=self.domain,
                                           nscales=self.nscales,
                                           wbasis=self.wbasis, mode=self.mode)


class DiscreteWaveletTrafoAdjoint(odl.Operator):
    pass


class DiscreteWaveletTrafoInverse(odl.Operator):

    """Discrete inverse wavelet trafo between discrete L2 spaces."""

    def __init__(self, ran, nscales, wbasis, mode):
        """Initialize a new instance.

        Parameters
        ----------
        ran : `odl.DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent `p` of the discrete :math:`L^p`
            space must be equal to 2.0.
        nscales : int
            Number of scales in the coefficient list
        wbasis:  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode: `str`
            Signal extention mode. For possible extensions see
            ``pywt.MODES.modes``
        See also
        --------
        pywt : http://www.pybytes.com/pywavelets/
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.mode = str(mode).lower()

        max_level = pywt.dwt_max_level(ran.grid.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales >= max_level:
            raise ValueError('Cannot use more than {} scaling levels, '
                             'got {}.'.format(max_level, self.nscales))

        self.size_list = list_of_coeff_sizes(ran.grid.shape, self.nscales,
                                             self.wbasis, self.mode)

        dom_size = np.prod(self.size_list[0])
        dom_size += sum(3 * np.prod(shape) for shape in self.size_list[1:-1])
        # TODO: Check if complex spaces work
        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        dom = ran.dspace_type(dom_size, dtype=ran.dtype)
        super().__init__(dom, ran)

        if not isinstance(ran, odl.DiscreteL2):
            raise TypeError('range {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))
#
#        if dom.exponent != 2.0:
#            raise ValueError('domain Lp exponent is {} instead of 2.0.'
#                             ''.format(dom.exponent))
#        if ran.exponent != 2.0:
#            raise ValueError('range Lp exponent is {} instead of 2.0.'
#                             ''.format(ran.exponent))
#        if not np.all(dom.grid.stride == 1):
#            raise NotImplementedError('non-uniform grid cell sizes not yet '
#                                      'supported.')

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, coeff):
        """Compute the discrete 2D inverse multiresolution wavelet transform

        Parameters
        ----------
        coeff: class:`DiscreteLp.Vector` (or `list`??)

        Returns
        -------
        arr : `numpy.ndarray` or `DiscreteLp.Vector`

        """
        coeff_list = array_to_pywt_coeff2d(coeff, self.size_list, self.nscales)
        x = pywt.waverec2(coeff_list, self.wbasis, self.mode)
        return self.range.element(x)


    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return None

    @property
    def inverse(self):
        """The inverse wavelet transform."""

        return DiscreteWaveletTrafo(dom=self.range, nscales = self.nscales,
                                    wbasis = self.wbasis, mode = self.mode)
