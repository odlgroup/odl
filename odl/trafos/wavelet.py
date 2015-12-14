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

# Internal
from odl.discr.lp_discr import DiscreteLp
from odl.operator.operator import Operator

try:
    import pywt
    PYWAVELETS_AVAILABLE = True
except ImportError:
    PYWAVELETS_AVAILABLE = False

__all__ = ('DiscreteWaveletTransform', 'DiscreteWaveletTransformInverse',
           'PYWAVELETS_AVAILABLE')


_SUPPORTED_IMPL = ('pywt',)


def coeff_size_list(shape, nscales, wbasis, mode):
    """Construct a size list from given wavelet coefficients.

    Related to 2D and 3D multidimensional wavelet transforms that utilize
    `PyWavelets
    <http://www.pybytes.com/pywavelets/>`_.

    Parameters
    ----------
    shape : `tuple`
        Number of pixels/voxels in the image. Its length must be 2 or 3.
    nscales : `int`
        Number of scales in the multidimensional wavelet
        transform.  This parameter is checked against the maximum number of
        scales returned by ``pywt.dwt_max_level``. For more information
        see the corresponding `documentation
        <http://www.pybytes.com/pywavelets/ref/\
dwt-discrete-wavelet-transform.html#maximum-decomposition-level\
-dwt-max-level>`_
        of PyWavelets.

    wbasis : ``pywt.Wavelet``
        Selected wavelet basis

    mode : `str`
        Signal extention mode. Possible extension modes are

        'zpd': zero-padding -- signal is extended by adding zero samples

        'cpd': constant padding -- border values are replicated

        'sym': symmetric padding -- signal extension by mirroring samples

        'ppd': periodic padding -- signal is trated as a periodic one

        'sp1': smooth padding -- signal is extended according to the
        first derivatives calculated on the edges (straight line)

        'per': periodization -- like periodic-padding but gives the
        smallest possible number of decomposition coefficients.

    Returns
    -------
    size_list : `list`
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
    if len(shape) not in (2, 3):
        raise ValueError('shape must have length 2 or 3, got {}.'
                         ''.format(len(shape)))

    max_level = pywt.dwt_max_level(shape[0], filter_len=wbasis.dec_len)
    if nscales >= max_level:
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
        shp = tuple(pywt.dwt_coeff_len(n, filter_len=wbasis.dec_len, mode=mode)
                    for n in size_list[scale])
        size_list.append(shp)

    # Add a duplicate of the last entry for the approximation coefficients
    size_list.append(size_list[-1])

    # We created the list in reversed order compared to what pywt expects
    size_list.reverse()
    return size_list


def pywt_coeff_to_array2d(coeff, size_list):
    """Convert a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient into a flat array.

    Related to 2D multilevel discrete wavelet transform.

    Parameters
    ----------
    coeff : ordered `list`
        Coefficient are organized in the list in the following way:

        ``[aaN, (adN, daN, ddN), ... (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        where ``N`` is the number of scales.

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels,

        ``size_list[0]`` = size of approximation coefficients at
            the coarsest level, i.e. size of aaN,

        ``size_list[1]`` = size of the detailed coefficients at
            the coarsest level, i.e.  size of adN/daN/ddN,

        ``size_list[N]`` = size of the detailed coefficients at
            the finest level, i.e. size of ad1/da1/dd1,

        ``size_list[N+1]`` = size of original image,

        where ``N`` is the number of scales.

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array.
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
        If the size of the input image is :math:`2^n \\times 2^n`,
        the length of the wavelet coefficient array is the same.
    """
    # Last entry not used since it does not correspond to a coefficient size
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]

    # Approx. coefficients appear once, while there are 3 types of detail
    # coefficients per level
    flat_total_size = flat_sizes[0] + 3 * sum(flat_sizes[1:])
    flat_coeff = np.empty(flat_total_size)

    # Special case at index 0 since the entry is a single array, not a tuple
    start = 0
    stop = flat_sizes[0]
    flat_coeff[:flat_sizes[0]] = coeff[0].ravel()

    for fsize, detail_coeffs in zip(flat_sizes[1:], coeff[1:]):
        for dc in detail_coeffs:
            start, stop = stop, stop + fsize
            flat_coeff[start:stop] = dc.ravel()

    return flat_coeff


def array_to_pywt_coeff2d(coeff, size_list):
    """Convert a flat array into a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient list.
    For multilevel 2D discrete wavelet transform

    Parameters
    ----------
    coeff : `DiscreteLp.Vector`
        A flat coefficient vector containing the approximation,
        horizontal detail, vertical detail and diagonal detail
        coefficients in the following order:

        ``[aaN, adN, daN, ddN, ... ad1, da1, dd1]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        where ``N`` is the number of scales.

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels,

        ``size_list[0]`` = size of approximation coefficients at
            the coarsest level, i.e. size of aaN,

        ``size_list[1]`` = size of the detail coefficients at
            the coarsest level, i.e.  size of adN/daN/ddN,

        ``size_list[N]`` = size of the detail coefficients at
            the finest level, i.e. size of ad1/da1/dd1,

        ``size_list[N+1]`` = size of original image,

        where ``N`` is the number of scales.

    Returns
    -------
    coeff : ordered `list`
        Coefficient are organized in the list in the following way:

        ``[aaN, (adN, daN, ddN), ... (ad1, da1, dd1)]``

        The abbreviations refer to

        ``aa`` = approx. on 1st dim, approx. on 2nd dim (approximation),

        ``ad`` = approx. on 1st dim, detail on 2nd dim (horizontal),

        ``da`` = detail on 1st dim, approx. on 2nd dim (vertical),

        ``dd`` = detail on 1st dim, detail on 2nd dim (diagonal),

        where ``N`` is the number of scales.
    """
    # Last entry not used since it does not correspond to a coefficient size
    flat_sizes = [np.prod(shp) for shp in size_list[:-1]]

    # Special case at index 0 since the entry is a single array, not a tuple
    start = 0
    stop = flat_sizes[0]
    coeff_list = [np.asarray(coeff)[start:stop].reshape(size_list[0])]

    for fsize, shape in zip(flat_sizes[1:], size_list[1:]):
        start, stop = stop, stop + 3 * fsize
        # Split section of the flat array into 3 parts, each corresponding
        # to a detail coefficient. Reshape according to shape in size_list.
        detail_coeffs = tuple(c.reshape(shape)
                              for c in np.split(np.asarray(coeff)[start:stop],
                                                3))
        coeff_list.append(detail_coeffs)

    return coeff_list


def pywt_coeff_to_array3d(coeff, size_list, nscales):
    """Convert a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient into a flat array.

    Related to 3D multilevel discrete wavelet transform.

    Parameters
    ----------
    coeff : `list`
        Coefficient are organized in the list in the following way
        [aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)].
        The appreviations refer to

        aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        N =  the number of scaling levels

    size_list : `list`
        A list containing the sizes of the wavelet (approximation
        and detail) coefficients at different scaling levels

        size_list[0] = size of approximation coefficients at
            the coarsest level,
        size_list[1] = size of the detailed coefficients at
            the coarsest level,
        size_list[N] = size of the detailed coefficients at
            the finest level,
        size_list[N+1] = size of original image,
        N = number of scales

    nscales : `int`
        Number of scales in the coefficient list

    Returns
    -------
    arr : `numpy.ndarray`
        Flattened and concatenated coefficient array
        The length of the array depends on the size of input image to
        be transformed and on the chosen wavelet basis.
        If the size of the input image is 2^n x 2^n, the length of the
        wavelet coefficient array is the same.
    """
    size_flatCoeff = np.prod(size_list[0])
    for kk in range(1, nscales + 1):
        size_flatCoeff += 7 * np.prod(size_list[kk])

    flat_coeff = np.empty(size_flatCoeff)
    approx = coeff[0]
    start = 0
    stop = np.prod(size_list[0])
    flat_coeff[start:stop] = approx.ravel()

    for kk in range(1, nscales + 1):
        for subarray in coeff[kk]:
            start, stop = stop, stop + np.prod(size_list[kk])
            flat_coeff[start:stop] = subarray.ravel()

    return flat_coeff


def array_to_pywt_coeff3d(coeff, size_list, nscales):
    """Convert a flat array into a `pywt
    <http://www.pybytes.com/pywavelets/>`_ coefficient list.

    For multilevel 3D discrete wavelet transform

    Parameters
    ----------
    coeff : `DiscreteLp.Vector`
        A flat coefficient vector containing the approximation,
        and detail coefficients in the following order
        [aaaN, aadN, adaN, addN, daaN, dadN, ddaN, dddN, ...
        aad1, ada1, add1, daa1, dad1, dda1, ddd1]

    size_list : list
       A list of coefficient sizes such that,

       size_list[0] = size of approximation coefficients at the coarsest level,

       size_list[1] = size of the detailedetails at the coarsest level,

       size_list[N] = size of the detailed coefficients at the finest level,

       size_list[N+1] = size of original image,

       N = nscales

    nscales : int
        Number of scales in the coefficient list

    Returns
    -------
    coeff_list : `list`
        A list of coefficient organized in the following way
        [aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
        (aad1, ada1, add1, daa1, dad1, dda1, ddd1)].
        The appreviations refer to,

        aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

        add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

        daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

        dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

        dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

        ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

        N = the number of scaling levels
    """

    size1 = np.prod(size_list[0])
    approx_flat = coeff[0:size1]
    approx = np.asarray(approx_flat).reshape(size_list[0])
    kk = 1
    coeff_list = []
    coeff_list.append(approx)

    while kk <= nscales:
        detail_shape = size_list[kk]
        size2 = np.prod(size_list[kk])
        details = []
        for ii in range(1, 8):
            flat = coeff[size1 + (ii - 1) * size2:size1 + ii * size2]
            details += [np.asarray(flat).reshape(detail_shape)]

        coeff_list.append(tuple(details))
        kk = kk + 1
        size1 = size1 + 7 * size2

    return coeff_list


def wavelet_decomposition3d(x, wbasis, mode, nscales):
    """Discrete 3D multiresolution wavelet decomposition.

    Compute the discrete 3D multiresolution wavelet decomposition
    at the given level (nscales) for a given 3D image.
    Utilizes a `n-dimensional PyWavelet
    <http://www.pybytes.com/pywavelets/ref/other-functions.html>`_
    function ``pywt.dwtn``.

    Parameters
    ----------
    x : `DiscreteLp.Vector`
    wbasis:  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
    mode : `str`
            Signal extention mode. For possible extensions see the
            `signal extenstion modes
            <http://www.pybytes.com/pywavelets/ref/\
signal-extension-modes.html>`_
            of PyWavelets.
    nscales : `int`
            Number of scales in the coefficient list

    Returns
    -------
    coeff_list : `list`
        A list of coefficient organized in the following way
         [aaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
         (aad1, ada1, add1, daa1, dad1, dda1, ddd1)] .
         The appreviations refer to,

         aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

         aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

         ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

         add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

         daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

         dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

         dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

         ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

         N = the number of scaling levels
    """

    wcoeffs = pywt.dwtn(x, wbasis, mode)
    aaa = wcoeffs['aaa']
    aad = wcoeffs['aad']
    ada = wcoeffs['ada']
    add = wcoeffs['add']
    daa = wcoeffs['daa']
    dad = wcoeffs['dad']
    dda = wcoeffs['dda']
    ddd = wcoeffs['ddd']

    details = (aad, ada, add, daa, dad, dda, ddd)
    coeff_list = []
    coeff_list.append(details)

    for kk in range(1, nscales):
        wcoeffs = pywt.dwtn(aaa, wbasis, mode)
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


def wavelet_reconstruction3d(coeff_list, wbasis, mode, nscales):
    """Discrete 3D multiresolution wavelet reconstruction

    Compute a discrete 3D multiresolution wavelet reconstruction
    from a given wavelet coefficient list.
    Utilizes a `PyWavelet
    <http://www.pybytes.com/pywavelets/ref/other-functions.html>`_
    function ``pywt.dwtn``

    Parameters
    ----------
    coeff_list : `list`
            A list of wavelet approximation and detail coefficients
            organized in the following way
            [caaaN, (aadN, adaN, addN, daaN, dadN, ddaN, dddN), ...
            (aad1, ada1, add1, daa1, dad1, dda1, ddd1)].
            The appreviations refer to,

            aaa = approx. on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

            aad = approx. on 1st dim, approx. on 2nd dim, detail on 3rd dim,

            ada = approx. on 1st dim, detail on 3nd dim, approx. on 3rd dim,

            add = approx. on 1st dim, detail on 3nd dim, detail on 3rd dim,

            daa = detail on 1st dim, approx. on 2nd dim, approx. on 3rd dim,

            dad = detail on 1st dim, approx. on 2nd dim, detail on 3rd dim,

            dda = detail on 1st dim, detail on 2nd dim, approx. on 3rd dim,

            ddd = detail on 1st dim, detail on 2nd dim, detail on 3rd dim,

            N = the number of scaling levels
    wbasis :  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basisodl.discr.lp_discr.
    mode : `str`
            Signal extention mode. For possible extensions see the
            `signal extenstion modes
            <http://www.pybytes.com/pywavelets/ref/\
signal-extension-modes.html>`_
            of PyWavelets.
    nscales : `int`
            Number of scales in the coefficient list

    Returns
    -------
    x : `numpy.ndarray`.
        A wavalet reconstruction.

    See also
    --------
    pywt : http://www.pybytes.com/pywavelets/
    """
    aaa = coeff_list[0]
    k = 1
    (aad, ada, add, daa, dad, dda, ddd) = coeff_list[k]
    coeff_dict = {'aaa': aaa, 'aad': aad, 'ada': ada, 'add': add,
                  'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}
    for k in range(2, nscales + 1):
        aaa = pywt.idwtn(coeff_dict, wbasis, mode)
        (aad, ada, add, daa, dad, dda, ddd) = coeff_list[k]
        coeff_dict = {'aaa': aaa, 'aad': aad, 'ada': ada, 'add': add,
                      'daa': daa, 'dad': dad, 'dda': dda, 'ddd': ddd}

    x = pywt.idwtn(coeff_dict, wbasis, mode)
    return x


class DiscreteWaveletTransform(Operator):

    """Discrete wavelet trafo between discrete L2 spaces."""

    def __init__(self, dom, nscales, wbasis, mode):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent :math:`p` of the discrete :math:`L^p`
            space must be equal to 2.0.
        nscales : `int`
            Number of scales in the coefficient list
        wbasis :  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode : `str`
            Signal extension mode. For possible extensions see
            ``pywt.MODES.modes``
            http://www.pybytes.com/pywavelets/ref/signal-extension-modes.html

        See also
        --------
        pywt : http://www.pybytes.com/pywavelets/
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.mode = str(mode).lower()

        if not isinstance(dom, DiscreteLp):
            raise TypeError('domain {!r} is not a `DiscreteLp` instance.'
                            ''.format(dom))

        if dom.exponent != 2.0:
            raise ValueError('domain Lp exponent is {} instead of 2.0.'
                             ''.format(dom.exponent))

        max_level = pywt.dwt_max_level(dom.grid.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales >= max_level:
            raise ValueError('Cannot use more than {} scaling levels, '
                             'got {}.'.format(max_level, self.nscales))

        self.size_list = coeff_size_list(dom.grid.shape, self.nscales,
                                         self.wbasis, self.mode)

        ran_size = np.prod(self.size_list[0])

        if len(dom.grid.shape) == 2:
            ran_size += sum(3 * np.prod(shape) for shape in
                            self.size_list[1:-1])
        elif len(dom.grid.shape) == 3:
            ran_size += sum(7 * np.prod(shape) for shape in
                            self.size_list[1:-1])
        else:
            raise NotImplementedError('ndim {} not 2 or 3'
                                      ''.format(len(dom.grid.shape)))

        # TODO: Maybe allow other ranges like Besov spaces (yet to be crated)
        ran = dom.dspace_type(ran_size, dtype=dom.dtype)
        super().__init__(dom, ran)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, x):
        """Compute the discrete 2D and 3D multiresolution wavelet transform
        Parameters
        ----------
        x : `DiscreteLp.Vector`

        Returns
        -------
        arr : `numpy.ndarray`
            Flattened and concatenated coefficient array
            The length of the array depends on the size of input image to
            be transformed and on the chosen wavelet basis.
        """
        if len(x.shape) == 2:
            coeff_list = pywt.wavedec2(x, self.wbasis, self.mode, self.nscales)
            coeff_arr = pywt_coeff_to_array2d(coeff_list, self.size_list)
            return self.range.element(coeff_arr)

        if len(x.shape) == 3:
            coeff_dict = wavelet_decomposition3d(x, self.wbasis, self.mode,
                                                 self.nscales)
            coeff_arr = pywt_coeff_to_array3d(coeff_dict, self.size_list,
                                              self.nscales)

            return self.range.element(coeff_arr)

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return super().adjoint

    @property
    def inverse(self):
        """The inverse wavelet transform."""

        return DiscreteWaveletTransformInverse(
            ran=self.domain, nscales=self.nscales, wbasis=self.wbasis,
            mode=self.mode)


class DiscreteWaveletTransformInverse(Operator):

    """Discrete inverse wavelet trafo between discrete L2 spaces."""

    def __init__(self, ran, nscales, wbasis, mode):
        """Initialize a new instance.

        Parameters
        ----------
        ran : `DiscreteLp`
            Domain of the wavelet transform (the "image domain").
            The exponent `p` of the discrete :math:`L^p`
            space must be equal to 2.0.
        nscales : `int`
            Number of scales in the coefficient list
        wbasis :  ``_pywt.Wavelet``
            Describes properties of a selected wavelet basis
        mode : `str`
            Signal extension mode. For possible extensions see
            ``pywt.MODES.modes``
        See also
        --------
        pywt : http://www.pybytes.com/pywavelets/
        """
        self.nscales = int(nscales)
        self.wbasis = wbasis
        self.mode = str(mode).lower()

        if not isinstance(ran, DiscreteLp):
            raise TypeError('range {!r} is not a `DiscreteLp` instance.'
                            ''.format(ran))

        if ran.exponent != 2.0:
            raise ValueError('range Lp exponent is {} instead of 2.0.'
                             ''.format(ran.exponent))
#        if not np.all(dom.grid.stride == 1):
#            raise NotImplementedError('non-uniform grid cell sizes not yet '
#                                      'supported.')

        max_level = pywt.dwt_max_level(ran.grid.shape[0],
                                       filter_len=self.wbasis.dec_len)
        # TODO: maybe the error message could tell how to calculate the
        # max number of levels
        if self.nscales >= max_level:
            raise ValueError('Cannot use more than {} scaling levels, '
                             'got {}.'.format(max_level, self.nscales))

        self.size_list = coeff_size_list(ran.grid.shape, self.nscales,
                                         self.wbasis, self.mode)

        dom_size = np.prod(self.size_list[0])
        if len(ran.grid.shape) == 2:
            dom_size += sum(3 * np.prod(shape) for shape in
                            self.size_list[1:-1])
        elif len(ran.grid.shape) == 3:
            dom_size += sum(7 * np.prod(shape) for shape in
                            self.size_list[1:-1])
        else:
            raise NotImplementedError('ndim {} not 2 or 3'
                                      ''.format(len(ran.grid.shape)))

        # TODO: Maybe allow other ranges like Besov spaces (yet to be created)
        dom = ran.dspace_type(dom_size, dtype=ran.dtype)
        super().__init__(dom, ran)

    @property
    def is_orthogonal(self):
        """Whether or not the wavelet basis is orthogonal."""
        return self.wbasis.orthogonal

    @property
    def is_biorthogonal(self):
        """Whether or not the wavelet basis is bi-orthogonal."""
        return self.wbasis.biorthogonal

    def _call(self, coeff):
        """Compute the discrete 2D or 3D inverse wavelet transform.

        Parameters
        ----------
        coeff : `DiscreteLp.Vector`

        Returns
        -------
        arr : `DiscreteLp.Vector`

        """
        if len(self.range.grid.shape) == 2:
            coeff_list = array_to_pywt_coeff2d(coeff, self.size_list)
            x = pywt.waverec2(coeff_list, self.wbasis, self.mode)
            return self.range.element(x)
        elif len(self.range.grid.shape) == 3:
            coeff_dict = array_to_pywt_coeff3d(coeff, self.size_list,
                                               self.nscales)
            x = wavelet_reconstruction3d(coeff_dict, self.wbasis, self.mode,
                                         self.nscales)
            return self.range.element(x)

    @property
    def adjoint(self):
        """The adjoint wavelet transform."""
        if self.is_orthogonal:
            return self.inverse
        else:
            # TODO: put adjoint here
            return super().adjoint

    @property
    def inverse(self):
        """The inverse wavelet transform."""
        return DiscreteWaveletTransform(dom=self.range, nscales=self.nscales,
                                        wbasis=self.wbasis, mode=self.mode)
